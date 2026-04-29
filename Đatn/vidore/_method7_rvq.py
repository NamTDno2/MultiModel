# ==============================================================================
# METHOD 7 — Residual Vector Quantization (RVQ) Compression
#
# ĐẶT SAU cell Method 1 (hoặc Method 6), TRƯỚC cell FINAL SUMMARY
# KHÔNG cần chạy ở session riêng — chạy cùng session với các method khác.
# KHÔNG cần cell pip install riêng — tự install từ dataset offline.
#
# Pipeline (tất cả trong 1 cell):
#   Phase 1: Train RVQ codebooks offline trên sampled patch embeddings (EMA).
#            → Chỉ dùng ~100MB VRAM, mất ~2 phút/config.
#   Phase 2: Quantize toàn bộ index → lưu chỉ uint8 codebook indices.
#            → Giảm RAM 64× (512 bytes/patch → 8 bytes/patch).
#   Phase 3: Score bằng Asymmetric Distance Computation (ADC).
#            → Build LUT query×centroid, rồi gather+sum. Không decode float32.
#
# Tuned cho RTX Pro 6000 (48GB VRAM, 30GB RAM).
# ==============================================================================

# ── Install vector-quantize-pytorch từ dataset offline ────────────────────────
import subprocess, os
_rvq_wheel_dir = "/kaggle/input/datasets/thinam4/rvq-wheels/rvq_wheels"
if os.path.isdir(_rvq_wheel_dir):
    subprocess.run([
        "pip", "install", "--quiet",
        "--no-index",
        "--find-links", _rvq_wheel_dir,
        "vector-quantize-pytorch",
    ], check=True)
    print(f"✅ Installed vector-quantize-pytorch from {_rvq_wheel_dir}")
else:
    print(f"⚠️  Wheel dir not found: {_rvq_wheel_dir}, assuming already installed")

import torch
import torch.nn.functional as F
import numpy as np
import time
import gc
from tqdm.notebook import tqdm
from vector_quantize_pytorch import ResidualVQ

print(">>> METHOD 7: Residual Vector Quantization (RVQ) Compression")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Ensure page embeddings are in memory ─────────────────────────────────────
all_page_embeddings = ensure_all_page_embeddings_loaded(
    INDEX_PKL_PATH,
    globals().get("all_page_embeddings", None),
)
n_pages_total = len(all_page_embeddings)
EMB_DIM = all_page_embeddings[0].shape[-1]   # 128 for ColPali
print(f"Pages: {n_pages_total}, Dim: {EMB_DIM}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — tuned cho RTX Pro 6000 (48GB VRAM)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Mỗi tuple: (num_quantizers, codebook_size, label)
#  Storage/patch = NQ × (1 byte nếu CB≤256, 2 bytes nếu CB>256)
#
#  Ablation study: NQ=8, 16, 32
#    NQ=8:   balanced (64×)
#    NQ=16:  conservative (32×)
#    NQ=32:  near-lossless (16×) — dự kiến R@10 ≈ 46-47%
#
RVQ_CONFIGS = [
    (16, 256, "RVQ_NQ16_CB256"),   # 16 bytes/patch, 32× compression
    (32, 256, "RVQ_NQ32_CB256"),   # 32 bytes/patch, 16× compression
]

# Codebook training config
RVQ_TRAINING_SAMPLES    = 200_000   # patch vectors sampled cho EMA training
RVQ_TRAINING_EPOCHS     = 30        # epochs qua training data
RVQ_TRAINING_BATCH_SIZE = 4096      # batch size mỗi forward pass training

# Quantization config
RVQ_QUANTIZE_BATCH_SIZE = 8192      # batch size khi quantize index

# ADC scoring config
ADC_CHUNK_SIZE = 1024

# Re-ranking config
# ADC lấy top RERANK_TOP_K candidates → exact float32 MaxSim re-rank → top-10
RERARNK_TOP_K = 100

M7_MEASURE_LATENCY = True


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _sample_patch_embeddings(embeddings_list, n_samples, seed=42):
    """Sample n_samples individual patch vectors from the full corpus."""
    rng = np.random.default_rng(seed)
    all_lengths = [e.shape[0] for e in embeddings_list]
    total_patches = sum(all_lengths)
    n_take = min(n_samples, total_patches)

    flat_idx = rng.choice(total_patches, size=n_take, replace=False)
    flat_idx.sort()

    result = np.empty((n_take, embeddings_list[0].shape[1]), dtype=np.float32)
    cumsum = 0
    out_pos = 0
    flat_iter = iter(flat_idx)
    next_fi = next(flat_iter, None)

    for page_i, L in enumerate(all_lengths):
        page_end = cumsum + L
        while next_fi is not None and next_fi < page_end:
            local = next_fi - cumsum
            result[out_pos] = embeddings_list[page_i][local].astype(np.float32)
            out_pos += 1
            next_fi = next(flat_iter, None)
        cumsum = page_end
        if next_fi is None:
            break

    return torch.from_numpy(result[:out_pos])


def _extract_codebooks(rvq_model, NQ, device):
    """
    Extract codebook centroids from trained ResidualVQ model.
    Handles multiple API versions of vector-quantize-pytorch.
    """
    codebooks = []
    for q_idx in range(NQ):
        vq_layer = rvq_model.layers[q_idx]
        cb = None

        # Method 1: _codebook.embed (most common)
        if cb is None:
            try:
                embed = vq_layer._codebook.embed
                if embed.ndim == 3:
                    cb = embed[0]        # (num_codebooks, CB, D) → (CB, D)
                elif embed.ndim == 2:
                    cb = embed           # (CB, D)
            except (AttributeError, IndexError):
                pass

        # Method 2: codebook (some versions expose directly)
        if cb is None:
            try:
                embed = vq_layer.codebook
                if embed.ndim == 3:
                    cb = embed[0]
                elif embed.ndim == 2:
                    cb = embed
            except AttributeError:
                pass

        # Method 3: _codebook.embed_avg (EMA codebook)
        if cb is None:
            try:
                embed = vq_layer._codebook.embed_avg
                if embed.ndim == 3:
                    cb = embed[0]
                elif embed.ndim == 2:
                    cb = embed
            except AttributeError:
                pass

        if cb is None:
            raise RuntimeError(
                f"Cannot extract codebook from VQ layer {q_idx}. "
                f"Available attrs: {[a for a in dir(vq_layer) if not a.startswith('__')]}"
            )

        codebooks.append(cb.detach().to(device).float())

    return codebooks


@torch.no_grad()
def adc_maxsim_chunked(q_norm, doc_indices, doc_mask, codebooks, chunk_size=1024):
    """
    Asymmetric Distance Computation (ADC) MaxSim — memory-safe chunked version.

    Thay vì decode documents → float32 rồi matmul (tốn VRAM),
    ta tính trước LUT (query × codebook centroids) rồi dùng gather
    trên stored indices để tính MaxSim.

    Args:
        q_norm     : (N_q, D)   — L2-normalized query token vectors, GPU
        doc_indices: (n_docs, max_len, NQ)  — uint8/int16 codebook indices, GPU
        doc_mask   : (n_docs, max_len)      — bool padding mask, GPU
        codebooks  : list[Tensor(CB, D)]    — centroids per quantizer, GPU
        chunk_size : int                    — docs per chunk (tune for VRAM)

    Returns:
        scores: (n_docs,) — MaxSim scores
    """
    n_docs = doc_indices.shape[0]
    NQ = doc_indices.shape[2]
    N_q = q_norm.shape[0]

    # Step 1: Precompute LUT — (NQ, N_q, CB_size)
    # LUT[q][t, c] = dot(query_token_t, centroid_c) for quantizer q
    lut_list = []
    for q_idx in range(NQ):
        cb = codebooks[q_idx]   # (CB_size, D)
        lut_list.append(torch.mm(q_norm, cb.t()))   # (N_q, CB_size)

    all_scores = torch.zeros(n_docs, device=q_norm.device, dtype=torch.float32)

    # Step 2: Process docs in chunks
    for start in range(0, n_docs, chunk_size):
        end = min(start + chunk_size, n_docs)
        chunk_idx  = doc_indices[start:end]    # (cs, max_len, NQ)
        chunk_mask = doc_mask[start:end]        # (cs, max_len)
        cs = end - start
        max_len = chunk_idx.shape[1]

        # Accumulate approx dot product: sum_q LUT[q][t, idx[d,p,q]]
        sim = torch.zeros(N_q, cs, max_len, device=q_norm.device, dtype=torch.float32)
        for q_idx in range(NQ):
            lut_q = lut_list[q_idx]                          # (N_q, CB_size)
            idx_q = chunk_idx[:, :, q_idx].long()            # (cs, max_len)
            idx_exp = idx_q.unsqueeze(0).expand(N_q, -1, -1) # (N_q, cs, max_len)
            gathered = torch.gather(
                lut_q.unsqueeze(1).expand(-1, cs, -1),       # (N_q, cs, CB_size)
                dim=2,
                index=idx_exp,                                # (N_q, cs, max_len)
            )
            sim += gathered

        # Mask padding, then MaxSim: max over patches, sum over query tokens
        sim.masked_fill_(~chunk_mask.unsqueeze(0), float('-inf'))
        all_scores[start:end] = sim.max(dim=-1).values.sum(dim=0)

    return all_scores


@torch.no_grad()
def rerank_exact_maxsim(q_norm, candidate_indices, all_page_embeddings, device):
    """
    Re-rank candidates bằng exact float32 MaxSim.

    Args:
        q_norm            : (N_q, D) query tokens, L2-normalized, GPU
        candidate_indices : list[int] — top-K doc indices từ ADC
        all_page_embeddings : list[ndarray] — original float32 embeddings
        device            : str

    Returns:
        reranked_top10 : list[int] — top-10 doc indices sau re-rank
    """
    K = len(candidate_indices)
    if K == 0:
        return []

    # Build mini doc matrix chỉ cho K candidates
    arrays = []
    for idx in candidate_indices:
        emb = all_page_embeddings[idx]
        arrays.append(torch.from_numpy(emb.astype(np.float32)))

    max_len = max(a.shape[0] for a in arrays)
    D = arrays[0].shape[1]
    mini_mat  = torch.zeros(K, max_len, D, dtype=torch.float32)
    mini_mask = torch.zeros(K, max_len, dtype=torch.bool)

    for i, a in enumerate(arrays):
        L = a.shape[0]
        mini_mat[i, :L]  = F.normalize(a, dim=-1)
        mini_mask[i, :L] = True

    mini_mat  = mini_mat.to(device)
    mini_mask = mini_mask.to(device)

    # Exact MaxSim
    sim = torch.einsum('qd,nld->qnl', q_norm, mini_mat)   # (N_q, K, max_len)
    sim.masked_fill_(~mini_mask.unsqueeze(0), float('-inf'))
    scores = sim.max(dim=-1).values.sum(dim=0)              # (K,)

    # Map back to original doc indices
    top_k = min(10, K)
    local_top = torch.topk(scores, top_k).indices.cpu().tolist()
    return [candidate_indices[i] for i in local_top]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RVQ SWEEP
# ══════════════════════════════════════════════════════════════════════════════

rvq_metrics        = {}
rvq_domain_metrics = {}
rvq_query_rows     = []
rvq_latency        = LatencyTracker("RVQ ADC")
rvq_compression_info = []

# ── Step 0: Sample training data once (reused across all configs) ────────────
print(f"Sampling {RVQ_TRAINING_SAMPLES:,} patch vectors for codebook training...")
train_data = _sample_patch_embeddings(all_page_embeddings, RVQ_TRAINING_SAMPLES)
train_data = F.normalize(train_data, dim=-1)
print(f"  Training data: {train_data.shape} ({train_data.nbytes/1e6:.0f} MB)")

# ── Precompute doc lengths (reused across all configs) ────────────────────────
doc_lengths = [e.shape[0] for e in all_page_embeddings]
max_doc_len = max(doc_lengths)
total_patches = sum(doc_lengths)
bytes_per_patch_f32 = EMB_DIM * 4  # 512 bytes for dim=128
print(f"  max_doc_len: {max_doc_len}, total_patches: {total_patches:,}")

# ══════════════════════════════════════════════════════════════════════════════
for cfg_idx, (NQ, CB_SIZE, cfg_label) in enumerate(RVQ_CONFIGS):
    bytes_per_patch_rvq = NQ * (1 if CB_SIZE <= 256 else 2)
    compression_ratio = bytes_per_patch_f32 / bytes_per_patch_rvq
    mem_rvq_mb = total_patches * bytes_per_patch_rvq / 1e6
    mem_f32_mb = total_patches * bytes_per_patch_f32 / 1e6

    print(f"\n{'='*70}")
    print(f"[{cfg_idx+1}/{len(RVQ_CONFIGS)}] {cfg_label}")
    print(f"  NQ={NQ}, CB={CB_SIZE} → {bytes_per_patch_rvq} B/patch → {compression_ratio:.0f}× compression")
    print(f"  Index size: {mem_rvq_mb:.1f} MB (RVQ) vs {mem_f32_mb:.1f} MB (FP32)")
    print(f"{'='*70}")

    rvq_compression_info.append({
        'config': cfg_label, 'num_quantizers': NQ,
        'codebook_size': CB_SIZE, 'bytes_per_patch': bytes_per_patch_rvq,
        'compression_ratio': compression_ratio,
        'index_mb_rvq': round(mem_rvq_mb, 1),
        'index_mb_f32': round(mem_f32_mb, 1),
    })

    # ══ Phase 1: Train codebooks ══════════════════════════════════════════
    print("  Phase 1: Training RVQ codebooks...")
    t_train_start = time.time()

    rvq_model = ResidualVQ(
        dim=EMB_DIM,
        num_quantizers=NQ,
        codebook_size=CB_SIZE,
        kmeans_init=True,
        threshold_ema_dead_code=2,
    ).to(device)
    rvq_model.train()

    n_train = train_data.shape[0]
    for epoch in range(RVQ_TRAINING_EPOCHS):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, RVQ_TRAINING_BATCH_SIZE):
            batch = train_data[perm[i:i+RVQ_TRAINING_BATCH_SIZE]].to(device)
            # ResidualVQ expects (batch, seq_len, dim) — treat each patch as seq=1
            _, _, commit_loss = rvq_model(batch.unsqueeze(1))
            epoch_loss += commit_loss.sum().item()
            n_batches += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{RVQ_TRAINING_EPOCHS}: "
                  f"commit_loss={epoch_loss/max(n_batches,1):.6f}")

    rvq_model.eval()
    t_train = time.time() - t_train_start
    print(f"  Codebooks frozen. Training took {t_train:.1f}s")

    # Extract codebook centroids
    codebooks = _extract_codebooks(rvq_model, NQ, device)
    print(f"  Codebooks: {len(codebooks)} × {codebooks[0].shape}")

    # ══ Phase 2: Quantize index ═══════════════════════════════════════════
    print("  Phase 2: Quantizing index...")
    t_quant_start = time.time()
    idx_dtype = np.uint8 if CB_SIZE <= 256 else np.uint16

    quantized_index = np.zeros((n_pages_total, max_doc_len, NQ), dtype=idx_dtype)
    quantized_mask  = np.zeros((n_pages_total, max_doc_len), dtype=bool)

    for page_i in tqdm(range(n_pages_total), desc="  Quantizing", leave=False):
        emb = all_page_embeddings[page_i]
        L = emb.shape[0]
        quantized_mask[page_i, :L] = True

        page_t = torch.from_numpy(emb.astype(np.float32)).to(device)
        page_t = F.normalize(page_t, dim=-1)

        for s in range(0, L, RVQ_QUANTIZE_BATCH_SIZE):
            e = min(s + RVQ_QUANTIZE_BATCH_SIZE, L)
            with torch.no_grad():
                _, indices, _ = rvq_model(page_t[s:e].unsqueeze(0))
            quantized_index[page_i, s:e, :] = indices[0].cpu().numpy().astype(idx_dtype)

    doc_indices_t = torch.from_numpy(quantized_index).to(device)
    doc_mask_t    = torch.from_numpy(quantized_mask).to(device)
    t_quant = time.time() - t_quant_start
    print(f"  Quantized {n_pages_total} pages in {t_quant:.1f}s")
    print(f"  GPU index: {doc_indices_t.shape}, {doc_indices_t.element_size()*doc_indices_t.nelement()/1e6:.0f} MB")

    # Free numpy arrays immediately (GPU copy is enough)
    del quantized_index, quantized_mask
    gc.collect()

    # ══ Phase 3: Evaluate (ADC-only + ADC+Rerank) ═════════════════════════
    rerank_label = f"{cfg_label}+rerank{RERARNK_TOP_K}"
    print(f"  Phase 3: ADC MaxSim + Re-ranking (top-{RERARNK_TOP_K})...")
    t_eval_start = time.time()

    for q_idx, item in tqdm(enumerate(qa_pairs), total=len(qa_pairs),
                            desc=f"  Eval {cfg_label}", leave=False):
        question = item['question']
        gt_set   = item.get('gt_relevance', item['gt_embed_indices'])
        domain   = item['domain']

        # Encode query live
        q_inputs = query_processor.process_queries([question]).to(device)
        if 'token_type_ids' not in q_inputs and 'input_ids' in q_inputs:
            q_inputs['token_type_ids'] = torch.zeros_like(q_inputs['input_ids'])

        with torch.no_grad():
            q_proj = query_model(**q_inputs)

        if hasattr(q_proj, 'last_hidden_state') and q_proj.last_hidden_state is not None:
            q_proj = q_proj.last_hidden_state
        elif isinstance(q_proj, (tuple, list)) and len(q_proj) > 0:
            q_proj = q_proj[0]

        attn_mask = q_inputs['attention_mask'][0]
        trad_idx  = torch.where(attn_mask > 0)[0]
        q_emb     = q_proj[0][trad_idx].float()
        q_norm    = F.normalize(q_emb, dim=-1)

        # ── Stage 1: ADC → top-K candidates ──────────────────────────────
        if M7_MEASURE_LATENCY and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        scores = adc_maxsim_chunked(
            q_norm, doc_indices_t, doc_mask_t, codebooks,
            chunk_size=ADC_CHUNK_SIZE,
        )

        # ADC-only top-10 (for comparison)
        top10_adc = torch.topk(scores, min(10, n_pages_total)).indices.cpu().tolist()

        if M7_MEASURE_LATENCY and torch.cuda.is_available():
            torch.cuda.synchronize()
        score_ms_adc = (time.perf_counter() - t0) * 1000.0

        # ── Stage 2: Re-rank top-K with exact float32 MaxSim ─────────────
        top_k_candidates = torch.topk(scores, min(RERARNK_TOP_K, n_pages_total)).indices.cpu().tolist()

        if M7_MEASURE_LATENCY and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        top10_reranked = rerank_exact_maxsim(
            q_norm, top_k_candidates, all_page_embeddings, device
        )

        if M7_MEASURE_LATENCY and torch.cuda.is_available():
            torch.cuda.synchronize()
        score_ms_rerank = (time.perf_counter() - t1) * 1000.0

        rvq_latency.add_ratio(NQ, score_ms_adc)

        # ── Record ADC-only metrics ──────────────────────────────────────
        m_adc = hit_metrics(top10_adc, gt_set)
        record(rvq_metrics, rvq_domain_metrics, cfg_label, m_adc, domain)

        # ── Record reranked metrics ──────────────────────────────────────
        m_rr = hit_metrics(top10_reranked, gt_set)
        record(rvq_metrics, rvq_domain_metrics, rerank_label, m_rr, domain)

        rvq_query_rows.append({
            'query_id': q_idx, 'doc_name': item['doc_name'],
            'domain': domain, 'question': question,
            'rvq_config': cfg_label,
            # ADC-only
            'adc_r@1': m_adc['r1'], 'adc_r@5': m_adc['r5'], 'adc_r@10': m_adc['r10'],
            'adc_ndcg@10': round(m_adc['n10'], 4),
            # Reranked
            'rr_r@1': m_rr['r1'], 'rr_r@5': m_rr['r5'], 'rr_r@10': m_rr['r10'],
            'rr_ndcg@10': round(m_rr['n10'], 4),
            'adc_ms': round(score_ms_adc, 3),
            'rerank_ms': round(score_ms_rerank, 3),
        })

    t_eval = time.time() - t_eval_start
    print(f"  Eval done: {t_eval:.1f}s ({t_eval/len(qa_pairs)*1000:.1f} ms/query)")

    # Free GPU memory for this config
    del doc_indices_t, doc_mask_t, rvq_model, codebooks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

del train_data
gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

# Build method keys: both ADC-only and reranked
METHOD_KEYS_RVQ = []
for cfg in RVQ_CONFIGS:
    METHOD_KEYS_RVQ.append(cfg[2])
    METHOD_KEYS_RVQ.append(f"{cfg[2]}+rerank{RERARNK_TOP_K}")

print_summary(rvq_metrics, rvq_domain_metrics, METHOD_KEYS_RVQ,
              title="RVQ Compression Results (ADC-only vs ADC+Rerank)")

# Compression vs Recall table
print(f"\n{'='*80}")
print(f"RVQ Compression vs Retrieval Quality (Rerank top-{RERARNK_TOP_K})")
print(f"{'='*80}")
print(f"{'Config':<30} {'NQ':>4} {'B/patch':>8} {'Ratio':>7} "
      f"{'R@10':>8} {'nDCG@10':>9}")
print(f"{'-'*72}")

# FP32 baseline
trad_m = trad_metrics.get('traditional', _init_metric())
trad_cnt = max(trad_m.get('count', 1), 1)
print(f"{'FP32 (baseline)':<30} {'—':>4} {bytes_per_patch_f32:>8} {'1.0×':>7} "
      f"{trad_m.get('r10',0)/trad_cnt*100:>7.2f}% "
      f"{trad_m.get('n10',0)/trad_cnt:>8.4f}")
print(f"{'-'*72}")

for info in rvq_compression_info:
    cfg_key = info['config']
    rr_key  = f"{cfg_key}+rerank{RERARNK_TOP_K}"

    # ADC-only
    rm = rvq_metrics.get(cfg_key, _init_metric())
    cnt = max(rm.get('count', 1), 1)
    print(f"{cfg_key:<30} {info['num_quantizers']:>4} "
          f"{info['bytes_per_patch']:>8} {info['compression_ratio']:>6.0f}× "
          f"{rm.get('r10',0)/cnt*100:>7.2f}% "
          f"{rm.get('n10',0)/cnt:>8.4f}")

    # Reranked
    rm_rr = rvq_metrics.get(rr_key, _init_metric())
    cnt_rr = max(rm_rr.get('count', 1), 1)
    print(f"  {'+ rerank':<28} {'':>4} "
          f"{'':>8} {'':>7} "
          f"{rm_rr.get('r10',0)/cnt_rr*100:>7.2f}% "
          f"{rm_rr.get('n10',0)/cnt_rr:>8.4f}")

if M7_MEASURE_LATENCY:
    rvq_latency.report()

pd.DataFrame(rvq_query_rows).to_csv(
    os.path.join(WORKING_DIR, "rvq_queries.csv"), index=False)
pd.DataFrame(rvq_compression_info).to_csv(
    os.path.join(WORKING_DIR, "rvq_compression_info.csv"), index=False)
print("\n✅ Saved: rvq_queries.csv, rvq_compression_info.csv")



                