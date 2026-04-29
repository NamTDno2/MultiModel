# ==============================================================================
# FINAL SUMMARY — aggregate results from all methods
# Safe khi chỉ chạy Method 1 + 7 (hoặc bất kỳ tổ hợp nào)
# ==============================================================================

print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

def save_summary_csv(metrics, domain_metrics, method_keys, prefix):
    # Overall
    rows = []
    for key in method_keys:
        if key not in metrics: continue
        m = metrics[key]; cnt = m['count'] or 1
        rows.append({
            'method':   key,
            'r@1':      round(m['r1']  / cnt * 100, 4),
            'r@5':      round(m['r5']  / cnt * 100, 4),
            'r@10':     round(m['r10'] / cnt * 100, 4),
            'ndcg@1':   round(m['n1']  / cnt,       6),
            'ndcg@5':   round(m['n5']  / cnt,       6),
            'ndcg@10':  round(m['n10'] / cnt,       6),
            'count':    cnt,
        })
    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(os.path.join(WORKING_DIR, f"{prefix}_summary.csv"), index=False)

    # Per-domain
    dom_rows = []
    for domain in sorted(domain_metrics):
        dm  = domain_metrics[domain]
        row = {'domain': domain}
        for key in method_keys:
            m_   = dm.get(key, _init_metric()); cnt_ = m_['count'] or 1
            row[f'{key}_ndcg10'] = round(m_['n10'] / cnt_, 6)
            row[f'{key}_r10']    = round(m_['r10'] / cnt_ * 100, 4)
        dom_rows.append(row)
    pd.DataFrame(dom_rows).to_csv(
        os.path.join(WORKING_DIR, f"{prefix}_domain.csv"), index=False)

    print(f"✅ Saved: {prefix}_summary.csv and {prefix}_domain.csv")

# ---- Traditional (Method 1) ----
if 'trad_metrics' in dir() and trad_metrics:
    print_summary(trad_metrics, trad_domain_metrics, ['traditional'],
                  title="Traditional MaxSim")
    save_summary_csv(trad_metrics, trad_domain_metrics, ['traditional'], "traditional")

# ---- Hierarchical (Method 3) ----
if 'hier_metrics' in dir() and hier_metrics:
    print_summary(hier_metrics, hier_domain_metrics, METHOD_KEYS_HIER,
                  title="Hierarchical Ward Pooling")
    save_summary_csv(hier_metrics, hier_domain_metrics, METHOD_KEYS_HIER, "hierarchical")

# ---- Ours (Method 2) ----
if 'ours_metrics' in dir() and ours_metrics:
    print_summary(ours_metrics, ours_domain_metrics, ['traditional', 'trad_weighted'],
                  title="Ours — Baseline rows (add importance pkl for full ablation)")
    _ours_keys = ABLATION_KEYS_OURS if 'ABLATION_KEYS_OURS' in dir() else list(ours_metrics.keys())
    save_summary_csv(ours_metrics, ours_domain_metrics, _ours_keys, "ours_ablation")

# ---- Attention (Method 4) ----
if 'attn_metrics' in dir() and attn_metrics:
    _attn_keys = METHOD_KEYS_ATTN if 'METHOD_KEYS_ATTN' in dir() else list(attn_metrics.keys())
    print_summary(attn_metrics, attn_domain_metrics,
                  ['traditional'] + [f"attn_L1_r{int(r*100)}_trad" for r in TOPK_RATIOS],
                  title="Attention Score Pruning (L=1)")
    save_summary_csv(attn_metrics, attn_domain_metrics, _attn_keys, "attention_pruning")

# ---- Spherical KMeans (Method 5) ----
if 'kmeans_metrics' in dir() and kmeans_metrics:
    _km_keys = METHOD_KEYS_KMEANS if 'METHOD_KEYS_KMEANS' in dir() else list(kmeans_metrics.keys())
    print_summary(kmeans_metrics, kmeans_domain_metrics, _km_keys,
                  title="Spherical KMeans Pooling")
    save_summary_csv(kmeans_metrics, kmeans_domain_metrics, _km_keys, "spherical_kmeans")

# ---- Random Pruning (Method 6) ----
if 'rand_metrics' in dir() and rand_metrics:
    _rnd_keys = METHOD_KEYS_RAND if 'METHOD_KEYS_RAND' in dir() else list(rand_metrics.keys())
    print_summary(rand_metrics, rand_domain_metrics, _rnd_keys,
                  title="Random Token Pruning (Ablation Baseline)")
    save_summary_csv(rand_metrics, rand_domain_metrics, _rnd_keys, "random_pruning")

# ---- RVQ Compression (Method 7) ----
if 'rvq_metrics' in dir() and rvq_metrics:
    _rvq_keys = [cfg[2] for cfg in RVQ_CONFIGS] if 'RVQ_CONFIGS' in dir() else list(rvq_metrics.keys())
    print_summary(rvq_metrics, rvq_domain_metrics, _rvq_keys,
                  title="RVQ Compression")
    save_summary_csv(rvq_metrics, rvq_domain_metrics, _rvq_keys, "rvq_compression")

print("\n>>> All evaluations complete.")
