"""
Inject RVQ cells into METHODOLOGY ColPali ViDoRe.ipynb.

Chạy 1 lần trên local PC (KHÔNG cần GPU):
    python _inject_rvq_cells.py

Kết quả: notebook sẽ có thêm 2 cells mới (pip install + Method 7 RVQ)
         và FINAL SUMMARY cell được cập nhật để hiển thị RVQ kết quả.
"""

import json
import os
import sys

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(SCRIPT_DIR, "METHODOLOGY ColPali ViDoRe.ipynb")
RVQ_SOURCE    = os.path.join(SCRIPT_DIR, "_method7_rvq.py")

# ── Read notebook ────────────────────────────────────────────────────────────
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)
cells = nb["cells"]

# ── Check if already injected ────────────────────────────────────────────────
for cell in cells:
    src = "".join(cell.get("source", []))
    if "METHOD 7" in src and "Residual Vector Quantization" in src:
        print("⚠️  RVQ cells already exist in notebook. Skipping.")
        sys.exit(0)

# ── Find FINAL SUMMARY cell ─────────────────────────────────────────────────
final_idx = None
for i, cell in enumerate(cells):
    src = "".join(cell.get("source", []))
    if "FINAL SUMMARY" in src and "aggregate results" in src:
        final_idx = i
        break

if final_idx is None:
    print("❌ Cannot find FINAL SUMMARY cell. Add cells manually.")
    sys.exit(1)

print(f"Found FINAL SUMMARY at cell index {final_idx}")

# ── Helper ───────────────────────────────────────────────────────────────────
def make_cell(source_lines, cell_id):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {"vscode": {"languageId": "plaintext"}},
        "outputs": [],
        "source": source_lines,
    }

# ── Create pip install cell ──────────────────────────────────────────────────
pip_cell = make_cell([
    "# Install vector-quantize-pytorch (required for METHOD 7)\n",
    "!pip install -q vector-quantize-pytorch\n",
    'print("✅ vector-quantize-pytorch installed.")\n',
], "rvq_pip_install")

# ── Create Method 7 cell from source file ────────────────────────────────────
with open(RVQ_SOURCE, "r", encoding="utf-8") as f:
    rvq_lines = f.readlines()
rvq_cell = make_cell(rvq_lines, "rvq_method7")

# ── Insert cells before FINAL SUMMARY ────────────────────────────────────────
cells.insert(final_idx, rvq_cell)
cells.insert(final_idx, pip_cell)
final_idx += 2  # shifted

# ── Update FINAL SUMMARY to include RVQ ──────────────────────────────────────
final_cell = cells[final_idx]
src_lines = final_cell["source"]
final_src = "".join(src_lines)

if "rvq_metrics" not in final_src:
    # 1. Add RVQ print_summary before "Save master summary"
    insert_before = None
    for li, line in enumerate(src_lines):
        if "Save master summary" in line:
            insert_before = li - 1
            break

    if insert_before is not None:
        rvq_print = [
            "\n",
            "# ---- RVQ Compression ----\n",
            "if 'rvq_metrics' in dir() and rvq_metrics:\n",
            "    _rvq_keys = [cfg[2] for cfg in RVQ_CONFIGS] if 'RVQ_CONFIGS' in dir() else list(rvq_metrics.keys())\n",
            "    print_summary(rvq_metrics, rvq_domain_metrics, _rvq_keys,\n",
            '                  title="RVQ Compression")\n',
            "\n",
        ]
        for offset, line in enumerate(rvq_print):
            src_lines.insert(insert_before + offset, line)

    # 2. Add RVQ save_summary_csv before "All evaluations complete"
    save_before = None
    for li, line in enumerate(src_lines):
        if "All evaluations complete" in line:
            save_before = li
            break

    if save_before is not None:
        rvq_save = [
            "if 'rvq_metrics' in dir() and rvq_metrics:\n",
            "    _rvq_keys = [cfg[2] for cfg in RVQ_CONFIGS] if 'RVQ_CONFIGS' in dir() else list(rvq_metrics.keys())\n",
            '    save_summary_csv(rvq_metrics, rvq_domain_metrics, _rvq_keys, "rvq_compression")\n',
            "\n",
        ]
        for offset, line in enumerate(rvq_save):
            src_lines.insert(save_before + offset, line)

    final_cell["source"] = src_lines

# ── Write notebook ───────────────────────────────────────────────────────────
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Done! Injected 2 cells into {os.path.basename(NOTEBOOK_PATH)}")
print(f"   Total cells: {len(cells)}")
print(f"   Position: pip install → cell {final_idx-2}, Method 7 → cell {final_idx-1}")
