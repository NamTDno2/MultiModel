# ==============================================================================
# CELL 2/2 — Install RVQ từ dataset offline (chạy ở session GPU, KHÔNG INTERNET)
#
# Yêu cầu: đã add dataset "rvq-wheels" vào notebook
# Path mặc định: /kaggle/input/rvq-wheels/
# ==============================================================================

import subprocess
import os

# ── Tìm folder chứa .whl files ───────────────────────────────────────────────
# Kaggle dataset path có thể khác tùy username
CANDIDATES = [
    "/kaggle/input/rvq-wheels",
    "/kaggle/input/rvq-wheels/rvq_wheels",
    "/kaggle/input/datasets/rvq-wheels",
]

WHEEL_DIR = None
for path in CANDIDATES:
    if os.path.isdir(path):
        whl_files = [f for f in os.listdir(path) if f.endswith(('.whl', '.tar.gz'))]
        if whl_files:
            WHEEL_DIR = path
            break

if WHEEL_DIR is None:
    # Fallback: search all input directories
    for root, dirs, files in os.walk("/kaggle/input"):
        whl_files = [f for f in files if f.endswith(('.whl', '.tar.gz'))]
        if whl_files:
            WHEEL_DIR = root
            break

if WHEEL_DIR is None:
    raise FileNotFoundError(
        "Không tìm thấy folder chứa .whl files. "
        "Hãy add dataset 'rvq-wheels' vào notebook."
    )

print(f"Found wheels in: {WHEEL_DIR}")
whl_files = [f for f in os.listdir(WHEEL_DIR) if f.endswith(('.whl', '.tar.gz'))]
print(f"  {len(whl_files)} packages found")

# ── Install offline (không cần internet) ──────────────────────────────────────
subprocess.run([
    "pip", "install",
    "--no-index",                    # không tải từ PyPI
    "--find-links", WHEEL_DIR,       # tìm packages trong folder local
    "vector-quantize-pytorch",
], check=True)

# ── Verify ────────────────────────────────────────────────────────────────────
from vector_quantize_pytorch import ResidualVQ
print("\n✅ vector-quantize-pytorch installed successfully!")
print(f"   ResidualVQ ready: {ResidualVQ}")
