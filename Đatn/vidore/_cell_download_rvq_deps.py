# ==============================================================================
# CELL 1/2 — Download RVQ dependencies (chạy ở session CÓ INTERNET)
#
# Quy trình:
#   1. Tạo notebook MỚI trên Kaggle, BẬT INTERNET
#   2. Paste cell này vào, chạy
#   3. Output folder /kaggle/working/rvq_wheels/ sẽ chứa tất cả .whl files
#   4. Tạo Dataset mới trên Kaggle, upload folder rvq_wheels
#   5. Ở notebook GPU (không internet), add dataset đó vào rồi chạy cell install
# ==============================================================================

import subprocess
import os

WHEEL_DIR = "/kaggle/working/rvq_wheels"
os.makedirs(WHEEL_DIR, exist_ok=True)

# Download vector-quantize-pytorch + tất cả dependencies vào folder
subprocess.run([
    "pip", "download",
    "vector-quantize-pytorch",
    "--dest", WHEEL_DIR,
    "--no-cache-dir",
], check=True)

# Liệt kê các file đã download
files = sorted(os.listdir(WHEEL_DIR))
print(f"\n✅ Downloaded {len(files)} packages to {WHEEL_DIR}:")
for f in files:
    size_mb = os.path.getsize(os.path.join(WHEEL_DIR, f)) / 1e6
    print(f"  {f}  ({size_mb:.1f} MB)")

print(f"\n>>> Bây giờ:")
print(f"    1. Click 'Save Version' (Quick Save)")
print(f"    2. Vào Output tab, tạo Dataset từ folder rvq_wheels")
print(f"    3. Đặt tên dataset: rvq-wheels")
