import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_cache

for name in ("train", "val", "test"):
    print(f"building {name} cache...", flush=True)
    build_cache(os.path.join("C:/Projects/LipReader/training/data", f"{name}.txt"), num_workers=16)
    print(f"done {name}", flush=True)
print("ALL_CACHE_DONE")
