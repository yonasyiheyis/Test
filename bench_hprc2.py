#!/usr/bin/env python3
"""
Benchmark script for HPRC++

Usage:
  python bench_hprc2.py images_folder/

It will:
  - For each image in the folder:
      * Save as PNG (optimized) to measure PNG size
      * Save as HPRC++ (.hprc2)
      * Compare sizes
  - Print a summary table
"""

import os
import sys
import tempfile
import numpy as np
from PIL import Image
import subprocess

import hprc2  # assumes hprc2.py is in the same folder

def get_png_size(img: Image.Image) -> int:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    img.save(path, format="PNG", optimize=True)
    size = os.path.getsize(path)
    os.remove(path)
    return size

def get_hprc2_size(img: Image.Image) -> int:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    blob = hprc2.encode_hprc2_select(arr)
    return len(blob)

def main():
    if len(sys.argv) < 2:
        print("Usage: python bench_hprc2.py images_folder/")
        return

    folder = sys.argv[1]
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".gif",".tiff"))]

    if not files:
        print("No images found in", folder)
        return

    results = []
    for fname in files:
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path)
        except Exception as e:
            print("Skipping", fname, ":", e)
            continue

        png_size = get_png_size(img)
        hprc2_size = get_hprc2_size(img)

        ratio = hprc2_size / png_size if png_size > 0 else float("inf")
        results.append((fname, png_size, hprc2_size, ratio))
        print(f"{fname:20} PNG={png_size:8d}  HPRC2={hprc2_size:8d}  ratio={ratio:6.2f}")

    print("\nSummary:")
    total_png = sum(r[1] for r in results)
    total_hprc2 = sum(r[2] for r in results)
    print(f"Total PNG size   : {total_png} bytes")
    print(f"Total HPRC2 size : {total_hprc2} bytes")
    print(f"Overall ratio    : {total_hprc2/total_png:6.2f}")

if __name__ == "__main__":
    main()