#!/usr/bin/env python3
"""
Benchmark: PNG vs HPRC++ (.hprc2) vs WebP-lossless vs JPEG XL (if cjxl present)

Usage:
  python bench_hprc2_plus.py /path/to/images [--csv out.csv]

Notes:
  - Expects hprc2.py in the same directory (importable as `hprc2`).
  - WebP-lossless is encoded via Pillow if WebP is available.
  - JPEG XL is encoded via external `cjxl` tool (skipped if not found).
"""

import argparse
import os
import sys
import tempfile
import shutil
import subprocess
import csv

import numpy as np
from PIL import Image

import hprc2  # make sure hprc2.py is in the same folder

def img_iter(folder):
    exts = (".png",".jpg",".jpeg",".bmp",".gif",".tiff",".webp")
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(exts):
            yield name, os.path.join(folder, name)

def png_size(img: Image.Image) -> int:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        img.save(tmp.name, format="PNG", optimize=True)
        return os.path.getsize(tmp.name)
    finally:
        os.unlink(tmp.name)

def webp_lossless_size(img: Image.Image) -> int | None:
    # Pillow must have WebP; else this raises
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".webp", delete=False)
        tmp.close()
        try:
            img.save(tmp.name, format="WEBP", lossless=True, quality=100, method=6)
            return os.path.getsize(tmp.name)
        finally:
            os.unlink(tmp.name)
    except Exception:
        return None

def jxl_size(img: Image.Image) -> int | None:
    # Requires cjxl (encoder). If missing, skip.
    if shutil.which("cjxl") is None:
        return None
    # Write input PNG temp, run cjxl, read file size
    pin = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pout = tempfile.NamedTemporaryFile(suffix=".jxl", delete=False)
    pin.close(); pout.close()
    try:
        img.save(pin.name, format="PNG", optimize=False)
        # lossless mode: --lossless (or -e 9 -d 0); keep defaults that favor lossless
        # Also use effort 7 by default (balance speed/ratio)
        cmd = ["cjxl", pin.name, pout.name, "--lossless", "-e", "7"]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return os.path.getsize(pout.name)
    except Exception:
        return None
    finally:
        for p in (pin.name, pout.name):
            if os.path.exists(p):
                os.unlink(p)

def hprc2_size(img: Image.Image) -> int:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    blob = hprc2.encode_hprc2_select(arr)
    return len(blob)

def main():
    ap = argparse.ArgumentParser(description="Benchmark PNG vs HPRC++ vs WebP-lossless vs JPEG XL")
    ap.add_argument("folder", help="folder containing images to test")
    ap.add_argument("--csv", help="optional path to write a CSV summary")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        print("Not a directory:", args.folder)
        sys.exit(1)

    rows = []
    have_webp = False
    have_jxl = shutil.which("cjxl") is not None

    print(f"JPEG XL (cjxl) available: {'YES' if have_jxl else 'NO'}")
    print()

    header = ["file", "png", "hprc2", "webp_lossless", "jxl", "hprc2/png", "webp/png", "jxl/png"]
    print(f"{'file':30} {'png':>10} {'hprc2':>10} {'webp':>10} {'jxl':>10}  {'hprc2/png':>10} {'webp/png':>10} {'jxl/png':>10}")

    total_png = total_hprc2 = total_webp = total_jxl = 0
    count_webp = count_jxl = 0

    for name, path in img_iter(args.folder):
        try:
            img = Image.open(path)
        except Exception as e:
            print(f"(skip) {name}: {e}")
            continue

        psize = png_size(img)
        hsize = hprc2_size(img)
        wsize = webp_lossless_size(img)
        jsize = jxl_size(img)

        r_h = (hsize/psize) if psize else float("inf")
        r_w = (wsize/psize) if (psize and wsize is not None) else None
        r_j = (jsize/psize) if (psize and jsize is not None) else None

        total_png += psize
        total_hprc2 += hsize
        if wsize is not None:
            total_webp += wsize; count_webp += 1
        if jsize is not None:
            total_jxl += jsize; count_jxl += 1

        print(f"{name[:30]:30} {psize:10d} {hsize:10d} "
              f"{(wsize if wsize is not None else -1):10d} "
              f"{(jsize if jsize is not None else -1):10d}  "
              f"{r_h:10.3f} "
              f"{(r_w if r_w is not None else float('nan')):10.3f} "
              f"{(r_j if r_j is not None else float('nan')):10.3f}")

        rows.append({
            "file": name,
            "png": psize,
            "hprc2": hsize,
            "webp_lossless": wsize,
            "jxl": jsize,
            "hprc2/png": r_h,
            "webp/png": r_w,
            "jxl/png": r_j
        })

    print("\nTotals:")
    print(f"PNG total       : {total_png} bytes")
    print(f"HPRC2 total     : {total_hprc2} bytes (ratio {total_hprc2/total_png:.3f})")
    if count_webp > 0:
        print(f"WebP-lossless   : {total_webp} bytes over {count_webp} imgs (ratio {total_webp/total_png:.3f})")
    else:
        print("WebP-lossless   : (not available / failed)")
    if count_jxl > 0:
        print(f"JPEG XL         : {total_jxl} bytes over {count_jxl} imgs (ratio {total_jxl/total_png:.3f})")
    else:
        print("JPEG XL         : (cjxl not found or failed)")

    if args.csv:
        with open(args.csv, "w", newline="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                out = {k: r.get(k, None) for k in header}
                w.writerow(out)
        print(f"\nWrote CSV -> {args.csv}")

if __name__ == "__main__":
    main()