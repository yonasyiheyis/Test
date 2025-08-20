#!/usr/bin/env python3
"""
HPRC++: A tiny experimental image compressor (lossless)
- Auto-selects among: palette-RLE (+ optional zlib), predictive (+ optional zlib), raw
- Decodes to the original image exactly (lossless).

Usage:
  Encode: python hprc2.py encode input.png output.hprc2
  Decode: python hprc2.py decode input.hprc2 output.png

Dependencies:
  - Python 3.8+
  - numpy
  - pillow

This is a demo/prototype. No warranties. Enjoy!
"""

import argparse
import io
import math
import zlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

# ---------------- Utilities ----------------

def ensure_uint8(a):
    a = np.asarray(a, dtype=np.int32)
    a = np.clip(a, 0, 255).astype(np.uint8)
    return a

def zigzag_map(r: int) -> int:
    return (r << 1) if r >= 0 else (-r * 2 - 1)

def inv_zigzag(z: int) -> int:
    return (z >> 1) if (z & 1) == 0 else -(z >> 1) - 1

def median3_fast(a, b, c):
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return b

# ---------------- Bit IO ----------------

class BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.cur = 0
        self.nbits = 0

    def write_bit(self, b: int):
        self.cur = (self.cur << 1) | (b & 1)
        self.nbits += 1
        if self.nbits == 8:
            self.buf.append(self.cur & 0xFF)
            self.cur = 0
            self.nbits = 0

    def write_bits(self, val: int, n: int):
        for i in range(n - 1, -1, -1):
            self.write_bit((val >> i) & 1)

    def write_unary(self, q: int):
        for _ in range(q):
            self.write_bit(1)
        self.write_bit(0)

    def write_rice(self, value: int, k: int):
        if k < 0:
            k = 0
        q = value >> k
        r = value & ((1 << k) - 1) if k > 0 else 0
        self.write_unary(q)
        if k > 0:
            self.write_bits(r, k)

    def write_byte(self, val: int):
        self.write_bits(val & 0xFF, 8)

    def get_bytes(self) -> bytes:
        if self.nbits > 0:
            self.cur <<= (8 - self.nbits)
            self.buf.append(self.cur & 0xFF)
            self.cur = 0
            self.nbits = 0
        return bytes(self.buf)

class BitReader:
    def __init__(self, data: bytes):
        self.data = data
        self.idx = 0
        self.cur = 0
        self.nbits = 0

    def read_bit(self) -> int:
        if self.nbits == 0:
            if self.idx >= len(self.data):
                raise EOFError("BitReader: out of data")
            self.cur = self.data[self.idx]
            self.idx += 1
            self.nbits = 8
        b = (self.cur >> 7) & 1
        self.cur = (self.cur << 1) & 0xFF
        self.nbits -= 1
        return b

    def read_bits(self, n: int) -> int:
        val = 0
        for _ in range(n):
            val = (val << 1) | self.read_bit()
        return val

    def read_unary(self) -> int:
        q = 0
        while True:
            b = self.read_bit()
            if b == 0:
                return q
            q += 1

    def read_rice(self, k: int) -> int:
        if k < 0:
            k = 0
        q = self.read_unary()
        r = self.read_bits(k) if k > 0 else 0
        return (q << k) | r

    def read_byte(self) -> int:
        return self.read_bits(8)

# varint for run lengths
def write_varint(bw: BitWriter, n: int):
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            bw.write_byte(byte | 0x80)
        else:
            bw.write_byte(byte)
            break

def read_varint(br: BitReader) -> int:
    shift = 0
    result = 0
    while True:
        byte = br.read_byte()
        result |= ((byte & 0x7F) << shift)
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result

# ---------------- Color Transform ----------------

def rgb_to_rct(img: np.ndarray) -> np.ndarray:
    R = img[..., 0].astype(np.int32)
    G = img[..., 1].astype(np.int32)
    B = img[..., 2].astype(np.int32)
    Y = (R + 2 * G + B) // 4
    U = B - G
    V = R - G
    out = np.stack([Y, U, V], axis=-1).astype(np.int32)
    return out

def rct_to_rgb(ych: np.ndarray) -> np.ndarray:
    Y = ych[..., 0].astype(np.int32)
    U = ych[..., 1].astype(np.int32)
    V = ych[..., 2].astype(np.int32)
    G = Y - ((U + V) // 4)
    R = V + G
    B = U + G
    rgb = np.stack([R, G, B], axis=-1)
    return ensure_uint8(rgb)

# ---------------- Header ----------------

@dataclass
class HPRCHeader:
    width: int
    height: int
    channels: int
    mode: int      # 0=lossless predictive, 1=palette-RLE, 2=raw
    tile: int
    k_init: int
    chunk: int

def pack_header(W,H,C,mode,tile,k_init,chunk) -> bytes:
    bio = io.BytesIO()
    bio.write(b'HPRC2')
    bio.write(np.uint32(W).tobytes())
    bio.write(np.uint32(H).tobytes())
    bio.write(np.uint8(C).tobytes())
    bio.write(np.uint8(mode).tobytes())
    bio.write(np.uint16(tile).tobytes())
    bio.write(np.uint8(k_init).tobytes())
    bio.write(np.uint16(chunk).tobytes())
    return bio.getvalue()

def unpack_header(blob: bytes):
    if blob[:5] != b'HPRC2':
        raise ValueError("Not an HPRC2 file")
    off = 5
    W = int(np.frombuffer(blob[off:off+4], dtype=np.uint32)[0]); off += 4
    H = int(np.frombuffer(blob[off:off+4], dtype=np.uint32)[0]); off += 4
    C = int(np.frombuffer(blob[off:off+1], dtype=np.uint8)[0]); off += 1
    mode = int(np.frombuffer(blob[off:off+1], dtype=np.uint8)[0]); off += 1
    tile = int(np.frombuffer(blob[off:off+2], dtype=np.uint16)[0]); off += 2
    k_init = int(np.frombuffer(blob[off:off+1], dtype=np.uint8)[0]); off += 1
    chunk = int(np.frombuffer(blob[off:off+2], dtype=np.uint16)[0]); off += 2
    return (W,H,C,mode,tile,k_init,chunk,off)

# ---------------- Palette Encoder/Decoder ----------------

def encode_palette_rle(img: np.ndarray) -> bytes:
    H, W, C = img.shape
    flat = img.reshape(-1,3)
    colors, inv = np.unique(flat, axis=0, return_inverse=True)
    K = colors.shape[0]

    bw = BitWriter()
    header = pack_header(W,H,3,1,0,0,0)
    out = bytearray(header)

    out += bytes([K])
    out += colors.astype(np.uint8).tobytes()

    data = inv.astype(np.uint32)
    payload_literal = bytearray()
    idx = 0
    while idx < data.size:
        j = idx + 1
        while j < data.size and data[j] == data[idx]:
            j += 1
        run_len = j - idx
        if run_len >= 2:
            bw.write_bit(1)
            bw.write_bits(int(data[idx]) & 0xFF, 8)
            write_varint(bw, run_len)
            idx = j
        else:
            j = idx + 1
            while j < data.size:
                if j+1 < data.size and data[j] == data[j+1]:
                    break
                j += 1
            lit_len = j - idx
            bw.write_bit(0)
            write_varint(bw, lit_len)
            payload_literal.extend((data[idx:j] & 0xFF).astype(np.uint8).tolist())
            idx = j

    bit_payload = bw.get_bytes()
    core = np.uint32(len(bit_payload)).tobytes() + bit_payload + payload_literal
    comp = zlib.compress(core, level=9)
    if len(comp) < len(core):
        out += bytes([1])
        out += np.uint32(len(comp)).tobytes()
        out += comp
    else:
        out += bytes([0])
        out += core
    return bytes(out)

def decode_palette_rle(blob: bytes) -> np.ndarray:
    W,H,C,mode,tile,k_init,chunk,off = unpack_header(blob)
    K = blob[off]; off += 1
    palette = np.frombuffer(blob[off:off+3*K], dtype=np.uint8).reshape(K,3); off += 3*K
    is_z = blob[off]; off += 1
    if is_z == 1:
        clen = int(np.frombuffer(blob[off:off+4], dtype=np.uint32)[0]); off += 4
        core = zlib.decompress(blob[off:off+clen])
        off2 = 0
    else:
        core = blob[off:]
        off2 = 0
    bit_len = int(np.frombuffer(core[off2:off2+4], dtype=np.uint32)[0]); off2 += 4
    bit_payload = core[off2:off2+bit_len]; off2 += bit_len
    literal_payload = core[off2:]
    br = BitReader(bit_payload)
    lit_idx = 0

    indices = []
    total = W*H
    while len(indices) < total:
        tag = br.read_bit()
        if tag == 1:
            idx = br.read_byte()
            length = read_varint(br)
            indices.extend([idx]*length)
        else:
            length = read_varint(br)
            indices.extend(literal_payload[lit_idx:lit_idx+length])
            lit_idx += length
    arr = np.array(indices[:total], dtype=np.uint8).reshape(H,W)
    img = palette[arr]
    return img.astype(np.uint8)

# ---------------- Predictive (lossless) with bias correction ----------------

def choose_k(sum_vals: int, count: int) -> int:
    if count <= 0:
        return 0
    mean = sum_vals / count
    k = int(max(0, math.floor(math.log2(mean + 1e-9))))
    return max(0, min(8, k))

def encode_predictive(img: np.ndarray, tile=64, k_init=2, chunk=256, use_rct=True) -> bytes:
    if img.ndim == 2:
        img = img[..., None]
    H, W, C = img.shape
    if C == 3 and use_rct:
        work = rgb_to_rct(img)
        use_rct_flag = 1
    else:
        work = img.astype(np.int32)
        use_rct_flag = 0

    header = pack_header(W,H,C,0,tile,k_init,chunk)
    bw = BitWriter()
    out = bytearray(header)
    out += bytes([use_rct_flag])

    recon = np.zeros_like(work, dtype=np.int32)
    bias = np.zeros(C, dtype=np.int32)
    bias_shift = 4
    sum_nonzero = 0
    cnt_nonzero = 0
    k = k_init

    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            th = min(tile, H - ty)
            tw = min(tile, W - tx)
            for ch in range(C):
                run_zero = 0
                for oy in range(th):
                    for ox in range(tw):
                        x = tx + ox
                        y = ty + oy
                        L = recon[y, x - 1, ch] if x > 0 else 0
                        T = recon[y - 1, x, ch] if y > 0 else 0
                        TL = recon[y - 1, x - 1, ch] if (x > 0 and y > 0) else 0
                        baseP = median3_fast(int(L), int(T), int(L + T - TL))
                        P = baseP + (bias[ch] >> bias_shift)
                        true_val = int(work[y, x, ch])
                        r = true_val - P
                        e = zigzag_map(r)
                        recon[y, x, ch] = P + r
                        bias[ch] += r - (bias[ch] >> bias_shift)

                        if e == 0:
                            run_zero += 1
                        else:
                            if run_zero > 0:
                                bw.write_bit(0)
                                write_varint(bw, run_zero)
                                run_zero = 0
                            bw.write_bit(1)
                            bw.write_rice(e - 1, k)
                            sum_nonzero += e
                            cnt_nonzero += 1
                            if cnt_nonzero % chunk == 0:
                                k = choose_k(sum_nonzero, cnt_nonzero)
                if run_zero > 0:
                    bw.write_bit(0)
                    write_varint(bw, run_zero)

    core = bw.get_bytes()
    comp = zlib.compress(core, level=9)
    if len(comp) < len(core):
        out += bytes([1])
        out += np.uint32(len(comp)).tobytes()
        out += comp
    else:
        out += bytes([0])
        out += core
    return bytes(out)

def decode_predictive(blob: bytes) -> np.ndarray:
    W,H,C,mode,tile,k_init,chunk,off = unpack_header(blob)
    use_rct_flag = blob[off]; off += 1
    is_z = blob[off]; off += 1
    if is_z == 1:
        clen = int(np.frombuffer(blob[off:off+4], dtype=np.uint32)[0]); off += 4
        core = zlib.decompress(blob[off:off+clen])
    else:
        core = blob[off:]
    br = BitReader(core)

    recon = np.zeros((H,W,C), dtype=np.int32)
    bias = np.zeros(C, dtype=np.int32)
    bias_shift = 4
    sum_nonzero = 0
    cnt_nonzero = 0
    k = k_init

    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            th = min(tile, H - ty)
            tw = min(tile, W - tx)
            for ch in range(C):
                coords = [(x,y) for y in range(th) for x in range(tw)]
                idx = 0
                while idx < len(coords):
                    tag = br.read_bit()
                    if tag == 0:
                        run_zero = read_varint(br)
                        for _ in range(run_zero):
                            x = tx + coords[idx][0]
                            y = ty + coords[idx][1]
                            L = recon[y, x - 1, ch] if x > 0 else 0
                            T = recon[y - 1, x, ch] if y > 0 else 0
                            TL = recon[y - 1, x - 1, ch] if (x > 0 and y > 0) else 0
                            baseP = median3_fast(int(L), int(T), int(L + T - TL))
                            P = baseP + (bias[ch] >> bias_shift)
                            recon[y, x, ch] = P
                            r = 0
                            bias[ch] += r - (bias[ch] >> bias_shift)
                            idx += 1
                    else:
                        val = br.read_rice(k)
                        e = val + 1
                        x = tx + coords[idx][0]
                        y = ty + coords[idx][1]
                        L = recon[y, x - 1, ch] if x > 0 else 0
                        T = recon[y - 1, x, ch] if y > 0 else 0
                        TL = recon[y - 1, x - 1, ch] if (x > 0 and y > 0) else 0
                        baseP = median3_fast(int(L), int(T), int(L + T - TL))
                        P = baseP + (bias[ch] >> bias_shift)
                        r = inv_zigzag(e)
                        recon[y, x, ch] = P + r
                        bias[ch] += r - (bias[ch] >> bias_shift)
                        idx += 1
                        sum_nonzero += e
                        cnt_nonzero += 1
                        if cnt_nonzero % chunk == 0:
                            k = choose_k(sum_nonzero, cnt_nonzero)

    if C == 3 and use_rct_flag == 1:
        return rct_to_rgb(recon)
    elif C == 1:
        return ensure_uint8(recon[...,0])
    else:
        return ensure_uint8(recon)

# ---------------- Raw passthrough ----------------

def encode_raw(img: np.ndarray) -> bytes:
    H,W,C = img.shape
    header = pack_header(W,H,C,2,0,0,0)
    return header + img.astype(np.uint8).tobytes()

def decode_raw(blob: bytes) -> np.ndarray:
    W,H,C,mode,tile,k_init,chunk,off = unpack_header(blob)
    data = np.frombuffer(blob[off:], dtype=np.uint8)[:W*H*C]
    return data.reshape(H,W,C)

# ---------------- Top-level encode/decode ----------------

def encode_hprc2_select(img: np.ndarray) -> bytes:
    H,W,C = img.shape
    candidates = []
    # Raw
    candidates.append(encode_raw(img))
    # Predictive
    candidates.append(encode_predictive(img, tile=64, k_init=2, chunk=256, use_rct=(C==3)))
    # Palette if small unique colors
    if C == 3:
        colors = np.unique(img.reshape(-1,3), axis=0)
        if colors.shape[0] <= 64:
            candidates.append(encode_palette_rle(img))
    return min(candidates, key=len)

def decode_hprc2_any(blob: bytes) -> np.ndarray:
    W,H,C,mode,tile,k_init,chunk,off = unpack_header(blob)
    if mode == 0:
        return decode_predictive(blob)
    elif mode == 1:
        return decode_palette_rle(blob)
    elif mode == 2:
        return decode_raw(blob)
    else:
        raise ValueError("Unknown mode")

# ---------------- CLI ----------------

def load_image_rgb(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.array(im, dtype=np.uint8)

def save_image_rgb(arr: np.ndarray, path: str):
    Image.fromarray(arr).save(path)

def cmd_encode(inp: str, outp: str):
    img = load_image_rgb(inp)
    blob = encode_hprc2_select(img)
    with open(outp, "wb") as f:
        f.write(blob)
    print(f"Encoded {inp} -> {outp} ({len(blob)} bytes)")

def cmd_decode(inp: str, outp: str):
    with open(inp, "rb") as f:
        blob = f.read()
    img = decode_hprc2_any(blob)
    save_image_rgb(img, outp)
    print(f"Decoded {inp} -> {outp}")

def main():
    ap = argparse.ArgumentParser(description="HPRC++ experimental image compressor (lossless)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap_e = sub.add_parser("encode", help="encode PNG/JPG/... to .hprc2 (lossless expects original PNGs/JPGs decoded to RGB)")
    ap_e.add_argument("input", help="input image (any Pillow-readable)")
    ap_e.add_argument("output", help="output .hprc2 file")
    ap_d = sub.add_parser("decode", help="decode .hprc2 back to PNG")
    ap_d.add_argument("input", help="input .hprc2 file")
    ap_d.add_argument("output", help="output image (.png recommended)")
    args = ap.parse_args()

    if args.cmd == "encode":
        cmd_encode(args.input, args.output)
    elif args.cmd == "decode":
        cmd_decode(args.input, args.output)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()