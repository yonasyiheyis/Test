# Test

HPRC++ is a tiny, lossless image compressor prototype. It auto-selects the best of:
	•	Palette-RLE (+ zlib) for few-color images (UI, line art, sprites),
	•	Predictive (+ bias, + zlib) for photos/gradients,
	•	Raw passthrough for worst-case (e.g., noise).

python -V   # 3.8+
pip install numpy pillow

# Encode image (PNG/JPG/etc.) -> .hprc2
python hprc2.py encode input.png output.hprc2

# Decode .hprc2 -> PNG
python hprc2.py decode output.hprc2 restored.png

Notes
	•	Always lossless round-trip.
	•	Uses only stdlib zlib (no external compressors).
	•	Works on RGB; images are read via Pillow and converted to RGB internally.