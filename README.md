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


How to use
	1.	Put hprc2.py and bench_hprc2.py in the same folder.
	2.	Prepare a folder of test images (e.g. the make_test_images.py outputs or your own).
	3.	Run
python bench_hprc2.py ./images/

You’ll get per-file and total size comparisons like:
lineart.png          PNG=   4211  HPRC2=   3011  ratio=  0.72
ui_palette.png       PNG=   8922  HPRC2=   4987  ratio=  0.56
sprite_sheet.png     PNG=  15332  HPRC2=   9021  ratio=  0.59
photoish.png         PNG=  44123  HPRC2=  39982  ratio=  0.91

Summary:
Total PNG size   : 72588 bytes
Total HPRC2 size : 57999 bytes
Overall ratio    :  0.80
