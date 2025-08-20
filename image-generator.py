# save as make_test_images.py, then run: python make_test_images.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def lineart(w=512,h=512):
    img = Image.new("RGB", (w, h), "white")
    dr = ImageDraw.Draw(img)
    for i in range(0, w, 16):
        dr.line([(i, 0), (i, h)], fill="black", width=1)
    for j in range(0, h, 16):
        dr.line([(0, j), (w, j)], fill="black", width=1)
    dr.rectangle([40, 40, 220, 220], outline="black", width=6)
    dr.ellipse([260, 60, 460, 260], outline="black", width=6)
    dr.line([(60, 440), (460, 380)], fill="black", width=7)
    try:
        font = ImageFont.load_default()
        dr.text((260, 300), "HPRC++", fill="black", font=font)
    except:
        dr.text((260, 300), "HPRC++", fill="black")
    return img

def ui_palette(w=512,h=512):
    img = Image.new("RGB", (w, h), (240,240,240))
    dr = ImageDraw.Draw(img)
    dr.rectangle([40,40,470,110], fill=(66,135,245))
    dr.rectangle([40,130,470,470], fill=(255,255,255), outline=(200,200,200), width=2)
    colors = [(66,135,245),(52,199,89),(255,149,0),(255,59,48)]
    for i,c in enumerate(colors):
        x0 = 60+i*100; y0=160; x1=x0+80; y1=y0+40
        dr.rectangle([x0,y0,x1,y1], fill=c)
    for r in range(10):
        y = 220 + r*20
        dr.rectangle([60,y, 420, y+8], fill=(180,180,180))
    return img

def sprite_sheet(w=512,h=512):
    img = Image.new("RGB", (w, h))
    px = img.load()
    tile = 32
    colors = [(i*16, j*16, ((i+j)%16)*16) for i in range(16) for j in range(16)]
    idx = 0
    for y in range(0,h,tile):
        for x in range(0,w,tile):
            r,g,b = colors[idx % len(colors)]
            for yy in range(y,y+tile):
                for xx in range(x,x+tile):
                    px[xx,yy] = (r,g,b)
            idx += 1
    return img

def photoish(w=512,h=512):
    yy, xx = np.mgrid[0:h, 0:w]
    R = 128 + 60*np.sin(2*np.pi*xx/w) + 40*np.cos(2*np.pi*yy/h)
    G = 128 + 50*np.sin(2*np.pi*(xx+yy)/w) + 30*np.cos(2*np.pi*(xx-yy)/h)
    B = 128 + 40*np.cos(2*np.pi*xx/w) + 50*np.sin(2*np.pi*yy/h)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 2.5, size=(h, w, 3))
    img = np.stack([R, G, B], axis=-1) + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

for name, fn in [
    ("lineart.png", lineart),
    ("ui_palette.png", ui_palette),
    ("sprite_sheet.png", sprite_sheet),
    ("photoish.png", photoish),
]:
    im = fn()
    im.save(name)
    print("wrote", name)