"""PIL BICUBIC outputs of synthetic RGB images, for an exact resampler test.

Usage: gen_pil_bicubic_synthetic.py pil_bicubic_synthetic.bin
Pixel (x, y) of a WxH input is ((x*37 + y*11) % 256, (x*5 + y*29 + 7) % 256, (x*x + 3*y) % 256).
The file holds, in order, the resized bytes (HWC u8) of each case below.
"""
import sys

from PIL import Image

CASES = [((37, 23), (64, 48)), ((97, 61), (32, 24))]


def synthetic(w, h):
    img = Image.new('RGB', (w, h))
    img.putdata([((x * 37 + y * 11) % 256, (x * 5 + y * 29 + 7) % 256, (x * x + 3 * y) % 256)
                 for y in range(h) for x in range(w)])
    return img


with open(sys.argv[1], 'wb') as out:
    for (w, h), (dw, dh) in CASES:
        out.write(synthetic(w, h).resize((dw, dh), resample=Image.BICUBIC).tobytes())
print('ok')
