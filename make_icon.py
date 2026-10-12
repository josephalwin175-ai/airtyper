"""Generates assets/airtyper.ico (run once; kept so the icon can be regenerated)."""
import os
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
S = 1024
NAVY, CYAN, GOLD, WHITE = (8, 15, 30), (90, 214, 255), (255, 178, 48), (232, 246, 255)

img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((40, 40, S - 40, S - 40), radius=200, fill=(14, 26, 48, 255), outline=(40, 90, 130, 255), width=12)

glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
g = ImageDraw.Draw(glow)
c = S // 2
cy = c - 60
for i in range(3):  # reactor arcs
    a0 = -90 + i * 120 + 15
    g.arc((c - 300, cy - 300, c + 300, cy + 300), a0, a0 + 90, fill=CYAN, width=46)
g.ellipse((c - 190, cy - 190, c + 190, cy + 190), outline=CYAN, width=14)
g.ellipse((c - 95, cy - 95, c + 95, cy + 95), outline=GOLD, width=26)
g.ellipse((c - 34, cy - 34, c + 34, cy + 34), fill=GOLD)
# keyboard row under the reactor
kw, kh, gap = 96, 70, 22
x = c - (5 * kw + 4 * gap) // 2
for k in range(5):
    g.rounded_rectangle((x, 790, x + kw, 790 + kh), radius=14, outline=CYAN, width=12,
                        fill=GOLD if k == 2 else None)
    x += kw + gap
blur = glow.filter(ImageFilter.GaussianBlur(28))
img = Image.alpha_composite(img, blur)
img = Image.alpha_composite(img, glow)

os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)
img.save(os.path.join(HERE, "assets", "airtyper.ico"),
         sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.resize((256, 256), Image.LANCZOS).save(os.path.join(HERE, "assets", "airtyper.png"))
print("icon written")
