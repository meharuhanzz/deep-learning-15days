"""Generates the repo's social preview image (GitHub recommends 1280x640,
min 200x200, shown when the repo link is shared on social media / chat
apps via Open Graph). GitHub only accepts this via Settings -> General ->
Social preview in the web UI - no API/CLI upload exists, so this script
just produces the PNG to upload manually.
"""
from PIL import Image, ImageDraw, ImageFont
import math
import random

W, H = 1280, 640
BG_TOP = (13, 17, 23)      # GitHub dark background tone
BG_BOTTOM = (22, 27, 34)
ACCENT = (88, 166, 255)     # GitHub blue
ACCENT2 = (63, 185, 80)     # GitHub green
WHITE = (230, 237, 243)
GRAY = (139, 148, 158)

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

random.seed(3)


def vertical_gradient(img, top, bottom):
    draw = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        c = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        draw.line([(0, y), (W, y)], fill=c)


def draw_network(draw, cx, cy, scale=1.0):
    """A small stylized neural-network diagram: 3 layers of nodes with
    connecting edges, on the right side of the card."""
    layers = [3, 5, 4, 2]
    xs = [cx + i * 95 * scale for i in range(len(layers))]
    positions = []
    for li, n in enumerate(layers):
        col = []
        span = (n - 1) * 48 * scale
        for j in range(n):
            y = cy - span / 2 + j * 48 * scale
            col.append((xs[li], y))
        positions.append(col)

    for li in range(len(positions) - 1):
        for p1 in positions[li]:
            for p2 in positions[li + 1]:
                draw.line([p1, p2], fill=(48, 58, 71), width=1)

    for li, col in enumerate(positions):
        color = ACCENT if li < len(positions) - 1 else ACCENT2
        for (x, y) in col:
            r = 7 * scale
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(10, 13, 18), width=2)


def main():
    img = Image.new("RGB", (W, H), BG_TOP)
    vertical_gradient(img, BG_TOP, BG_BOTTOM)
    draw = ImageDraw.Draw(img)

    # subtle scattered dots for texture
    for _ in range(60):
        x, y = random.randint(0, W), random.randint(0, H)
        r = random.choice([1, 1, 2])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(35, 42, 51))

    # neural network graphic, right side
    draw_network(draw, cx=810, cy=310, scale=1.15)

    # top accent bar
    draw.rectangle([0, 0, W, 8], fill=ACCENT)

    # title
    f_title = ImageFont.truetype(FONT_BOLD, 64)
    f_sub = ImageFont.truetype(FONT_REG, 28)
    f_badge = ImageFont.truetype(FONT_MONO, 22)
    f_foot = ImageFont.truetype(FONT_MONO, 20)

    draw.text((70, 130), "Deep Learning", font=f_title, fill=WHITE)
    draw.text((70, 205), "with PyTorch", font=f_title, fill=ACCENT)

    draw.text((70, 300), "A 15-day, from-first-principles course", font=f_sub, fill=GRAY)
    draw.text((70, 340), "math derivations · verified code · honest results", font=f_sub, fill=GRAY)

    # pill badges
    badges = ["autograd", "CNNs", "transformers", "GANs", "diffusion"]
    bx = 70
    by = 420
    for b in badges:
        tw = draw.textlength(b, font=f_badge)
        pad = 18
        bw = tw + pad * 2
        bh = 40
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=20,
                                outline=ACCENT2, width=2)
        draw.text((bx + pad, by + 8), b, font=f_badge, fill=ACCENT2)
        bx += bw + 14

    draw.text((70, 500), "github.com/meharuhanzz/deep-learning-15days",
               font=f_foot, fill=GRAY)

    # bottom accent bar
    draw.rectangle([0, H - 8, W, H], fill=ACCENT2)

    out = "social_preview.png"
    img.save(out)
    print(f"wrote {out} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
