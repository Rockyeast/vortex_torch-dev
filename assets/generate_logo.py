import math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = "/home/zhuominc/vortex_torch/assets"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# ---- blade geometry (log-spiral swirl) -------------------------------------
N, r0, T, tipR, w0 = 6, 0.16, 2.5, 0.95, 0.46
k = math.log(tipR / r0) / T
PALETTE = ['#2dd4bf', '#22d3ee', '#3b82f6', '#6366f1', '#a855f7', '#ec4899']

def hex2rgb(h): return tuple(int(h[i:i+2], 16) for i in (1, 3, 5))

def shape(t): return (1 - t) ** 0.85

def blade_poly(cx, cy, R, base, steps=160):
    lead, trail = [], []
    for i in range(steps + 1):
        t = i / steps; th = t * T
        r = r0 * math.exp(k * th); a = base + th; w = w0 * shape(t)
        lead.append((cx + R*r*math.cos(a-w), cy + R*r*math.sin(a-w)))
        trail.append((cx + R*r*math.cos(a+w), cy + R*r*math.sin(a+w)))
    return lead + trail[::-1]

def render_mark(px, glow=True):
    """Transparent RGBA mark of size px, gradient-shaded blades."""
    SS = 4
    S = px * SS
    cx = cy = S / 2
    R = S * 0.46
    # radial lightness map: deeper at center, brighter toward tip
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    rad = np.sqrt((xx - cx)**2 + (yy - cy)**2) / R
    light = np.clip(0.16 * rad, 0.0, 0.13)                 # subtle sheen at tips
    darken = 1.0 - 0.14 * (1.0 - np.clip(rad, 0, 1))        # mild depth at core
    out = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    for i in range(N):
        base = math.radians(-90) + i * 2 * math.pi / N
        m = Image.new('L', (S, S), 0)
        ImageDraw.Draw(m).polygon(blade_poly(cx, cy, R, base), fill=255)
        mask = np.asarray(m, np.float32) / 255.0
        col = np.array(hex2rgb(PALETTE[i]), np.float32)
        rgb = (col[None, None, :] * (1 - light[..., None]) + 255.0 * light[..., None])
        rgb = rgb * darken[..., None]
        layer = np.zeros((S, S, 4), np.float32)
        layer[..., :3] = rgb
        layer[..., 3] = mask * 255.0
        out = Image.alpha_composite(out, Image.fromarray(layer.astype(np.uint8)))
    return out.resize((px, px), Image.LANCZOS)

# ---- assets ----------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)

mark = render_mark(720)
mark.save(f"{OUT}/vortex_mark.png")

# full lockup on white (GitHub README, light bg) -> 1518x854 like the original
W, H = 1518, 854
flat = Image.new('RGBA', (W, H), (255, 255, 255, 255))
m = render_mark(560)
flat.alpha_composite(m, (int(W/2 - 280), 40))
draw = ImageDraw.Draw(flat)
txt = "VORTEX"
fsz = 150
font = ImageFont.truetype(FONT, fsz)
# letter-spaced, centered
spacing = int(fsz * 0.14)
widths = [draw.textlength(c, font=font) for c in txt]
total = sum(widths) + spacing * (len(txt) - 1)
x = (W - total) / 2
ty = 615
for c, w in zip(txt, widths):
    draw.text((x, ty), c, font=font, fill=(15, 23, 42, 255))
    x += w + spacing
flat.convert('RGB').save(f"{OUT}/vortex_logo_flat.png")

# ---- vector SVG mark (gradient blades, infinitely crisp) -------------------
def svg_blade_path(base, vb=200.0, steps=64):
    cx = cy = vb / 2; R = vb * 0.46
    pts = []
    for i in range(steps + 1):
        t = i / steps; th = t * T
        r = r0 * math.exp(k * th); a = base + th; w = w0 * shape(t)
        pts.append((cx + R*r*math.cos(a-w), cy + R*r*math.sin(a-w)))
    for i in range(steps, -1, -1):
        t = i / steps; th = t * T
        r = r0 * math.exp(k * th); a = base + th; w = w0 * shape(t)
        pts.append((cx + R*r*math.cos(a+w), cy + R*r*math.sin(a+w)))
    d = f"M{pts[0][0]:.2f},{pts[0][1]:.2f} " + " ".join(f"L{x:.2f},{y:.2f}" for x, y in pts[1:]) + " Z"
    return d

def tint(hexc, f):
    r, g, b = hex2rgb(hexc)
    return f"#{int(r+(255-r)*f):02x}{int(g+(255-g)*f):02x}{int(b+(255-b)*f):02x}"

def svg_mark(vb=200.0):
    grads, paths = [], []
    cx = vb / 2
    for i in range(N):
        base = math.radians(-90) + i * 2 * math.pi / N
        gid = f"g{i}"
        mid = base + T * 0.5
        x1 = cx + 8*math.cos(mid); y1 = cx + 8*math.sin(mid)
        x2 = cx + (vb*0.46)*math.cos(mid); y2 = cx + (vb*0.46)*math.sin(mid)
        grads.append(
            f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
            f'x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}">'
            f'<stop offset="0" stop-color="{tint(PALETTE[i],0.04)}"/>'
            f'<stop offset="1" stop-color="{tint(PALETTE[i],0.45)}"/></linearGradient>')
        paths.append(f'<path d="{svg_blade_path(base, vb)}" fill="url(#{gid})"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vb:.0f} {vb:.0f}" '
            f'width="{vb:.0f}" height="{vb:.0f}" fill="none">\n<defs>\n'
            + "\n".join(grads) + "\n</defs>\n" + "\n".join(paths) + "\n</svg>\n")

with open(f"{OUT}/vortex_mark.svg", "w") as f:
    f.write(svg_mark())

def svg_lockup():
    vb = 200.0
    inner = svg_mark(vb)
    body = inner[inner.index('>')+1: inner.rindex('</svg>')]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 270" '
        'width="200" height="270" fill="none">\n'
        f'<g>{body}</g>\n'
        '<text x="100" y="252" text-anchor="middle" '
        'font-family="DejaVu Sans, Arial, Helvetica, sans-serif" font-weight="700" '
        'font-size="46" letter-spacing="5" fill="#0f172a">VORTEX</text>\n</svg>\n')

with open(f"{OUT}/vortex_logo.svg", "w") as f:
    f.write(svg_lockup())

print("wrote vortex_mark.png, vortex_logo_flat.png, vortex_mark.svg, vortex_logo.svg")
