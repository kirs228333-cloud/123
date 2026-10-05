"""Процедурная моушн-графика (детерминированная, без random в рантайме): фон-дым, души-частицы, кольца Requiem,
карточки гипотез, схема сделки, нити марионетки, счётчик душ. Всё рисуется PIL/numpy."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
_rng = np.random.default_rng(11)
RED = (255, 52, 52); VIOLET = (178, 102, 255); ORANGE = (255, 140, 40); WHITE = (244, 240, 246)

def ease(p): p = min(1, max(0, p)); return p * p * (3 - 2 * p)
def eo3(p): p = min(1, max(0, p)); return 1 - (1 - p) ** 3

# ---------- дымный фон ----------
_smoke = None
def smoke_bg(t, tint=(120, 40, 200), dark=0.55):
    global _smoke
    if _smoke is None:
        a = _rng.random((54, 96)).astype(np.float32); b = _rng.random((27, 48)).astype(np.float32)
        A = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((2400, 1350), Image.BICUBIC)).astype(np.float32) / 255
        B = np.asarray(Image.fromarray((b * 255).astype(np.uint8)).resize((2400, 1350), Image.BICUBIC)).astype(np.float32) / 255
        _smoke = (A * 0.6 + B * 0.4)
    ox = int((t * 18) % 400); oy = int((math.sin(t * 0.2) * 0.5 + 0.5) * 200)
    n = _smoke[oy:oy + H, ox:ox + W]
    n = np.clip((n - 0.35) * 1.8, 0, 1) ** 1.4
    col = np.array(tint, np.float32)[None, None, :] / 255
    img = (n[..., None] * col * dark * 255 + np.array([6, 4, 12], np.float32))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

# ---------- души-частицы ----------
_P = None
def _particles(n=70):
    global _P
    if _P is None:
        r = np.random.default_rng(5)
        _P = dict(x=r.random(n), y=r.random(n), v=0.03 + 0.07 * r.random(n), a=0.01 + 0.04 * r.random(n), w=0.2 + 0.8 * r.random(n),
                  ph=r.random(n) * 6.28, s=2 + 7 * r.random(n), c=r.random(n))
    return _P
def souls_layer(t, color=VIOLET, n=70, alpha=1.0):
    """RGBA слой с мерцающими частицами-душами (рисуется на 1/4 разрешения, с блюром)"""
    P = _particles(); sw, sh = W // 4, H // 4
    base = Image.new('RGB', (sw, sh), (0, 0, 0)); d = ImageDraw.Draw(base)
    for i in range(min(n, len(P['x']))):
        x = (P['x'][i] + P['a'][i] * math.sin(t * P['w'][i] + P['ph'][i])) % 1.0 * sw
        y = (P['y'][i] - P['v'][i] * t) % 1.0 * sh
        fl = 0.55 + 0.45 * math.sin(t * 2.2 * P['w'][i] + P['ph'][i] * 2)
        s = P['s'][i] * 0.6 * (0.7 + 0.5 * fl); col = color if P['c'][i] > 0.25 else (200, 200, 255)
        c = tuple(int(v * fl) for v in col)
        d.ellipse((x - s, y - s * 1.5, x + s, y + s * 1.5), fill=c)
        d.line((x, y + s, x - 2 + 4 * math.sin(i), y + s * 5), fill=tuple(int(v * 0.4 * fl) for v in col), width=1)
    g = base.filter(ImageFilter.GaussianBlur(2.2)).resize((W, H), Image.BILINEAR)
    return g  # рисовать screen/add

def add_screen(frame, layer, k=1.0):
    a = np.asarray(frame).astype(np.float32); b = np.asarray(layer).astype(np.float32) * k
    out = 255 - (255 - a) * (255 - np.clip(b, 0, 255)) / 255
    return Image.fromarray(out.astype(np.uint8))

# ---------- кольца Requiem ----------
def rings_layer(t, cx=W // 2, cy=H // 2 - 40, speed=520, gap=260, rmax=1250, color=RED, t0=0.0):
    sw, sh = W // 2, H // 2
    base = Image.new('RGB', (sw, sh), (0, 0, 0)); d = ImageDraw.Draw(base); tt = max(0, t - t0)
    for k in range(7):
        r = (tt * speed + k * gap) % (rmax + gap) - gap * 0.2
        if r <= 4: continue
        life = 1 - min(1, r / rmax); wdt = max(2, int(14 * life) + 2)
        c = tuple(int(v * (0.25 + 0.75 * life)) for v in color)
        d.ellipse(((cx - r) / 2, (cy - r * 0.62) / 2, (cx + r) / 2, (cy + r * 0.62) / 2), outline=c, width=wdt // 2 + 1)
        c2 = tuple(int(v * 0.5 * life) for v in ORANGE)
        d.ellipse(((cx - r * 0.92) / 2, (cy - r * 0.57) / 2, (cx + r * 0.92) / 2, (cy + r * 0.57) / 2), outline=c2, width=max(1, wdt // 3))
    g = base.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR)
    sharp = base.resize((W, H), Image.BILINEAR)
    return Image.blend(g, sharp, 0.35)

# ---------- вспомогательное ----------
_fc = {}
def F(name, size, wt):
    k = (name, size, wt)
    if k not in _fc:
        f = ImageFont.truetype(f'fonts/{name}.ttf', size)
        try: f.set_variation_by_name(wt)
        except Exception: pass
        _fc[k] = f
    return _fc[k]

def glow_text(img, xy, text, font, color, anchor='mm', stroke=0):
    lay = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    d.text(xy, text, font=font, fill=color + (255,), anchor=anchor, stroke_width=4, stroke_fill=color + (255,))
    lay = lay.filter(ImageFilter.GaussianBlur(10)); img.paste(lay, (0, 0), Image.eval(lay.getchannel('A'), lambda v: int(v * 0.6)))
    d2 = ImageDraw.Draw(img); d2.text(xy, text, font=font, fill=color, anchor=anchor, stroke_width=stroke, stroke_fill=(6, 4, 10))

# ---------- карточки гипотез ----------
def cards_scene(t, cards):
    """cards: [(t_in, title, sub, color)] - 2x2 сетка карточек, каждая появляется в свой момент"""
    fr = smoke_bg(t, (90, 30, 150), 0.5); d = ImageDraw.Draw(fr)
    pos = [(110, 120), (990, 120), (110, 470), (990, 470)]
    for i, (ti, title, sub, col) in enumerate(cards):
        p = eo3((t - ti) / 0.45)
        if p <= 0: continue
        x, y = pos[i]; cw, ch = 820, 300; off = int((1 - p) * 70); a = p
        lay = Image.new('RGBA', (cw + 80, ch + 80), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
        ld.rectangle((40, 40, 40 + cw, 40 + ch), fill=col + (int(110 * a),)); lay = lay.filter(ImageFilter.GaussianBlur(26))
        ld = ImageDraw.Draw(lay)
        ld.rectangle((40, 40, 40 + cw, 40 + ch), fill=(10, 7, 18, int(235 * a)), outline=col + (int(255 * a),), width=4)
        ld.rectangle((40, 40, 56, 40 + ch), fill=col + (int(255 * a),))
        # номер и иконка-спираль
        ld.text((120, 90), f'0{i + 1}', font=F('Oswald', 64, 'Bold'), fill=col + (int(255 * a),))
        cx, cy = 40 + cw - 120, 40 + ch // 2
        for k in range(26):
            ang = k * 0.5 + t * 1.2; rad = 4 + k * 2.6
            ld.ellipse((cx + math.cos(ang) * rad - 3, cy + math.sin(ang) * rad - 3, cx + math.cos(ang) * rad + 3, cy + math.sin(ang) * rad + 3), fill=col + (int(220 * a),))
        ld.text((120, 170), title, font=F('Playfair', 64, 'Bold'), fill=WHITE + (int(255 * a),))
        ld.text((120, 255), sub, font=F('Oswald', 38, 'Medium'), fill=(190, 184, 205, int(255 * a)))
        fr.paste(lay, (x - 40, y - 40 + off), lay)
    return fr

# ---------- схема сделки ----------
def deal_scene(t, t_force, t_soul, t_dead, labels=('РИКС', 'NEVERMORE'), tag=('ДЕМОНИЧЕСКИЙ ДОГОВОР',)):
    fr = smoke_bg(t, (150, 20, 40), 0.55); d = ImageDraw.Draw(fr)
    # узлы
    for (x, name, col) in ((380, labels[0], WHITE), (1540, labels[1], RED)):
        p = eo3((t - (t_force - 1.4)) / 0.6)
        r = int(150 * p)
        if r > 4:
            glow = Image.new('RGBA', (500, 500), (0, 0, 0, 0)); ImageDraw.Draw(glow).ellipse((250 - r, 250 - r, 250 + r, 250 + r), fill=col + (140,)); glow = glow.filter(ImageFilter.GaussianBlur(30))
            fr.paste(glow, (x - 250, 380 - 250), glow)
            d.ellipse((x - r, 380 - r, x + r, 380 + r), fill=(12, 8, 20), outline=col, width=5)
            d.text((x, 380), name, font=F('Oswald', 40 if len(name) > 6 else 52, 'Bold'), fill=col, anchor='mm')
    # стрелка «СИЛА» слева направо = Nevermore -> Рикс (справа налево)
    def arrow(x0, x1, y, p, col, text):
        if p <= 0: return
        xe = x0 + (x1 - x0) * eo3(p); d.line((x0, y, xe, y), fill=col, width=8)
        sgn = 1 if x1 > x0 else -1
        if p > 0.95: d.polygon([(xe, y), (xe - sgn * 36, y - 22), (xe - sgn * 36, y + 22)], fill=col)
        a = min(1, p * 1.5)
        if a > 0.3: d.text(((x0 + x1) / 2, y - 52), text, font=F('Oswald', 52, 'Bold'), fill=col, anchor='mm')
    arrow(1380, 540, 330, (t - t_force) / 0.9, VIOLET, 'СИЛА')
    arrow(540, 1380, 450, (t - t_soul) / 0.9, RED, 'ДУША')
    # погибшие возвращаются: ряд фигур-«душ» поднимается
    if t > t_dead:
        for i in range(6):
            p = eo3((t - t_dead - i * 0.12) / 0.5)
            if p <= 0: continue
            x = 560 + i * 160; y = 760 - 90 * p
            d.ellipse((x - 26, y - 70, x + 26, y - 18), fill=(40, 30, 60), outline=VIOLET, width=3)
            d.polygon([(x - 36, y + 50), (x + 36, y + 50), (x + 22, y - 14), (x - 22, y - 14)], fill=(40, 30, 60), outline=VIOLET)
            # цепи договора
            d.line((x - 40, y + 60, x + 40, y + 60), fill=RED, width=3)
        d.text((W // 2, 640), 'ВОИНЫ ВОЗВРАЩЕНЫ — ПО ДОГОВОРУ', font=F('Oswald', 40, 'Medium'), fill=(220, 190, 200), anchor='mm')
    return fr

# ---------- нити марионетки ----------
def strings_layer(t, strength=1.0):
    lay = Image.new('RGB', (W // 2, H // 2), (0, 0, 0)); d = ImageDraw.Draw(lay)
    for i in range(14):
        x = 120 + i * 108 / 1.0; sway = math.sin(t * 0.9 + i * 0.7) * 8
        top = 0; bot = 170 + 120 * math.sin(i * 1.3) + 30 * strength
        xs = x / 2 + sway * 0.3
        d.line((xs, 0, xs + sway * 0.5, bot * 0.5 * min(1, t * 0.8) + 0), fill=(150, 130, 170), width=1)
        d.ellipse((xs + sway * 0.5 - 3, bot * 0.5 * min(1, t * 0.8) - 3, xs + sway * 0.5 + 3, bot * 0.5 * min(1, t * 0.8) + 3), fill=(200, 180, 230))
    return lay.resize((W, H), Image.BILINEAR)

# ---------- счётчик душ ----------
def counter_scene(t, t0, label='НЕКРОМАСТЕРСТВО', step=0.42, n=12):
    fr = smoke_bg(t, (150, 30, 60), 0.5); d = ImageDraw.Draw(fr)
    cnt = max(0, min(n, int((t - t0) / step) + 1)) if t >= t0 else 0
    d.text((W // 2, 150), label, font=F('Oswald', 54, 'Bold'), fill=(220, 190, 210), anchor='mm')
    # ряд душ
    for i in range(n):
        x = 200 + i * 135; y = 430
        on = i < cnt; age = (t - t0 - i * step)
        pulse = 1 + 0.35 * max(0, 1 - age / 0.4) if on else 1
        col = VIOLET if on else (50, 40, 70)
        r = int(34 * pulse)
        if on:
            g = Image.new('RGBA', (200, 200), (0, 0, 0, 0)); ImageDraw.Draw(g).ellipse((100 - r * 1.6, 100 - r * 1.6, 100 + r * 1.6, 100 + r * 1.6), fill=VIOLET + (150,)); g = g.filter(ImageFilter.GaussianBlur(18)); fr.paste(g, (x - 100, y - 100), g)
        d.ellipse((x - r, y - r * 1.3, x + r, y + r * 1.3), fill=col, outline=(255, 255, 255) if on else (80, 70, 100), width=2)
        if on: d.polygon([(x - r * 0.6, y + r * 1.1), (x + r * 0.6, y + r * 1.1), (x, y + r * 2.2)], fill=col)
    # шкала силы
    bw = 1500; bx = (W - bw) // 2; by = 700
    d.rectangle((bx, by, bx + bw, by + 46), fill=(14, 10, 22), outline=(120, 100, 150), width=3)
    fill = bw * (cnt / n) ** 1.0 * (0.98)
    if fill > 4: d.rectangle((bx + 4, by + 4, bx + 4 + fill, by + 42), fill=RED)
    d.text((W // 2, by - 40), 'СИЛА РАСТЁТ С КАЖДОЙ ДУШОЙ', font=F('Oswald', 40, 'Medium'), fill=(230, 200, 210), anchor='mm')
    return fr
