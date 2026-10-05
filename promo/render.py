"""Рендер ролика: PIL-кадры -> ffmpeg. Запуск: python render.py snap T1 T2 ...   |   python render.py full out/video_v1.mp4"""
import json, re, math, sys, os, subprocess, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1920, 1080, 30
RED = (255, 52, 52); VIOLET = (178, 102, 255); WHITE = (244, 240, 246)
S = json.load(open('script.json', encoding='utf-8'))
WORDS = json.load(open('words.json', encoding='utf-8'))
HOOK = json.load(open('chunks.json', encoding='utf-8'))['hook']
FILES = sorted(glob.glob('../*.jpg'))                 # номер картинки = позиция в алфавитном списке (1..26)
def IMG(n): return FILES[n - 1]

# ---------- слова / строки ----------
LINES = S['lines'] + S['outro']
def norm(w): return re.sub(r'[^0-9a-zа-я]', '', w.lower().replace('ё', 'е'))
line_rng = []; k = 0
for l in LINES:
    n = len(l.split()); line_rng.append((k, k + n)); k += n
assert k == len(WORDS), (k, len(WORDS))
def line_start(i): return WORDS[line_rng[i][0]]['s']
def line_end(i): return WORDS[line_rng[i][1] - 1]['e']
def T(line, prefix, nth=0):
    a, b = line_rng[line]; c = 0
    for j in range(a, b):
        if norm(WORDS[j]['w']).startswith(norm(prefix)):
            if c == nth: return WORDS[j]['s']
            c += 1
    print('WARN trigger not found', line, prefix, file=sys.stderr); return line_start(line)

# ---------- шрифты ----------
_fc = {}
def font(name, size, wt):
    key = (name, size, wt)
    if key not in _fc:
        f = ImageFont.truetype(f'fonts/{name}.ttf', size)
        try: f.set_variation_by_name(wt)
        except Exception: pass
        _fc[key] = f
    return _fc[key]

# ---------- картинки ----------
_ic = {}
def load_img(n):
    if n in _ic: return _ic[n]
    im = Image.open(IMG(n)).convert('RGB')
    a = np.asarray(im).astype(np.float32) / 255
    lum = float(a.mean())
    if lum > 0.42:   # светлые ч/б арты -> тёмная мрачная тонировка
        tint = np.array([0.50, 0.30, 0.44], np.float32)
        a = (a ** 1.25) * tint * 1.25
    else:
        a = a * np.array([0.97, 0.95, 1.0], np.float32)
    a = np.clip(a, 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8))
    sm = im.resize((480, 270), Image.BILINEAR, box=cover_box(im.size, 0, 0.5, 0.5)).filter(ImageFilter.GaussianBlur(14))
    bg = sm.resize((W, H), Image.BILINEAR)
    bg = Image.fromarray((np.asarray(bg).astype(np.float32) * 0.38).astype(np.uint8))
    _ic[n] = (im, bg); return _ic[n]
def cover_box(size, e, px, py, z=1.0, aspect=W / H):
    w, h = size; sc = max(W / w, H / h) if aspect == W / H else None
    cw, ch = W / sc / z, H / sc / z
    return ((w - cw) * px, (h - ch) * py, (w - cw) * px + cw, (h - ch) * py + ch)

def ease(p): p = min(1, max(0, p)); return p * p * (3 - 2 * p)
def eo3(p): p = min(1, max(0, p)); return 1 - (1 - p) ** 3
def eback(p):
    p = min(1, max(0, p)); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (p - 1) ** 3 + c1 * (p - 1) ** 2

# ---------- оверлеи-плашки ----------
_tc = {}
def text_layer(text, size, color, maxw, fname='Playfair', wt='Bold'):
    key = (text, size, color, maxw, fname, wt)
    if key in _tc: return _tc[key]
    f = font(fname, size, wt); tmp = ImageDraw.Draw(Image.new('RGB', (4, 4)))
    while True:
        lines = []
        for para in text.split('\n'):
            cur = ''
            for w in para.split():
                t = (cur + ' ' + w).strip()
                if tmp.textlength(t, font=f) <= maxw or not cur: cur = t
                else: lines.append(cur); cur = w
            lines.append(cur)
        mw = max(tmp.textlength(l, font=f) for l in lines)
        if mw <= maxw or size <= 40: break
        size -= 6; f = font(fname, size, wt)
    lh = int(size * 1.12); pad = 60
    lw, lhh = int(mw) + pad * 2, lh * len(lines) + pad * 2
    glow = Image.new('RGBA', (lw, lhh), (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    lay = Image.new('RGBA', (lw, lhh), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    for i, l in enumerate(lines):
        x = (lw - tmp.textlength(l, font=f)) / 2; y = pad + i * lh
        gd.text((x, y), l, font=f, fill=color + (255,), stroke_width=6, stroke_fill=color + (255,))
        d.text((x, y), l, font=f, fill=color + (255,), stroke_width=7, stroke_fill=(6, 4, 10, 255))
        d.text((x, y), l, font=f, fill=color + (255,))
    glow = glow.filter(ImageFilter.GaussianBlur(16))
    ga = np.asarray(glow).copy(); ga[..., 3] = (ga[..., 3] * 0.6).astype(np.uint8)
    out = Image.alpha_composite(Image.fromarray(ga), lay)
    _tc[key] = out; return out

def paste_alpha(base, layer, cx, cy, alpha=1.0, scale=1.0):
    if scale != 1.0:
        layer = layer.resize((max(1, int(layer.width * scale)), max(1, int(layer.height * scale))), Image.BILINEAR)
    if alpha < 1.0:
        a = layer.getchannel('A').point(lambda v: int(v * alpha)); layer = layer.copy(); layer.putalpha(a)
    base.paste(layer, (int(cx - layer.width / 2), int(cy - layer.height / 2)), layer)

# ---------- раскадровка ----------
# (первая строка, последняя строка, картинка, режим cover|card, сторона карточки L|R, акцент-цвет, [оверлеи], pan-вариант)
def OV(line, trig, text, color=RED, style='slide', size=116, nth=0, dt=-0.15):
    return {'t': T(line, trig, nth) + dt, 'text': text, 'color': color, 'style': style, 'size': size}
SB = [
 (0, 0, 24, 'cover', 'C', RED, [OV(0, 'забирает', 'ЗАБИРАЕТ\nДУШИ', VIOLET, 'stamp', 150)], 0),
 (1, 1, 20, 'card', 'R', RED, [OV(1, 'равны', 'ВСЕ РАВНЫ', RED, 'stamp', 128)], 1),
 (2, 3, 19, 'card', 'L', RED, [OV(2, 'Shadow', 'SHADOW\nFIEND', RED, 'stamp', 130), OV(3, 'Nevermore', 'NEVERMORE', VIOLET, 'stamp', 118)], 0),
 (4, 4, 14, 'cover', 'C', VIOLET, [OV(4, 'Abysm', 'ABYSM', VIOLET, 'stamp', 170)], 1),
 (5, 6, 16, 'card', 'R', VIOLET, [OV(5, 'марионетка', 'МАРИОНЕТКА?', RED, 'slide', 120)], 0),
 (7, 7, 12, 'cover', 'C', VIOLET, [OV(7, 'коллекции', 'КОЛЛЕКЦИЯ\nДУШ', VIOLET, 'slide', 130)], 0),
 (8, 9, 10, 'card', 'L', VIOLET, [OV(8, 'ресурс', 'РЕСУРС', VIOLET, 'stamp', 140), OV(9, 'забирает', 'ТОЛЬКО\nЗАБИРАЕТ', RED, 'slide', 120)], 0),
 (10, 11, 5, 'cover', 'C', VIOLET, [OV(11, 'Никто', 'КУДА?', RED, 'stamp', 220)], 1),
 (12, 12, 25, 'card', 'R', VIOLET, [OV(12, 'поглощает', 'ПОГЛОЩАЕТ', RED), OV(12, 'хранит', 'ХРАНИТ\nВ ХРАМЕ', VIOLET), OV(12, 'помещает', 'НЕКРО-\nЖИДКОСТЬ', VIOLET), OV(12, 'частью', 'ЧАСТЬ\nТЕЛА', RED)], 0),
 (13, 14, 18, 'cover', 'C', RED, [OV(14, 'Стать', 'СТАТЬ СЛЕДУЮЩЕЙ\nДУШОЙ', RED, 'stamp', 130)], 0),
 (15, 17, 6, 'card', 'L', RED, [OV(15, 'Necromastery', 'NECROMASTERY', VIOLET, 'stamp', 100), OV(17, 'накопленная', 'НАКОПЛЕННАЯ\nСМЕРТЬ', RED, 'stamp', 112)], 0),
 (18, 18, 21, 'cover', 'C', VIOLET, [OV(18, 'Presence', 'PRESENCE OF\nTHE DARK LORD', VIOLET, 'slide', 100)], 1),
 (19, 19, 2, 'card', 'R', RED, [OV(19, 'кому', 'КОМУ\nСЛУЖИТ?', RED, 'stamp', 140)], 0),
 (20, 21, 22, 'card', 'L', RED, [OV(20, 'Люцифера', 'ЛЮЦИФЕР', RED, 'stamp', 130, dt=-0.1), OV(21, 'Shadow', 'SHADOW\nDEMON', VIOLET, 'stamp', 128)], 0),
 (22, 23, 7, 'cover', 'C', RED, [OV(23, 'намёки', 'ТОЛЬКО НАМЁКИ', RED, 'slide', 120)], 0),
 (24, 26, 17, 'card', 'R', RED, [OV(24, 'Demon', 'DEMON EATER', RED, 'stamp', 118), OV(26, 'ошибкой', 'ОГРОМНАЯ\nОШИБКА', RED, 'stamp', 130)], 0),
 (27, 28, 1, 'cover', 'C', RED, [OV(27, 'тьмой', 'ТЬМА\nИ ЯРОСТЬ', RED, 'stamp', 150), OV(28, 'силу', 'ЦЕНА — СИЛА', VIOLET, 'slide', 120)], 1),
 (29, 30, 13, 'card', 'L', VIOLET, [OV(30, 'пожирать', 'ПОЖИРАЕТ\nДЕМОНОВ', RED, 'stamp', 120)], 0),
 (31, 32, 4, 'cover', 'C', VIOLET, [OV(31, 'Artifact', 'ARTIFACT', VIOLET, 'stamp', 150), OV(32, 'Court', 'COURT OF RISTUL', RED, 'slide', 110)], 0),
 (33, 33, 11, 'card', 'R', RED, [OV(33, 'командир', 'КОМАНДИР\nРИКС', RED, 'stamp', 130)], 0),
 (34, 35, 23, 'cover', 'C', RED, [OV(34, 'соглашается', 'СДЕЛКА', RED, 'stamp', 190), OV(35, 'цену', 'ЦЕНА', VIOLET, 'stamp', 190)], 0),
 (36, 37, 3, 'cover', 'C', VIOLET, [OV(36, 'душу', 'ДУША ЗА СИЛУ', VIOLET, 'slide', 120), OV(37, 'торговец', 'ТОРГОВЕЦ\nДУШАМИ', RED, 'stamp', 130)], 1),
 (38, 39, 8, 'cover', 'C', RED, [OV(39, 'Requiem', 'REQUIEM\nOF SOULS', RED, 'stamp', 150)], 0),
 (40, 42, 15, 'cover', 'C', RED, [OV(41, 'расплата', 'ПОСЛЕДНЯЯ\nРАСПЛАТА', RED, 'stamp', 130)], 1),
 (43, 44, 19, 'cover', 'C', RED, [OV(43, 'кто', 'КТО ОН?', RED, 'stamp', 170)], 1),
 (45, 46, 9, 'card', 'L', VIOLET, [OV(45, 'коллекционер', 'КОЛЛЕКЦИОНЕР\nДУШ', VIOLET, 'stamp', 112)], 0),
 (47, 48, 14, 'card', 'R', VIOLET, [OV(48, 'Куда', 'КУДА ПОПАДАЮТ\nДУШИ?', VIOLET, 'stamp', 112)], 0),
 (49, 50, 25, 'cover', 'C', VIOLET, [OV(49, 'поглощает', 'ПОГЛОЩАЕТ?', RED, 'stamp', 150), OV(50, 'хранит', 'ХРАНИТ\nВ БЕЗДНЕ?', VIOLET, 'stamp', 150)], 1),
 (51, 51, 10, 'cover', 'C', VIOLET, [OV(51, 'цели', 'ЦЕЛЬ?', RED, 'stamp', 190)], 1),
 (52, 53, 4, 'cover', 'C', RED, [OV(52, 'марионетка', 'МАРИОНЕТКА', RED, 'stamp', 150), OV(53, 'кто', 'КТО ДЁРГАЕТ\nЗА НИТКИ?', VIOLET, 'stamp', 130)], 1),
]
OUTRO = (54, 55)
END = line_end(55) + 2.2

def build_scenes():
    sc = []
    # затравка (10.4 c)
    sc.append({'t0': 0.0, 't1': 2.6, 'kind': 'hook', 'img': 19, 'z': (1.0, 1.30), 'p': ((0.5, 0.5), (0.5, 0.0)), 'tr': 'none', 'cap': None})
    sc.append({'t0': 2.6, 't1': 4.0, 'kind': 'hook', 'img': 12, 'z': (1.15, 1.0), 'p': ((0.5, 1.0), (0.5, 0.0)), 'tr': 'flash', 'cap': None})
    sc.append({'t0': 4.0, 't1': 5.2, 'kind': 'hook', 'img': 10, 'z': (1.0, 1.2), 'p': ((0.5, 0.2), (0.5, 0.8)), 'tr': 'flash', 'cap': None})
    sc.append({'t0': 5.2, 't1': 7.6, 'kind': 'hook', 'img': 8, 'z': (1.0, 1.25), 'p': ((0.5, 0.8), (0.5, 0.2)), 'tr': 'flash', 'cap': 'REQUIEM OF SOULS'})
    sc.append({'t0': 7.6, 't1': HOOK, 'kind': 'question', 'tr': 'flash'})
    kinds = ['h', 'v', 'c']; ti = 0
    for si, (a, b, img, mode, side, acc, ovs, var) in enumerate(SB):
        t0 = HOOK if si == 0 else line_start(a) - 0.22
        t1 = (line_start(SB[si + 1][0]) - 0.22) if si + 1 < len(SB) else line_start(OUTRO[0]) - 0.22
        trn = kinds[ti % 3]; ti += 1
        # внутренняя фаза для длинных сцен (>8.5 c)
        parts = [(t0, t1)]
        if t1 - t0 > 8.5:
            n = int(math.ceil((t1 - t0) / 7.0)); parts = []
            cuts = phrase_cut_times(a, b, t0, t1, n)
            pts = [t0] + cuts + [t1]; parts = list(zip(pts[:-1], pts[1:]))
        for pi, (p0, p1) in enumerate(parts):
            m, v = mode, var
            if pi % 2 == 1: m = 'card' if mode == 'cover' else 'cover'; v = 1 - var
            sc.append({'t0': p0, 't1': p1, 'kind': 'main', 'img': img, 'mode': m, 'side': side if pi % 2 == 0 else ('R' if side == 'L' else 'L'),
                       'acc': acc, 'ovs': ovs, 'var': v, 'tr': trn if pi == 0 else 'dip', 'sid': si})
    sc.append({'t0': line_start(OUTRO[0]) - 0.22, 't1': END, 'kind': 'outro', 'tr': 'c' if ti % 3 == 2 else kinds[ti % 3]})
    return sc

def phrase_cut_times(a, b, t0, t1, n):
    """точки разреза внутри сцены: ближайшие к равным долям начала слов после знаков препинания"""
    j0, j1 = line_rng[a][0], line_rng[b][1]; cand = []
    for j in range(j0 + 1, j1):
        if re.search(r'[.,;:!?—…]$', WORDS[j - 1]['w']): cand.append(WORDS[j]['s'] - 0.1)
    out = []
    for q in range(1, n):
        tgt = t0 + (t1 - t0) * q / n
        c = min(cand, key=lambda x: abs(x - tgt)) if cand else tgt
        out.append(c)
    return sorted(set(out))

SCENES = build_scenes()

# ---------- субтитры ----------
IMPORTANT_RED = ('смерт', 'демон', 'бездн', 'nevermore', 'shadow', 'fiend', 'requiem', 'necromastery', 'сделк', 'цен', 'марионет', 'хозяин', 'тайн', 'тьм', 'ярост', 'убива', 'abysm', 'dark', 'lord', 'рикс', 'rix', 'люцифер', 'расплат')
IMPORTANT_VIOLET = ('душ', 'душа', 'души', 'коллекц', 'abyss', 'soul', 'souls', 'некромант')
def word_color(w):
    n = norm(w)
    if any(n.startswith(p) for p in IMPORTANT_RED): return RED
    if any(n.startswith(p) for p in IMPORTANT_VIOLET): return VIOLET
    return None

def build_phrases():
    ph = []
    for li in range(54):
        a, b = line_rng[li]; cur = []
        for j in range(a, b):
            cur.append(j)
            if re.search(r'[,.;:!?—…]$', WORDS[j]['w']) or WORDS[j]['w'] == '—':
                ph.append(cur); cur = []
        if cur: ph.append(cur)
    # >7 слов -> делим поровну; одиночные слова приклеиваем
    out = []
    for p in ph:
        if len(p) > 7:
            n = math.ceil(len(p) / 7); sz = math.ceil(len(p) / n)
            out += [p[i:i + sz] for i in range(0, len(p), sz)]
        else: out.append(p)
    merged = []
    for p in out:
        if merged and len(p) == 1 and len(merged[-1]) + 1 <= 8 and p[0] == merged[-1][-1] + 1: merged[-1] = merged[-1] + p
        else: merged.append(p)
    res = []
    for i, p in enumerate(merged):
        s = WORDS[p[0]]['s'] - 0.1; e = WORDS[p[-1]]['e'] + 0.3
        if i + 1 < len(merged): e = min(e, WORDS[merged[i + 1][0]]['s'] - 0.05)
        res.append((s, max(e, s + 0.4), p))
    return res
PHRASES = build_phrases()
_sc = {}
def sub_layer(pi, cur):
    key = (pi, cur)
    if key in _sc: return _sc[key]
    s, e, idxs = PHRASES[pi]; size = 66; f = font('Oswald', size, 'SemiBold')
    tmp = ImageDraw.Draw(Image.new('RGB', (4, 4)))
    ws = [WORDS[j]['w'] for j in idxs]
    while True:
        space = tmp.textlength(' ', font=f); widths = [tmp.textlength(w, font=f) for w in ws]
        tw = sum(widths) + space * (len(ws) - 1)
        if tw <= 1700 or size <= 44: break
        size -= 4; f = font('Oswald', size, 'SemiBold')
    pw, ph_ = int(tw) + 90, size + 56
    lay = Image.new('RGBA', (pw, ph_), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    d.rounded_rectangle((0, 0, pw - 1, ph_ - 1), radius=18, fill=(6, 4, 10, 150))
    x = 45
    for k2, (j, w) in enumerate(zip(idxs, ws)):
        c = word_color(w); spoken = j < cur; now = (j == cur)
        if now: col = c or (255, 255, 255)
        elif spoken: col = c if c else (236, 232, 240)
        else: col = tuple(int(v * 0.42) for v in c) if c else (132, 128, 140)
        if now and not c: col = (255, 214, 214)
        d.text((x, 20), w, font=f, fill=col, stroke_width=3, stroke_fill=(6, 4, 10))
        x += widths[k2] + space
    _sc[key] = lay; return lay

def cur_word(pi, t):
    idxs = PHRASES[pi][2]; cur = idxs[0] - 1
    for j in idxs:
        if WORDS[j]['s'] <= t + 0.02: cur = j
    return cur if cur >= idxs[0] else idxs[0] - 1

# ---------- сцены ----------
def pan_for(sc):
    v = sc['var']
    return ((0.5, 0.0), (0.5, 1.0)) if v == 0 else ((0.5, 1.0), (0.5, 0.0))

VIG = None
def vignette():
    global VIG
    if VIG is None:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2)
        a = np.clip((r - 0.55) / 0.9, 0, 1) ** 1.6 * 0.72
        a = np.maximum(a, np.clip((y - 700) / 380, 0, 1) ** 1.3 * 0.78)       # низ под субтитры
        a = np.maximum(a, np.clip((140 - y) / 140, 0, 1) * 0.35)
        VIG = Image.fromarray((a * 255).astype(np.uint8), 'L')
    return VIG
BLACK = Image.new('RGB', (W, H), (4, 3, 8))

_glow = {}
def card_glow(cw, ch, color):
    key = (cw, ch, color)
    if key not in _glow:
        g = Image.new('RGBA', (cw + 160, ch + 160), (0, 0, 0, 0)); d = ImageDraw.Draw(g)
        d.rectangle((70, 70, 70 + cw, 70 + ch), fill=color + (200,))
        _glow[key] = g.filter(ImageFilter.GaussianBlur(34))
    return _glow[key]

def draw_scene(sc, t):
    k = sc['kind']; u = (t - sc['t0']) / max(0.01, sc['t1'] - sc['t0'])
    if k == 'hook':
        im, _ = load_img(sc['img']); e = ease(u) if sc['t1'] - sc['t0'] > 2 else u
        (px0, py0), (px1, py1) = sc['p']; z = sc['z'][0] + (sc['z'][1] - sc['z'][0]) * e
        box = cover_box(im.size, e, px0 + (px1 - px0) * e, py0 + (py1 - py0) * e, z)
        fr = im.resize((W, H), Image.BICUBIC, box=box)
        if sc['cap']:
            lay = text_layer(sc['cap'], 92, RED, 1500)
            dtt = t - sc['t0'] - 0.5
            if dtt > 0: paste_alpha(fr, lay, W / 2, 800, min(1, dtt / 0.3), 1.0 + 0.15 * (1 - eo3(dtt / 0.5)))
        return fr
    if k == 'question':
        fr = BLACK.copy(); words = [('Куда', None), ('на', None), ('самом', None), ('деле', None), ('исчезают', RED), ('души', VIOLET), ('Shadow', RED), ('Fiend?', RED)]
        f = font('Playfair', 104, 'Bold'); d = ImageDraw.Draw(fr)
        lines = [words[:4], words[4:]]
        y = 330; tt = t - (sc['t0'] + 0.25); idx = 0
        for ln in lines:
            sp = d.textlength(' ', font=f); tw = sum(d.textlength(w, font=f) for w, _ in ln) + sp * (len(ln) - 1); x = (W - tw) / 2
            for w, c in ln:
                a = min(1, max(0, (tt - idx * 0.27) / 0.25)); idx += 1
                if a > 0:
                    col = c or WHITE; lay = Image.new('RGBA', (int(d.textlength(w, font=f)) + 120, 200), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay)
                    if c:
                        ld.text((60, 40), w, font=f, fill=col + (255,), stroke_width=5, stroke_fill=col + (255,)); lay = lay.filter(ImageFilter.GaussianBlur(14))
                        ld = ImageDraw.Draw(lay); lay2 = Image.new('RGBA', lay.size, (0, 0, 0, 0))
                    top = Image.new('RGBA', lay.size, (0, 0, 0, 0)); ImageDraw.Draw(top).text((60, 40), w, font=f, fill=col + (255,))
                    lay = Image.alpha_composite(lay, top)
                    lay.putalpha(lay.getchannel('A').point(lambda v: int(v * a)))
                    fr.paste(lay, (int(x - 60), int(y - 40 + (1 - eo3(a)) * 18)), lay)
                x += d.textlength(w, font=f) + sp
            y += 150
        return fr
    if k == 'outro':
        im, _ = load_img(24); fr = im.resize((W, H), Image.BILINEAR, box=cover_box(im.size, 0, 0.5, 0.3 + 0.2 * u, 1.0 + 0.1 * u))
        fr = Image.fromarray((np.asarray(fr).astype(np.float32) * 0.22).astype(np.uint8))
        a0, b0 = line_rng[54]; a1, b1 = line_rng[55]
        for (a, b, y, size) in ((a0, b0, 380, 92), (a1, b1, 600, 120)):
            toks = list(range(a, b)); f = font('Playfair', size, 'Bold'); d = ImageDraw.Draw(fr)
            sp = d.textlength(' ', font=f); tw = sum(d.textlength(WORDS[j]['w'], font=f) for j in toks) + sp * (len(toks) - 1); x = (W - tw) / 2
            for j in toks:
                w = WORDS[j]['w']; al = min(1, max(0, (t - (WORDS[j]['s'] - 0.12)) / 0.4)); c = word_color(w) or WHITE
                if 'коллекц' in norm(w): c = RED
                if al > 0:
                    lay = text_layer_word(w, f, c); fr.paste(lay, (int(x - 40), int(y - 40)), alpha_scale(lay, al))
                x += d.textlength(w, font=f) + sp
        return fr
    # main
    im, bg = load_img(sc['img']); mode = sc['mode']; (px0, py0), (px1, py1) = pan_for(sc); e = ease(u)
    if mode == 'cover':
        z = 1.0 + 0.11 * e
        fr = im.resize((W, H), Image.BICUBIC, box=cover_box(im.size, e, px0 + (px1 - px0) * e, py0 + (py1 - py0) * e, z))
        tx, ty, tw_ = W / 2, 690, 1500
    else:
        fr = bg.copy(); iw, ih = im.size; asp = iw / ih
        ch = 780; cw = int(ch * asp)
        if cw > 800: cw = 800; ch = int(cw / asp)
        cx = {'L': 560, 'R': 1360, 'C': W / 2}[sc['side']]; x0 = int(cx - cw / 2); y0 = 70 + (780 - ch) // 2
        g = card_glow(cw, ch, sc['acc']); fr.paste(g, (x0 - 70, y0 - 70), g)
        z = 1.0 + 0.12 * e; vw, vh = iw / z, ih / z
        bx = (iw - vw) * (0.5 + 0.25 * (e - 0.5) * (1 if sc['var'] == 0 else -1)); by = (ih - vh) * (py0 + (py1 - py0) * e)
        card = im.resize((cw, ch), Image.BICUBIC, box=(bx, by, bx + vw, by + vh)); fr.paste(card, (x0, y0))
        ImageDraw.Draw(fr).rectangle((x0 - 3, y0 - 3, x0 + cw + 2, y0 + ch + 2), outline=sc['acc'], width=3)
        tx = 1470 if sc['side'] == 'L' else 450; ty = 440; tw_ = 820
        if sc['side'] == 'C': tx, ty, tw_ = W / 2, 690, 1500
    # оверлеи
    ovs = sc['ovs']
    for oi, ov in enumerate(ovs):
        t_in = ov['t']; nxt = ovs[oi + 1]['t'] if oi + 1 < len(ovs) else None
        if t < t_in or t > sc['t1'] + 0.4: continue
        if nxt is not None and t > nxt - 0.02: continue
        if ov['t'] > sc['t1']: continue
        lay = text_layer(ov['text'], ov['size'], ov['color'], tw_)
        dt = t - t_in; p = dt / 0.38
        al = min(1, p * 2.5); xx, yy, scl = tx, ty, 1.0
        if ov['style'] == 'stamp': scl = 1.0 + 1.0 * (1 - eo3(p))
        else: xx = tx + (-90 if tx > W / 2 else 90) * (1 - eback(p)) * -1
        out = nxt if nxt else sc['t1']
        if t > out - 0.2: al *= max(0, (out - t) / 0.2)
        if al > 0: paste_alpha(fr, lay, xx, yy, al, scl)
    return fr

def text_layer_word(w, f, color):
    key = ('w', w, id(f), color)
    if key in _tc: return _tc[key]
    d0 = ImageDraw.Draw(Image.new('RGB', (4, 4))); lw = int(d0.textlength(w, font=f)) + 80; lh = int(f.size * 1.5) + 40
    glow = Image.new('RGBA', (lw, lh), (0, 0, 0, 0)); ImageDraw.Draw(glow).text((40, 40), w, font=f, fill=color + (255,), stroke_width=5, stroke_fill=color + (255,))
    glow = glow.filter(ImageFilter.GaussianBlur(14)); ga = np.asarray(glow).copy(); ga[..., 3] = (ga[..., 3] * 0.55).astype(np.uint8)
    top = Image.new('RGBA', (lw, lh), (0, 0, 0, 0)); ImageDraw.Draw(top).text((40, 40), w, font=f, fill=color + (255,))
    out = Image.alpha_composite(Image.fromarray(ga), top); _tc[key] = out; return out
def alpha_scale(lay, a):
    return lay.getchannel('A').point(lambda v: int(v * a))

# ---------- переходы ----------
TR_HALF = 0.375
DIST = None
def panel_overlay(fr, typ, p):
    """p in [-1,1]: -1..0 закрытие, 0..1 открытие"""
    global DIST
    arr = None
    pan = Image.new('RGB', (W, H), (8, 6, 14)); mask = Image.new('L', (W, H), 0); d = ImageDraw.Draw(mask)
    acc = (255, 40, 60) if typ != 'v' else (160, 90, 255)
    edge = Image.new('L', (W, H), 0); de = ImageDraw.Draw(edge)
    q = ease(abs(p) if p < 0 else p)
    if typ == 'h':
        if p < 0: d.rectangle((0, 0, W * ease(1 + p), H), fill=255); xe = W * ease(1 + p); de.rectangle((xe - 14, 0, xe, H), fill=255)
        else: x0 = W * ease(p); d.rectangle((x0, 0, W, H), fill=255); de.rectangle((x0, 0, x0 + 14, H), fill=255)
    elif typ == 'v':
        if p < 0: ye = H * ease(1 + p); d.rectangle((0, 0, W, ye), fill=255); de.rectangle((0, ye - 14, W, ye), fill=255)
        else: y0 = H * ease(p); d.rectangle((0, y0, W, H), fill=255); de.rectangle((0, y0, W, y0 + 14), fill=255)
    else:
        if DIST is None:
            y, x = np.mgrid[0:H, 0:W].astype(np.float32); DIST = np.sqrt((x - W / 2) ** 2 + (y - H / 2) ** 2)
        R = math.hypot(W / 2, H / 2) + 20
        r = R * ease(1 + p) if p < 0 else R * ease(p)
        m = (DIST <= r) if p < 0 else (DIST >= r)
        ring = (np.abs(DIST - r) < 9)
        mask = Image.fromarray((m * 255).astype(np.uint8), 'L'); edge = Image.fromarray((ring * 255).astype(np.uint8), 'L')
    fr = Image.composite(pan, fr, mask)
    return Image.composite(Image.new('RGB', (W, H), acc), fr, edge)

# ---------- кадр ----------
def scene_at(t):
    for i, s in enumerate(SCENES):
        if s['t0'] <= t < s['t1']: return i
    return len(SCENES) - 1

def frame(t):
    i = scene_at(t); sc = SCENES[i]; fr = draw_scene(sc, t)
    # виньетка + низ
    fr = Image.composite(Image.new('RGB', (W, H), (3, 2, 6)), fr, vignette())
    # субтитры (не в затравке и не в финале)
    if sc['kind'] == 'main':
        for pi, (s, e, idxs) in enumerate(PHRASES):
            if s <= t < e:
                lay = sub_layer(pi, cur_word(pi, t)); a = min(1, (t - s) / 0.12)
                paste_alpha(fr, lay, W / 2, 962, a); break
    # переходы по границам
    for j in (i, i + 1):
        if 1 <= j < len(SCENES):
            tb = SCENES[j]['t0']; typ = SCENES[j]['tr']; dtb = t - tb
            if typ in ('h', 'v', 'c') and abs(dtb) < TR_HALF:
                fr = panel_overlay(fr, typ, dtb / TR_HALF)
            elif typ == 'dip' and abs(dtb) < 0.12:
                a = 1 - abs(dtb) / 0.12; fr = Image.blend(fr, Image.new('RGB', (W, H), (4, 2, 8)), 0.75 * a)
            elif typ == 'flash' and -0.04 < dtb < 0.28:
                a = 1 - max(0, dtb) / 0.28; col = (255, 235, 225) if j in (2, 3) else (255, 60, 40)
                fr = Image.blend(fr, Image.new('RGB', (W, H), col), 0.6 * a)
    # прогресс, фейды
    d = ImageDraw.Draw(fr); d.rectangle((0, 0, int(W * t / END), 5), fill=(190, 40, 80))
    f = 1.0
    if t < 0.8: f = t / 0.8
    if t > END - 1.4: f = max(0, (END - t) / 1.4)
    if f < 1: fr = Image.blend(Image.new('RGB', (W, H), (0, 0, 0)), fr, f)
    return fr

# ---------- запуск ----------
def render_range(args):
    a, b, path = args
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '19', '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
    for n in range(a, b): p.stdin.write(frame(n / FPS).tobytes())
    p.stdin.close(); p.wait(); return path

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'scenes':
        for i, s in enumerate(SCENES): print(i, s['kind'], round(s['t0'], 1), round(s['t1'], 1), s.get('img'), s.get('mode'), s['tr'], round(s['t1'] - s['t0'], 1))
        print('END', END)
    elif mode == 'snap':
        os.makedirs('snaps', exist_ok=True)
        for t in map(float, sys.argv[2:]): frame(t).save(f'snaps/t{int(t * 10):05d}.jpg', quality=88)
    elif mode == 'full':
        from multiprocessing import Pool
        out = sys.argv[2]; total = int(END * FPS); step = 300; os.makedirs('tmpseg', exist_ok=True)
        jobs = [(a, min(total, a + step), f'tmpseg/seg{a // step:03d}.mp4') for a in range(0, total, step)]
        with Pool(4) as pool:
            for r in pool.imap(render_range, jobs): print('done', r, flush=True)
        with open('tmpseg/list.txt', 'w') as f:
            for j in jobs: f.write(f"file '{os.path.abspath(j[2])}'\n")
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', 'tmpseg/list.txt', '-i', 'mix.wav', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', out], check=True)
        print('written', out)
