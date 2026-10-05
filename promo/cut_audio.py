"""Сокращение озвучки по смыслу: оставляем выбранные предложения, режем только в паузах (мин. RMS в окне 20 мс),
фейды 12 мс, пауза между кусками. Пишет vo.wav (с затравкой впереди), words.json, meta.json.
python cut_audio.py list   -> печатает предложения с таймкодами
python cut_audio.py build  -> собирает"""
import json, re, sys, wave
import numpy as np

HOOK = 10.4; SR = 24000; GAP = 0.34
S = json.load(open('script.json', encoding='utf-8'))
WF = json.load(open('words_full.json', encoding='utf-8'))
LINES = S['lines'] + S['outro']
# --- привязка токенов к строкам и предложениям ---
k = 0; sents = []          # sents[i] = {'line','k','a','b'}  токены [a,b)
for li, l in enumerate(LINES):
    n = len(l.split()); cur = k; sk = 0
    for j in range(k, k + n):
        WF[j]['line'] = li
        if re.search(r'[.!?…]["»]?$', WF[j]['w']) or j == k + n - 1:
            sents.append({'line': li, 'k': sk, 'a': cur, 'b': j + 1}); cur = j + 1; sk += 1
    k += n
assert k == len(WF)

# ---- ЧТО ОСТАВЛЯЕМ ----  line -> список индексов предложений (остальные из строки убираются); [] = строка убрана целиком
KEEP = json.load(open('keep.json', encoding='utf-8')) if len(sys.argv) > 2 or True else {}
KEEP = {int(a): b for a, b in KEEP.items()}

def kept(s): return s['line'] not in KEEP or s['k'] in KEEP[s['line']]

if sys.argv[1] == 'list':
    for i, s in enumerate(sents):
        t0, t1 = WF[s['a']]['s'], WF[s['b'] - 1]['e']
        print(f"L{s['line']}.{s['k']} [{t0:6.1f}-{t1:6.1f}] {t1 - t0:4.1f}s {'KEEP' if kept(s) else 'drop'} | {' '.join(w['w'] for w in WF[s['a']:s['b']])[:90]}")
    tot = sum(WF[s['b'] - 1]['e'] - WF[s['a']]['s'] for s in sents if kept(s))
    print('kept speech seconds:', round(tot, 1), 'words:', sum(s['b'] - s['a'] for s in sents if kept(s)))
    sys.exit()

w = wave.open('vo_full.wav'); full = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768; w.close()
def rms_min(t0, t1):
    a = int(max(0, t0) * SR); b = int(min(len(full) / SR, t1) * SR); win = int(0.02 * SR)
    best, bt = 1e9, (t0 + t1) / 2
    for p in range(a, max(a + 1, b - win), int(0.004 * SR)):
        r = float(np.sqrt(np.mean(full[p:p + win] ** 2)))
        if r < best: best, bt = r, (p + win / 2) / SR
    return bt, best

# группы подряд идущих оставленных предложений
groups = []; cur = None
for i, s in enumerate(sents):
    if kept(s):
        if cur and cur['last'] == i - 1: cur['last'] = i; cur['sents'].append(s)
        else:
            cur = {'first': i, 'last': i, 'sents': [s]}; groups.append(cur)
out = [np.zeros(int(HOOK * SR), np.float32)]; t_cur = HOOK; words = []; rep = []
for gi, g in enumerate(groups):
    a_tok = g['sents'][0]['a']; b_tok = g['sents'][-1]['b'] - 1
    prev_e = WF[a_tok - 1]['e'] if a_tok > 0 else 0.0
    nxt_s = WF[b_tok + 1]['s'] if b_tok + 1 < len(WF) else len(full) / SR
    tin, rin = rms_min(max(prev_e - 0.04, WF[a_tok]['s'] - 0.30), WF[a_tok]['s'] + 0.04) if a_tok > 0 else (max(0.0, WF[a_tok]['s'] - 0.15), 0)
    tout, rout = rms_min(WF[b_tok]['e'] - 0.04, min(nxt_s + 0.04, WF[b_tok]['e'] + 0.35))
    tin = min(tin, WF[a_tok]['s'] - 0.02); tout = max(tout, WF[b_tok]['e'] + 0.02)
    rep.append((round(tin, 2), round(tout, 2), round(rin * 1000, 1), round(rout * 1000, 1)))
    seg = full[int(tin * SR):int(tout * SR)].copy(); f = int(0.012 * SR); seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
    for j in range(a_tok, b_tok + 1):
        d = dict(WF[j]); d['s'] = round(d['s'] - tin + t_cur, 3); d['e'] = round(d['e'] - tin + t_cur, 3); words.append(d)
    out.append(seg); t_cur += len(seg) / SR
    gap = GAP + (0.45 if g['sents'][-1]['line'] in (52, 53) else 0) + (0.5 if g['sents'][-1]['line'] == 53 else 0)
    if gi + 1 < len(groups): out.append(np.zeros(int(gap * SR), np.float32)); t_cur += gap
vo = np.concatenate(out)
w = wave.open('vo.wav', 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((vo / max(np.abs(vo).max(), 1e-6) * 0.9 * 32767).astype(np.int16).tobytes()); w.close()
json.dump(words, open('words.json', 'w', encoding='utf-8'), ensure_ascii=False)
json.dump({'hook': HOOK, 'total': len(vo) / SR, 'voice_end': words[-1]['e']}, open('meta.json', 'w'))
print('groups', len(groups), 'vo', round(len(vo) / SR, 1), 's  voice end', words[-1]['e'])
print('cut points (in,out,rmsIn*1000,rmsOut*1000):'); [print(' ', r) for r in rep]
