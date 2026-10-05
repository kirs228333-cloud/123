"""Склеивает TTS-куски в vo.wav (начало озвучки = HOOK секунд), пишет chunks.json с границами кусков."""
import json, wave, numpy as np, os, glob

HOOK = 10.4          # длина затравки до начала голоса
SR = 24000
GAP = 0.55           # пауза между кусками
S = json.load(open('script.json', encoding='utf-8'))
texts = S['chunks'] + S['outro']

def load(f):
    w = wave.open(f); a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768; w.close(); return a

def trim(a, thr=0.012, pad=0.06):
    idx = np.where(np.abs(a) > thr)[0]
    if len(idx) == 0: return a
    s = max(0, idx[0] - int(pad * SR)); e = min(len(a), idx[-1] + int(pad * SR)); return a[s:e]

out = [np.zeros(int(HOOK * SR), np.float32)]; t = HOOK; meta = []
for i, tx in enumerate(texts):
    a = trim(load(f'tts/{i:02d}.wav'))
    f = int(0.012 * SR); a[:f] *= np.linspace(0, 1, f); a[-f:] *= np.linspace(1, 0, f)
    meta.append({'i': i, 'start': round(t, 3), 'end': round(t + len(a) / SR, 3), 'text': tx})
    out.append(a); t += len(a) / SR
    gap = GAP + (0.5 if i == len(texts) - 3 else 0) + (0.3 if i == len(texts) - 2 else 0)
    out.append(np.zeros(int(gap * SR), np.float32)); t += gap
vo = np.concatenate(out)
peak = np.abs(vo).max(); vo = vo / peak * 0.9
w = wave.open('vo.wav', 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((vo * 32767).astype(np.int16).tobytes()); w.close()
json.dump({'hook': HOOK, 'total': len(vo) / SR, 'chunks': meta}, open('chunks.json', 'w'), ensure_ascii=False, indent=1)
print('vo.wav', round(len(vo) / SR, 1), 's')
