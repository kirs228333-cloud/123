"""mix.wav: озвучка (с лёгкой обработкой) + тёмный дрон + whoosh на переходах + удары в затравке. Длина = длине ролика."""
import numpy as np, wave, subprocess, math
import render as R

SR = 44100; END = R.END; N = int(END * SR); rng = np.random.default_rng(7)
t = np.arange(N) / SR

# озвучка: чуть глубже, немного пространства
subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', 'vo.wav', '-af',
                'highpass=f=50,equalizer=f=110:t=q:w=1.0:g=3.5,equalizer=f=4500:t=q:w=1.5:g=-2,acompressor=threshold=-20dB:ratio=3:attack=10:release=120,aecho=0.8:0.6:60|130:0.2|0.1',
                '-ar', str(SR), '-ac', '1', 'vo_fx.wav'], check=True)
def rd(f):
    w = wave.open(f); a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768; w.close(); return a
vo = np.zeros(N, np.float32); v = rd('vo_fx.wav'); v = v[:N]; vo[:len(v)] = v
vo *= 0.95 / max(0.01, np.abs(vo).max())

# дрон
def lfo(f, ph): return 0.5 + 0.5 * np.sin(2 * np.pi * f * t + ph)
music = np.zeros(N, np.float32)
for f0, a, lf, ph in ((55.0, 0.30, 0.05, 0), (82.4, 0.22, 0.07, 1), (110.0, 0.16, 0.09, 2), (116.5, 0.10, 0.06, 3), (164.8, 0.07, 0.11, 4), (220.0, 0.04, 0.08, 5)):
    music += (a * (0.55 + 0.45 * lfo(lf, ph)) * (np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * (f0 * 1.004) * t))).astype(np.float32)
# ветер/шорох: сглаженный шум
nz = rng.standard_normal(N).astype(np.float32); k = 400; nz = np.convolve(nz, np.ones(k) / k, mode='same'); nz /= np.abs(nz).max()
music += 0.22 * nz * (0.4 + 0.6 * lfo(0.04, 1))
# медленный пульс в затравке
hook = R.HOOK
env_hook = np.clip(t / 7.6, 0, 1) ** 2.2
music *= (0.35 + 0.65 * (t < hook)) * 0.5 + 0.5 * (t >= hook) * 0.5
music *= 0.55
# громкость: затравка громче, под голосом тише
duck = np.convolve(np.abs(vo), np.ones(2200) / 2200, mode='same'); duck = np.clip(duck / 0.08, 0, 1)
gain = np.where(t < hook, 0.55 + 0.55 * env_hook, 0.30) * (1 - 0.45 * duck)
music *= gain.astype(np.float32)
fade_in = np.clip(t / 2.5, 0, 1); fade_out = np.clip((END - t) / 3.0, 0, 1); music *= fade_in * fade_out

fx = np.zeros(N, np.float32)
def add(sig, at):
    s = int(at * SR)
    if s < 0: sig = sig[-s:]; s = 0
    e = min(N, s + len(sig)); fx[s:e] += sig[:e - s]
def whoosh(dur=0.8, vol=0.22, up=True):
    n = int(dur * SR); x = np.arange(n) / SR; nz = rng.standard_normal(n).astype(np.float32)
    k = np.clip((1 - (x / dur) if not up else x / dur), 0.02, 1); k2 = 1 + (k * 40).astype(int)
    out = np.zeros(n, np.float32); sm = np.convolve(nz, np.ones(6) / 6, mode='same'); sm2 = np.convolve(nz, np.ones(60) / 60, mode='same') * 4
    out = sm * k + sm2 * (1 - k)
    env = np.sin(np.pi * np.clip(x / dur, 0, 1)) ** 1.5
    return (out * env * vol / max(1e-6, np.abs(out * env).max())).astype(np.float32)
def boom(vol=0.9, dur=2.2):
    n = int(dur * SR); x = np.arange(n) / SR
    s = np.sin(2 * np.pi * (48 * np.exp(-x * 0.6) + 30) * x) * np.exp(-x * 1.7)
    nz = np.convolve(rng.standard_normal(n), np.ones(30) / 30, mode='same') * np.exp(-x * 7) * 1.5
    return (vol * (s + nz) / 1.5).astype(np.float32)
def tick(vol=0.25):
    n = int(0.25 * SR); x = np.arange(n) / SR; return (vol * np.sin(2 * np.pi * 70 * x) * np.exp(-x * 18)).astype(np.float32)

# риз и удары затравки
n = int(7.6 * SR); xr = np.arange(n) / SR; rz = np.convolve(rng.standard_normal(n), np.ones(12) / 12, mode='same') * (xr / 7.6) ** 3
add((0.35 * rz / np.abs(rz).max()).astype(np.float32), 0)
add(boom(0.8), 5.2); add(boom(0.95), 7.6)
for sc in R.SCENES[1:]:
    tb = sc['t0']; tr = sc['tr']
    if tr in ('h', 'v', 'c'): add(whoosh(0.8, 0.26), tb - 0.4)
    elif tr == 'dip': add(tick(), tb - 0.05)
    elif tr == 'flash' and tb < 8: add(whoosh(0.35, 0.3), tb - 0.3)
# финальный удар на «коллекционирует»
last = R.WORDS[R.line_rng[55][0] + 2]['s'] if R.line_rng[55][1] - R.line_rng[55][0] > 2 else R.WORDS[-1]['s']
add(boom(0.55, 2.0), R.WORDS[R.line_rng[55][0]]['s'] - 0.1)

mix = vo * 1.0 + music + fx
mix = np.tanh(mix * 1.1) / np.tanh(1.1)
mix = mix / max(1e-6, np.abs(mix).max()) * 0.92
st = np.stack([mix, mix], 1)
w = wave.open('mix.wav', 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st * 32767).astype(np.int16).tobytes()); w.close()
print('mix.wav', round(N / SR, 2), 's')
