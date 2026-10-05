"""Пословные таймкоды: Gemini transcribe по vo.wav (частями по ~300 c), затем выравнивание на известный текст сценария -> words.json"""
import json, time, subprocess, re, difflib, os, sys, wave
from google import genai

env = dict(l.strip().split('=', 1) for l in open('.env') if '=' in l)
keys = [v for k, v in env.items() if k.startswith('GEMINI_API_KEY')]
C = json.load(open('chunks.json', encoding='utf-8'))
total = C['total']

def transcribe(path, offset):
    for k in keys:
        c = genai.Client(api_key=k, http_options={'timeout': 600000}); up = None
        for attempt in range(6):
            try:
                up = c.files.upload(file=path, config={'mime_type': 'audio/wav'})
                it = c.interactions.create(model='gemini-3.5-transcribe',
                    input=[{'type': 'audio', 'uri': up.uri, 'mime_type': 'audio/wav'}],
                    generation_config={'transcription_config': {'language_codes': ['ru-RU'],
                        'mode': {'type': 'verbatim', 'timestamp_granularities': ['word']}}})
                while it.status in ('in_progress', 'queued'):
                    time.sleep(3); it = c.interactions.get(id=it.id)
                words = []
                for st in it.steps or []:
                    for ct in st.content or []:
                        for an in ct.annotations or []:
                            if an.type == 'word_info':
                                s = float(str(an.start_offset).removesuffix('s')); e = float(str(an.end_offset).removesuffix('s'))
                                words.append({'s': round(s + offset, 3), 'e': round(e + offset, 3), 'w': an.text.strip()})
                if up:
                    try: c.files.delete(name=up.name)
                    except Exception: pass
                return words
            except Exception as ex:
                print('ERR', str(ex)[:160], flush=True); time.sleep(25)
    return []

# режем по паузам между кусками (границы chunks.json), части <= ~300 c
parts = []; cur_s = 0.0
for ch in C['chunks']:
    if ch['end'] - cur_s > 280 and ch['start'] - cur_s > 0:
        parts.append((cur_s, ch['start'] - 0.2)); cur_s = ch['start'] - 0.2
parts.append((cur_s, total))
tw = []
for n, (a, b) in enumerate(parts):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(a), '-t', str(b - a), '-i', 'vo_full.wav', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', f'part{n}.wav'], check=True)
    ws = transcribe(f'part{n}.wav', a); print('part', n, len(ws), 'words', flush=True); tw += ws
json.dump(tw, open('transcript_words.json', 'w', encoding='utf-8'), ensure_ascii=False)

# --- выравнивание на известный текст ---
S = json.load(open('script.json', encoding='utf-8'))
def norm(w): return re.sub(r'[^0-9a-zа-я]', '', w.lower().replace('ё', 'е'))
known = []   # (token, line_idx or 'outro', chunk_idx)
for ci, ch in enumerate(C['chunks']):
    for w in ch['text'].split(): known.append({'w': w, 'chunk': ci})
kn = [norm(k['w']) for k in known]; tn = [norm(w['w']) for w in tw]
sm = difflib.SequenceMatcher(None, kn, tn, autojunk=False)
for blk in sm.get_matching_blocks():
    for d in range(blk.size):
        known[blk.a + d]['s'] = tw[blk.b + d]['s']; known[blk.a + d]['e'] = tw[blk.b + d]['e']
matched = sum('s' in k for k in known); print('matched', matched, '/', len(known))
# интерполяция + привязка к границам кусков
for ci, ch in enumerate(C['chunks']):
    idx = [i for i, k in enumerate(known) if k['chunk'] == ci]
    ts = [(i, known[i]['s'], known[i]['e']) for i in idx if 's' in known[i]]
    anchors = [(-1, ch['start'], ch['start'])] + ts + [(10**9, ch['end'], ch['end'])]
    for (ia, sa, ea), (ib, sb, eb) in zip(anchors, anchors[1:]):
        gap = [i for i in idx if ia < i < ib]
        if not gap: continue
        lens = [max(2, len(norm(known[i]['w']))) for i in gap]; tot = sum(lens); t0 = ea; span = max(0.05 * len(gap), sb - ea); acc = 0
        for i, L in zip(gap, lens):
            known[i]['s'] = round(t0 + span * acc / tot, 3); acc += L; known[i]['e'] = round(t0 + span * acc / tot, 3)
for i in range(len(known) - 1):      # монотонность
    if known[i + 1]['s'] < known[i]['s']: known[i + 1]['s'] = known[i]['s']
    if known[i]['e'] > known[i + 1]['s'] and known[i]['e'] - known[i + 1]['s'] < 0.3: known[i]['e'] = known[i + 1]['s']
json.dump(known, open('words_full.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('words.json', len(known), 'last end', known[-1]['e'])
