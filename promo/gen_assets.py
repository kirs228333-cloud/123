import wave, base64, sys
from google import genai
from google.genai import types
env=dict(l.strip().split('=',1) for l in open('.env') if '=' in l)
c=genai.Client(api_key=env['GEMINI_API_KEY_1'])
TEXT=open('script.txt',encoding='utf-8').read().strip()
r=c.models.generate_content(model='gemini-2.5-flash-preview-tts',
  contents='Прочитай глубоким низким мужским голосом, медленно, мрачно и кинематографично, с паузами:\n'+TEXT,
  config=types.GenerateContentConfig(response_modalities=['AUDIO'],
   speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
     prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Charon')))))
pcm=r.candidates[0].content.parts[0].inline_data.data
w=wave.open('vo_raw.wav','wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(pcm); w.close()
print('tts ok')
PROMPTS={
'001':'Close-up cinematic portrait of Shadow Fiend from Dota 2, a towering demon with glowing red eyes, flaming-orange energy, black smoke, dark fantasy, dramatic lighting, 16:9',
'002':'Swarm of ghostly violet souls with screaming faces swirling through black smoke around a dark demon silhouette, dark fantasy, 16:9',
'003':'Demon unleashing Requiem of Souls: expanding rings of red fire and spectral souls bursting outward from a dark demon, black background, 16:9',
'004':'The Abyss: a vast dark void with a swirling vortex of purple mist, tormented souls falling into darkness, eerie, cinematic, 16:9',
'005':'Horde of horned demons in a hellish landscape lit by fire, black smoke, dark fantasy painting, 16:9',
'006':'Lone shadowy demon with burning red eyes standing before a giant glowing violet soul-mass, ominous, cinematic, 16:9',
}
for k,p in PROMPTS.items():
    try:
        r=c.models.generate_content(model='gemini-2.5-flash-image',contents=p,
          config=types.GenerateContentConfig(response_modalities=['IMAGE'],image_config=types.ImageConfig(aspect_ratio='16:9')))
        for part in r.candidates[0].content.parts:
            if part.inline_data: open(f'img/{k}.png','wb').write(part.inline_data.data); print('img',k); break
    except Exception as e: print('ERR',k,str(e)[:150])
