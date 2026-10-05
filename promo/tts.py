import wave, json, time, os
from google import genai
from google.genai import types
env=dict(l.strip().split('=',1) for l in open('.env') if '=' in l)
c=genai.Client(api_key=env['GEMINI_API_KEY_1'])
S=json.load(open('script.json',encoding='utf-8'))
items=S['chunks']+S['outro']
for i,t in enumerate(items):
    f=f'tts/{i:02d}.wav'
    if os.path.exists(f): continue
    for a in range(8):
        try:
            r=c.models.generate_content(model='gemini-2.5-flash-preview-tts',
              contents='Say in a deep, low, slow, dark and cinematic male voice, Russian, like a fantasy documentary narrator:\n'+t,
              config=types.GenerateContentConfig(response_modalities=['AUDIO'],
               speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                 prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name='Algenib')))))
            pcm=r.candidates[0].content.parts[0].inline_data.data
            w=wave.open(f,'wb');w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(pcm);w.close()
            print('ok',i,len(pcm)/48000,flush=True);break
        except Exception as e:
            print('retry',i,str(e)[:120],flush=True);time.sleep(20)
