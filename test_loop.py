import os
from TTS.utils.synthesizer import Synthesizer

tts_dir = os.path.join(os.path.dirname(__file__), 'models', 'custom_kiche_tts')
model_path = os.path.join(tts_dir, 'best_model.pth')
config_path = os.path.join(tts_dir, 'config.json')

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    use_cuda=False
)

text = "utz ib'ixik k'a pa ulew."

for i in range(5):
    wav = synthesizer.tts(text)
    synthesizer.save_wav(wav, f"test_loop_{i}.wav")
    print(f"Generated {i}")
