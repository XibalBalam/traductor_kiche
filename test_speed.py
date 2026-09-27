import os
import torch
import numpy as np
from TTS.utils.synthesizer import Synthesizer

tts_dir = os.path.join(os.path.dirname(__file__), 'models', 'custom_kiche_tts')
model_path = os.path.join(tts_dir, 'best_model.pth')
config_path = os.path.join(tts_dir, 'config.json')

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    use_cuda=False
)
synthesizer.tts_model.float()

text = "k'ax nu jolom"
with torch.no_grad():
    torch.manual_seed(0)
    wav = synthesizer.tts(
        text, 
        split_sentences=False, 
        noise_scale=0.333, 
        noise_scale_dp=0.333,
        length_scale=0.85
    )
synthesizer.save_wav(wav, "test_speed.wav")
print("Done!")
