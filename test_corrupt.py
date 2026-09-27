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

# Force model to FP32 if it's not
synthesizer.tts_model.float()

text = "ij"
wav = synthesizer.tts(text, split_sentences=False)
wav_arr = np.array(wav)
print(f"Max amp for ij: {np.max(np.abs(wav_arr))}")
print(f"Contains NaNs: {np.isnan(wav_arr).any()}")

text = "waqan"
wav = synthesizer.tts(text, split_sentences=False)
wav_arr = np.array(wav)
print(f"Max amp for waqan: {np.max(np.abs(wav_arr))}")

