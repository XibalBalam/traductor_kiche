import os
import torch
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

text = "k'ax"

def generate(ns, ns_dp, filename):
    with torch.no_grad():
        torch.manual_seed(0)
        wav = synthesizer.tts(
            text, 
            split_sentences=False, 
            noise_scale=ns, 
            noise_scale_dp=ns_dp,
            length_scale=0.85
        )
    synthesizer.save_wav(wav, filename)

generate(0.1, 0.1, "test_01.wav")
generate(0.333, 0.333, "test_0333.wav")
generate(0.667, 0.8, "test_0667.wav")
generate(1.0, 1.0, "test_10.wav")
print("Done testing scales.")
