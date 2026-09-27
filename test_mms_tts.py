import os
import torch
import scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer

model_id = "facebook/mms-tts-quc-dialect_central"

try:
    print(f"Loading {model_id}...")
    model = VitsModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    text = "saq q'ab' wuj ij waqan tz'ikin k'ax"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        output = model(**inputs).waveform
        
    audio_data = output.squeeze().cpu().numpy()
    scipy.io.wavfile.write("test_mms_quc.wav", model.config.sampling_rate, audio_data)
    print("Success! Generated test_mms_quc.wav")
except Exception as e:
    print(f"Error: {e}")
