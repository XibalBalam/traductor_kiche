from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile

try:
    model_name = "facebook/mms-tts-quc"
    print(f"Loading {model_name}...")
    model = VitsModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print("Error:", e)
