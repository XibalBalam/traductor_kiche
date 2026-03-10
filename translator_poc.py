import os
import sys
import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2ForCTC, AutoProcessor
from deep_translator import GoogleTranslator
from gtts import gTTS
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def setup_model():
    """
    Load the Meta MMS model and processor for K'iche' (quc).
    """
    print("Change: Loading Meta MMS model (facebook/mms-1b-all)... This may take a while first time.")
    model_id = "facebook/mms-1b-all"
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model.to(device)
        
        # Set target language to K'iche' (quc) - using Central dialect as default
        processor.tokenizer.set_target_lang("quc-dialect_central")
        model.load_adapter("quc-dialect_central")
        
        return processor, model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def transcribe_audio(audio_path, processor, model, device):
    """
    Transcribe audio file from K'iche' to text.
    """
    print(f"Processing audio: {audio_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load and resample audio to 16kHz (required by MMS)
    audio_input, sample_rate = librosa.load(audio_path, sr=16000)
    
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    ids = torch.argmax(outputs.logits, dim=-1)[0]
    transcription = processor.decode(ids)
    
    return transcription

def translate_text(text, target_lang='es'):
    """
    Translate text from K'iche' to Spanish using deep-translator.
    """
    print(f"Translating: '{text}'...")
    try:
        # Try specific code 'quc' first, fallback to 'auto' if issues
        # Note: Google Translate often uses 'quc' for K'iche'
        translator = GoogleTranslator(source='auto', target=target_lang)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return f"[Error de traducción] {text}"

def text_to_speech(text, output_file="output_es.mp3"):
    """
    Convert Spanish text to speech and save as MP3.
    """
    print(f"Synthesizing speech: '{text}'")
    try:
        tts = gTTS(text=text, lang='es')
        tts.save(output_file)
        print(f"Audio saved to: {output_file}")
        return True
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python translator_poc.py <path_to_audio.wav>")
        # Create a dummy file for testing if none provided? 
        # For now just exit
        sys.exit(1)
        
    audio_path = sys.argv[1]
    
    # 1. Setup
    processor, model, device = setup_model()
    
    # 2. Transcribe (ASR)
    try:
        kiche_text = transcribe_audio(audio_path, processor, model, device)
        print(f"\n📝 Transcripción K'iche': {kiche_text}")
    except Exception as e:
        print(f"Transcription failed: {e}")
        sys.exit(1)
        
    if not kiche_text or len(kiche_text.strip()) == 0:
        print("No speech detected.")
        sys.exit(0)

    # 3. Translate
    spanish_text = translate_text(kiche_text)
    print(f"🌎 Traducción Español: {spanish_text}")
    
    # 4. Synthesize (TTS)
    text_to_speech(spanish_text)
    
    print("\n✅ Proceso completado.")

if __name__ == "__main__":
    main()
