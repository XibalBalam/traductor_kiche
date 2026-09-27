import os
from TTS.utils.synthesizer import Synthesizer

tts_dir = os.path.join(os.path.dirname(__file__), 'models', 'custom_kiche_tts')
model_path = os.path.join(tts_dir, 'best_model.pth')
config_path = os.path.join(tts_dir, 'config.json')

if os.path.exists(model_path) and os.path.exists(config_path):
    print("Cargando modelo Synthesizer...")
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        use_cuda=False
    )
    
    text = "utz ib'ixik k'a pa ulew."
    text = text.replace("ꞌ", "'").replace("’", "'")
    print(f"Sintetizando: {text}")
    
    output_path = "test_custom.wav"
    wav = synthesizer.tts(text)
    synthesizer.save_wav(wav, output_path)
    print(f"¡Éxito! Audio guardado en {output_path}")
else:
    print(f"Error: No se encontraron los archivos en {tts_dir}")
