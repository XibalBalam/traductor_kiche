"""
Carga los modelos entrenados en Colab y los integra con la app.

Uso:
  1. Descarga mms_kiche_trained.zip y mms_tts_kiche.zip de Colab
  2. Descomprime ambos ZIPs en la carpeta models/ de este proyecto
  3. Ejecuta: python load_trained_model.py

Estructura esperada:
  models/
    mms_kiche_trained/    (modelo ASR afinado)
    mms_tts_kiche/        (modelo TTS K'iche')
"""

import os
import sys
import torch

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
ASR_MODEL_DIR = os.path.join(MODELS_DIR, 'mms_kiche_trained')
TTS_MODEL_DIR = os.path.join(MODELS_DIR, 'mms_tts_kiche')


def check_models():
    """Verifica que los modelos existan."""
    print("=== Verificando modelos ===\n")

    asr_ok = os.path.exists(os.path.join(ASR_MODEL_DIR, 'adapter_config.json'))
    tts_ok = os.path.exists(os.path.join(TTS_MODEL_DIR, 'config.json'))

    if asr_ok:
        print(f"  ✅ ASR (reconocimiento de voz): {ASR_MODEL_DIR}")
    else:
        print(f"  ❌ ASR no encontrado en: {ASR_MODEL_DIR}")
        print(f"     Descarga mms_kiche_trained.zip de Colab y descomprímelo ahí.")

    if tts_ok:
        print(f"  ✅ TTS K'iche' (síntesis de voz): {TTS_MODEL_DIR}")
    else:
        print(f"  ❌ TTS no encontrado en: {TTS_MODEL_DIR}")
        print(f"     Descarga mms_tts_kiche.zip de Colab y descomprímelo ahí.")

    return asr_ok, tts_ok


def test_asr():
    """Prueba el modelo ASR afinado."""
    print("\n=== Probando ASR (reconocimiento de voz K'iche') ===\n")

    from transformers import Wav2Vec2ForCTC, AutoProcessor
    from peft import PeftModel

    MODEL_ID = "facebook/mms-1b-all"
    TARGET_LANG = "quc-dialect_central"

    print("Cargando modelo base MMS...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    base_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

    processor.tokenizer.set_target_lang(TARGET_LANG)
    base_model.load_adapter(TARGET_LANG)

    print("Cargando pesos LoRA entrenados...")
    model = PeftModel.from_pretrained(base_model, ASR_MODEL_DIR)
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    print(f"  Modelo cargado en: {device}")
    print(f"  ✅ ASR listo para usar")
    return processor, model, device


def test_tts():
    """Prueba el modelo TTS K'iche'."""
    print("\n=== Probando TTS (síntesis de voz K'iche') ===\n")

    from transformers import VitsModel, AutoTokenizer
    import soundfile as sf

    print("Cargando modelo TTS K'iche'...")
    tts_model = VitsModel.from_pretrained(TTS_MODEL_DIR)
    tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_DIR)
    tts_model.eval()

    # Generar audio de prueba
    test_text = "sib'alaj maltyox"
    print(f"  Generando audio para: '{test_text}'")

    inputs = tts_tokenizer(test_text, return_tensors="pt")
    with torch.no_grad():
        output = tts_model(**inputs)

    waveform = output.waveform[0].cpu().numpy()
    sr = tts_model.config.sampling_rate

    output_path = os.path.join(os.path.dirname(__file__), 'test_tts_kiche.wav')
    sf.write(output_path, waveform, sr)

    print(f"  Audio guardado: {output_path}")
    print(f"  Duración: {len(waveform)/sr:.1f}s")
    print(f"  ✅ TTS K'iche' funcionando")

    return tts_model, tts_tokenizer


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    asr_ok, tts_ok = check_models()

    if not asr_ok and not tts_ok:
        print("\n❌ No hay modelos para probar.")
        print("Entrena primero en Google Colab con colab_train_asr.ipynb")
        sys.exit(1)

    if asr_ok:
        test_asr()

    if tts_ok:
        test_tts()

    print("\n" + "=" * 50)
    print("✅ ¡Modelos verificados!")
    print("=" * 50)
    print("\nPara usar en la app web:")
    print("  1. Los modelos se cargarán automáticamente al iniciar")
    print("  2. Ejecuta: python app.py")
    print("  3. Abre: http://localhost:5000")


if __name__ == '__main__':
    main()
