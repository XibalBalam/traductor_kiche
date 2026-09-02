import os
import sys

def download_mms_tts_quc():
    try:
        from transformers import VitsModel, AutoTokenizer
    except ImportError:
        print("Instalando transformers...")
        os.system(f"{sys.executable} -m pip install -q transformers")
        from transformers import VitsModel, AutoTokenizer

    model_id = "facebook/mms-tts-quc-dialect_central"
    save_dir = os.path.join(os.path.dirname(__file__), 'models', 'mms_tts_kiche')
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Descargando modelo TTS: {model_id}...")
    print("Esto puede tardar unos minutos ya que el modelo pesa alrededor de 140MB...")
    
    try:
        model = VitsModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print("Guardando en almacenamiento local...")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        print(f"✅ ¡Modelo TTS K'iche' descargado exitosamente en: {save_dir}!")
        print("Ya puedes verificarlo ejecutando: python load_trained_model.py")
    except Exception as e:
        print(f"❌ Error durante la descarga: {e}")

if __name__ == "__main__":
    download_mms_tts_quc()
