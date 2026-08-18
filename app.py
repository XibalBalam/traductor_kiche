
import os
import random
import csv
import time
import json
import unicodedata
import requests as http_requests
import torch
import librosa
from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import Wav2Vec2ForCTC, AutoProcessor
from deep_translator import GoogleTranslator
from gtts import gTTS
import uuid
import rapidfuzz
import boto3
import io
import zipfile
from dotenv import load_dotenv


app = Flask(__name__)
ESCRITURA_CSV = os.path.join(os.path.dirname(__file__), 'escritura_dataset.csv')
DICT_PATH = os.path.join(os.path.dirname(__file__), 'medical_dictionary.json')

# --- ESCRITURA TRAINING PAGE ---

def get_escritura_vocab():
    with open(DICT_PATH, encoding='utf-8-sig') as f:
        raw = json.load(f)
        
    ya_respondidos = set()
    if os.path.exists(ESCRITURA_CSV):
        import csv
        with open(ESCRITURA_CSV, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ya_respondidos.add(row['espanol'].strip())
                
    vocab_pendientes = []
    vocab_general = []
    
    for section, entries in raw.items():
        if section.startswith('_') and section != '_pendientes_traduccion':
            continue
        for kiche, espanol in entries.items():
            base = espanol.split('(')[0].strip() if '(' in espanol and ')' in espanol else espanol.strip()
            
            # Skip if already translated and pending review
            if base in ya_respondidos:
                continue
                
            aclaracion = espanol[espanol.find('('):].strip() if '(' in espanol and ')' in espanol else None
            
            if section == '_pendientes_traduccion':
                vocab_pendientes.append((base, aclaracion))
            else:
                vocab_general.append((base, aclaracion))
                
    if vocab_pendientes:
        return vocab_pendientes
    return [("No hay palabras pendientes. Usa el botón de IA para generar más.", None)]


@app.route('/escritura', methods=['GET', 'POST'])
def escritura_page():
    vocab = get_escritura_vocab()
    success = False
    
    if request.method == 'POST':
        submitted_espanol = request.form.get('espanol', '').strip()
        kiche = request.form.get('kiche', '').strip()
        
        # Normalizar b con apóstrofe
        if 'b' in kiche and "b'" not in kiche:
            kiche = kiche.replace('b', "b'")
            
        variantes_existentes = set()
        if os.path.exists(ESCRITURA_CSV):
            with open(ESCRITURA_CSV, encoding='utf-8') as f:
                import csv
                reader = csv.DictReader(f)
                for row in reader:
                    if row['espanol'] == submitted_espanol:
                        variantes_existentes.add(row['kiche'])
                        
        if submitted_espanol and kiche and kiche not in variantes_existentes:
            write_header = not os.path.exists(ESCRITURA_CSV)
            with open(ESCRITURA_CSV, 'a', encoding='utf-8', newline='') as f:
                import csv
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['espanol', 'kiche'])
                writer.writerow([submitted_espanol, kiche])
            sync_file_to_s3(ESCRITURA_CSV, 'escritura_dataset.csv')
            success = True
            
            # Remove from pendientes_traduccion so it doesn't stay there forever
            if os.path.exists(DICT_PATH):
                import collections
                with open(DICT_PATH, 'r', encoding='utf-8-sig') as f:
                    diccionario = json.load(f, object_pairs_hook=collections.OrderedDict)
                if '_pendientes_traduccion' in diccionario:
                    keys_to_delete = []
                    for k, v in diccionario['_pendientes_traduccion'].items():
                        base_v = v.split('(')[0].strip() if '(' in v and ')' in v else v.strip()
                        if base_v == submitted_espanol:
                            keys_to_delete.append(k)
                    if keys_to_delete:
                        for k in keys_to_delete:
                            del diccionario['_pendientes_traduccion'][k]
                        with open(DICT_PATH, 'w', encoding='utf-8-sig') as f:
                            json.dump(diccionario, f, ensure_ascii=False, indent=4)
                        sync_file_to_s3(DICT_PATH, 'medical_dictionary.json')
                        load_medical_dict()

    # Siempre elegir una nueva palabra al final, asegurando que la palabra, aclaración y variantes coincidan
    import random
    palabra, aclaracion = random.choice(vocab)
    variantes_pendientes = []
    variantes_pendientes_set = set()
    variantes_aprobadas = []
    
    if os.path.exists(ESCRITURA_CSV):
        with open(ESCRITURA_CSV, encoding='utf-8') as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                if row['espanol'] == palabra:
                    kiche_norm = row['kiche'].replace('b', "b'") if 'b' in row['kiche'] and "b'" not in row['kiche'] else row['kiche']
                    if kiche_norm not in variantes_pendientes_set:
                        variantes_pendientes.append(kiche_norm)
                        variantes_pendientes_set.add(kiche_norm)
                        
    with open(DICT_PATH, encoding='utf-8-sig') as f:
        raw_dict = json.load(f)
        for cat, entries in raw_dict.items():
            if cat != '_pendientes_traduccion' and isinstance(entries, dict):
                for kiche, esp in entries.items():
                    base = esp.split('(')[0].strip() if '(' in esp and ')' in esp else esp.strip()
                    if base.lower() == palabra.lower():
                        variantes_aprobadas.append(kiche)
                        
    return render_template('escritura.html', palabra_espanol=palabra, aclaracion=aclaracion, variantes_pendientes=variantes_pendientes, variantes_aprobadas=variantes_aprobadas, success=success)

def get_recommended_category(espanol, raw_dict, categorias):
    espanol_lower = espanol.lower()
    for cat in categorias:
        if espanol in raw_dict.get(cat, {}).values():
            return cat
    if 'dolor' in espanol_lower or 'duele' in espanol_lower: return 'sintomas_dolor'
    elif '?' in espanol or '¿' in espanol: return 'preguntas_doctor'
    elif 'sangr' in espanol_lower or 'herida' in espanol_lower: return 'sangrado_heridas'
    elif 'mare' in espanol_lower or 'vómit' in espanol_lower or 'vomit' in espanol_lower: return 'mareo_vomito'
    elif 'fiebre' in espanol_lower or 'calentura' in espanol_lower: return 'fiebre'
    elif 'embaraz' in espanol_lower: return 'embarazo'
    elif 'golpe' in espanol_lower or 'caíd' in espanol_lower or 'cay' in espanol_lower: return 'caidas_golpes'
    return categorias[0] if categorias else ''

@app.route('/eliminar-pendiente-escritura', methods=['POST'])
def eliminar_pendiente_escritura():
    data = request.get_json()
    espanol_target = data.get('espanol')
    if not espanol_target:
        return jsonify({'error': 'Falta palabra'}), 400

    if os.path.exists(DICT_PATH):
        import collections
        with open(DICT_PATH, 'r', encoding='utf-8-sig') as f:
            diccionario = json.load(f, object_pairs_hook=collections.OrderedDict)
        
        if '_pendientes_traduccion' in diccionario:
            keys_to_delete = []
            for k, v in diccionario['_pendientes_traduccion'].items():
                base_v = v.split('(')[0].strip() if '(' in v and ')' in v else v.strip()
                if base_v == espanol_target:
                    keys_to_delete.append(k)
            
            for k in keys_to_delete:
                del diccionario['_pendientes_traduccion'][k]
                
            if keys_to_delete:
                with open(DICT_PATH, 'w', encoding='utf-8-sig') as f:
                    json.dump(diccionario, f, ensure_ascii=False, indent=4)
                sync_file_to_s3(DICT_PATH, 'medical_dictionary.json')
                load_medical_dict()
                return jsonify({'status': 'ok'})
                
    return jsonify({'error': 'No se encontró la palabra'}), 404

@app.route('/revisar')
def revisar_page():
    variantes = []
    if os.path.exists(ESCRITURA_CSV):
        with open(ESCRITURA_CSV, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                variantes.append(row)
    
    with open(DICT_PATH, encoding='utf-8-sig') as f:
        raw = json.load(f)
    categorias = [k for k in raw.keys() if not k.startswith('_')]
    
    for v in variantes:
        v['recomendada'] = get_recommended_category(v['espanol'], raw, categorias)
    
    return render_template('revisar.html', variantes=variantes, categorias=categorias)

@app.route('/aprobar', methods=['POST'])
def aprobar():
    espanol = request.form.get('espanol')
    kiche = request.form.get('kiche')
    categoria = request.form.get('categoria')
    
    if not espanol or not kiche or not categoria:
        return jsonify({'error': 'Faltan datos'}), 400
        
    # 1. Update dictionary
    import collections
    with open(DICT_PATH, encoding='utf-8-sig') as f:
        raw = json.load(f, object_pairs_hook=collections.OrderedDict)
    
    if categoria not in raw:
        raw[categoria] = collections.OrderedDict()
        
    raw[categoria][kiche] = espanol
    
    # Agregar a _escritura_aprobada para que aparezca en el entrenamiento de voz
    if '_escritura_aprobada' not in raw:
        raw['_escritura_aprobada'] = collections.OrderedDict()
    raw['_escritura_aprobada'][kiche] = espanol
    
    # If the Spanish phrase was in _pendientes_traduccion, we can optionally remove it
    if '_pendientes_traduccion' in raw:
        keys_to_delete = []
        for pk, pv in raw['_pendientes_traduccion'].items():
            if pv.strip() == espanol.strip():
                keys_to_delete.append(pk)
        for pk in keys_to_delete:
            del raw['_pendientes_traduccion'][pk]
            
    with open(DICT_PATH, 'w', encoding='utf-8') as f:
        json.dump(raw, f, indent=4, ensure_ascii=False)
    sync_file_to_s3(DICT_PATH, 'medical_dictionary.json')
        
    load_medical_dict() # Reload into memory
    
    # 2. Remove from CSV
    if os.path.exists(ESCRITURA_CSV):
        rows = []
        with open(ESCRITURA_CSV, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for r in reader:
                if len(r) >= 2 and not (r[0] == espanol and r[1] == kiche):
                    rows.append(r)
        
        with open(ESCRITURA_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(rows)
        sync_file_to_s3(ESCRITURA_CSV, 'escritura_dataset.csv')
            
    return jsonify({'status': 'ok'})

@app.route('/eliminar-frase-entrenamiento', methods=['POST'])
def eliminar_frase_entrenamiento():
    data = request.get_json()
    kiche = data.get('kiche')
    if not kiche:
        return jsonify({'error': 'Falta frase'}), 400

    if os.path.exists(DICT_PATH):
        with open(DICT_PATH, 'r', encoding='utf-8-sig') as f:
            diccionario = json.load(f, object_pairs_hook=collections.OrderedDict)
        
        deleted = False
        if '_escritura_aprobada' in diccionario:
            if kiche in diccionario['_escritura_aprobada']:
                del diccionario['_escritura_aprobada'][kiche]
                deleted = True
                
        # Opcional: buscar en otras categorías si fuera necesario, pero la IA genera para _escritura_aprobada
        
        if deleted:
            with open(DICT_PATH, 'w', encoding='utf-8-sig') as f:
                json.dump(diccionario, f, ensure_ascii=False, indent=4)
            sync_file_to_s3(DICT_PATH, 'medical_dictionary.json')
            return jsonify({'status': 'ok'})
        
    return jsonify({'error': 'No encontrada o no se puede borrar'}), 404

@app.route('/eliminar_variante', methods=['POST'])
def eliminar_variante():
    espanol = request.form.get('espanol')
    kiche = request.form.get('kiche')
    
    if os.path.exists(ESCRITURA_CSV):
        rows = []
        with open(ESCRITURA_CSV, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for r in reader:
                if len(r) >= 2 and not (r[0] == espanol and r[1] == kiche):
                    rows.append(r)
        
        with open(ESCRITURA_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(rows)
        sync_file_to_s3(ESCRITURA_CSV, 'escritura_dataset.csv')
            
    return jsonify({'status': 'ok'})

# ── HUGGINGFACE, GROQ & AWS S3 CONFIG ──────────────────────────────────────────
load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

if HF_TOKEN:
    print(f"[HF] Token loaded ({HF_TOKEN[:8]}...)")
else:
    print("[HF] WARNING: No HF_TOKEN found in .env")

if GROQ_API_KEY:
    print(f"[GROQ] API Key loaded ({GROQ_API_KEY[:8]}...)")
else:
    print("[GROQ] WARNING: No GROQ_API_KEY found in .env")

s3_client = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME:
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        print(f"[S3] Connected to bucket: {S3_BUCKET_NAME}")
    except Exception as e:
        print(f"[S3] Failed to connect: {e}")
else:
    print("[S3] WARNING: Missing AWS credentials. Training data will be saved locally only.")

NLLB_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/nllb-200-distilled-600M"

# ── S3 FILE SYNC ───────────────────────────────────────────────────────────────
def sync_file_from_s3(filename, filepath):
    if not s3_client or not S3_BUCKET_NAME: return
    try:
        s3_client.download_file(S3_BUCKET_NAME, filename, filepath)
        print(f"[S3] Downloaded {filename}")
    except Exception as e:
        print(f"[S3] Could not download {filename} (might not exist yet): {e}")

def sync_file_to_s3(filepath, filename):
    if not s3_client or not S3_BUCKET_NAME: return
    try:
        s3_client.upload_file(filepath, S3_BUCKET_NAME, filename)
        print(f"[S3] Uploaded {filename}")
    except Exception as e:
        print(f"[S3] Failed to upload {filename}: {e}")

# ── MEDICAL DICTIONARY ─────────────────────────────────────────────────────────
DICT_PATH = os.path.join(os.path.dirname(__file__), 'medical_dictionary.json')
_medical_dict = {}  # flat key → translation

def load_medical_dict():
    global _medical_dict
    if not os.path.exists(DICT_PATH):
        return
    with open(DICT_PATH, encoding='utf-8-sig') as f:
        raw = json.load(f)
    flat = {}
    for section, entries in raw.items():
        if section.startswith('_'):
            continue
        for kiche, spanish in entries.items():
            flat[kiche.lower()] = spanish
    _medical_dict = flat
    print(f"[Dict] Loaded {len(_medical_dict)} medical entries.")

# ── NORMALIZATION RULES ────────────────────────────────────────────────────────
RULES_PATH = os.path.join(os.path.dirname(__file__), 'normalization_rules.json')
NORMALIZATION_RULES = {}

def load_normalization_rules():
    global NORMALIZATION_RULES
    if os.path.exists(RULES_PATH):
        with open(RULES_PATH, encoding='utf-8') as f:
            NORMALIZATION_RULES = json.load(f)
    else:
        NORMALIZATION_RULES = {'nukab': "nuq'ab'", 'qax': "k'ax"}
        with open(RULES_PATH, 'w', encoding='utf-8') as f:
            json.dump(NORMALIZATION_RULES, f, indent=4)
        sync_file_to_s3(RULES_PATH, 'normalization_rules.json')

# Ejecutar sincronización al inicio
sync_file_from_s3('medical_dictionary.json', DICT_PATH)
sync_file_from_s3('escritura_dataset.csv', ESCRITURA_CSV)
sync_file_from_s3('normalization_rules.json', RULES_PATH)

TRAINING_FOLDER = 'training_data'
TRAINING_CSV = os.path.join(TRAINING_FOLDER, 'metadata.csv')
os.makedirs(TRAINING_FOLDER, exist_ok=True)
sync_file_from_s3('training_data/metadata.csv', TRAINING_CSV)

load_medical_dict()
load_normalization_rules()

def apply_normalization_rules(text):
    if not text: return text
    words = text.split()
    normalized_words = [NORMALIZATION_RULES.get(w.lower(), w) for w in words]
    return ' '.join(normalized_words)

def _normalize(text):
    """Normalize K'iche' text for dictionary matching.

    Strategy: simple stripping only. Both the ASR input AND the dictionary
    keys pass through the same function — so 'cꞌäx wakan' and 'cax wakan'
    both become 'cax wakan' and match correctly.

    MMS produces: ä/ë/ï/ö/ü vowels + ꞌ apostrophe variants
    We strip all of those to get a bare consonant+vowel string.
    """
    text = text.lower().strip()

    # Strip umlaut vowels added by MMS
    text = text.replace('ä', 'a').replace('ë', 'e').replace('ï', 'i')
    text = text.replace('ö', 'o').replace('ü', 'u')

    # Strip ALL apostrophe/glottal variants (U+02BC, U+02C0, U+2019, etc.)
    for apos in ["'", '\u02bc', '\u02c0', '\u0027', '\u2019', '\u02bb', 'ꞌ']:
        text = text.replace(apos, '')

    # Strip remaining diacritics
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')

    return ' '.join(text.split())

def to_almg_display(text):
    """Convert raw MMS ASR output to ALMG standard K'iche' orthography for display.
    
    Reversals of what MMS does:
      cꞌ → k'   (MMS writes k' as cꞌ)
      kꞌ → q'   (MMS writes q' as kꞌ)
      ä → a, ë → e, etc.
    This is ONLY for display — dictionary lookup still uses the raw text.
    """
    import re

    # 1. Unify apostrophe variants to standard '
    for apos in ['ꞌ', '\u02bc', '\u02c0', '\u2019', '\u02bb']:
        text = text.replace(apos, "'")

    # 2. Convert MMS umlaut vowels back to plain vowels
    text = text.replace('ä', 'a').replace('ë', 'e').replace('ï', 'i')
    text = text.replace('ö', 'o').replace('ü', 'u')

    # 3. cꞌ / c' → k'  (glottalized k)
    text = text.replace("<unk>", "'")
    text = text.replace("c'", "k'")

    # 4. kꞌ → q'  only when followed by a vowel (glottalized q/uvular)
    text = re.sub(r"k'([aeiou])", r"q'\1", text)

    # 5. Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    return text


def lookup_dictionary(text):
    """Try to find a translation in the medical dictionary.
    Returns (translation, confidence) or (None, 0).
    confidence: 'exact', 'normalized', 'partial', None
    """
    if not _medical_dict:
        return None, None

    # 1. Exact match
    key = text.lower().strip()
    if key in _medical_dict:
        return _medical_dict[key], 'exact'

    # 2. Normalized match (removes apostrophes, accents)
    norm_input = _normalize(text)
    norm_dict = {_normalize(k): v for k, v in _medical_dict.items()}
    if norm_input in norm_dict:
        return norm_dict[norm_input], 'normalized'

    # 3. Partial / word overlap match — if input contains a known key phrase
    best_match = None
    best_score = 0
    for k, v in _medical_dict.items():
        norm_k = _normalize(k)
        # Check if dictionary phrase is contained as complete words in the input
        if norm_k and f" {norm_k} " in f" {norm_input} ":
            score = len(norm_k.split())
            if score > best_score:
                best_score = score
                best_match = v
    if best_match:
        return best_match, 'partial'

    # 4. Levenshtein / Fuzzy Match as fallback
    try:
        best_match_key, score, _ = rapidfuzz.process.extractOne(norm_input, norm_dict.keys(), scorer=rapidfuzz.fuzz.ratio)
        if score >= 80:
            return norm_dict[best_match_key], f'rapidfuzz ({round(score)}%)'
    except Exception as e:
        print(f"[RapidFuzz] Error: {e}")

    return None, None

def translate_kiche_to_spanish(text):
    """Translate K'iche' → Spanish. Dictionary first, then NLLB API fallback."""
    if not text or not text.strip():
        return text

    # 0. Apply manual normalization rules first
    normalized_text = apply_normalization_rules(text)
    if normalized_text != text:
        print(f"[Norm] Replaced: '{text}' -> '{normalized_text}'")
        text = normalized_text

    # 1. Try medical dictionary
    translation, confidence = lookup_dictionary(text)
    if translation:
        print(f"[Dict] '{text}' → '{translation}' ({confidence})")
        return translation

    # 2. Fallback: NLLB-200 via HuggingFace router API
    print(f"[NLLB] '{text}' not in dictionary, using NLLB API...")
    if not HF_TOKEN:
        print("[NLLB] No token, skipping NLLB.")
        return f"[Sin traducción: {text}]"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": text,
        "parameters": {
            "src_lang": "quc_Latn",
            "tgt_lang": "spa_Latn"
        }
    }
    try:
        resp = http_requests.post(NLLB_API_URL, headers=headers, json=payload, timeout=30)
        print(f"[NLLB] Status: {resp.status_code}, Body: {resp.text[:200]}")
        if resp.status_code == 200:
            result = resp.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('translation_text', text)
            elif isinstance(result, dict) and 'translation_text' in result:
                return result['translation_text']
        # Any error → fallback
        print(f"[NLLB] Non-200 response, falling back.")
        return GoogleTranslator(source='auto', target='es').translate(text)
    except Exception as e:
        print(f"[NLLB] Exception: {e}")
        return GoogleTranslator(source='auto', target='es').translate(text)



UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global model variables
processor = None
model = None
tts_model = None
tts_tokenizer = None
device = None

import threading

model_lock = threading.Lock()
is_loading = False

def load_model():
    global processor, model, tts_model, tts_tokenizer, device, is_loading
    with model_lock:
        if model is not None or is_loading:
            return
        is_loading = True

    try:
        print("Loading base MMS model...")
        model_id = "facebook/mms-1b-all"
        processor = AutoProcessor.from_pretrained(model_id)
        base_model = Wav2Vec2ForCTC.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        processor.tokenizer.set_target_lang("quc-dialect_central")
        base_model.load_adapter("quc-dialect_central")
        
        # Intentar cargar pesos LoRA entrenados
        trained_model_dir = os.path.join(os.path.dirname(__file__), 'models', 'mms_kiche_trained')
        if os.path.exists(os.path.join(trained_model_dir, 'adapter_config.json')):
            print(f"Detectado modelo entrenado local en: {trained_model_dir}")
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(base_model, trained_model_dir)
                print("✅ Pesos LoRA cargados exitosamente sobre el modelo base.")
            except ImportError:
                print("❌ Error: La librería 'peft' no está instalada. Ejecuta 'pip install peft'. Usando modelo genérico.")
                model = base_model
            except Exception as e:
                print(f"❌ Error cargando modelo LoRA: {e}. Usando modelo genérico.")
                model = base_model
        else:
            print("No se encontró modelo entrenado localmente. Usando adaptador genérico de MMS.")
            model = base_model
            
        model.to(device)
        print(f"Model loaded on {device}.")

        if device == "cpu":
            print("Optimizing for CPU: limiting threads...")
            torch.set_num_threads(8)

        # Cargar TTS K'iche'
        print("Loading TTS model...")
        tts_dir = os.path.join(os.path.dirname(__file__), 'models', 'mms_tts_kiche')
        if os.path.exists(os.path.join(tts_dir, 'config.json')):
            try:
                from transformers import VitsModel, AutoTokenizer
                tts_model = VitsModel.from_pretrained(tts_dir)
                tts_tokenizer = AutoTokenizer.from_pretrained(tts_dir)
                tts_model.eval()
                print("✅ Modelo TTS K'iche' cargado exitosamente.")
            except Exception as e:
                print(f"❌ Error cargando modelo TTS: {e}")
        else:
            print("❌ No se encontró modelo TTS en models/mms_tts_kiche. Ejecuta download_tts.py")
    finally:
        with model_lock:
            is_loading = False

# Start loading the model in a background thread
threading.Thread(target=load_model, daemon=True).start()

@app.route('/status')
def status():
    return jsonify({
        'ready': model is not None,
        'message': 'Modelo listo' if model is not None else 'Cargando modelos en segundo plano...'
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    global model, processor, device
    if model is None:
        load_model()
        
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)
    
    try:
        # ASR
        audio_input, _ = librosa.load(filepath, sr=16000)
        
        # Prevent Wav2Vec2 "Kernel size can't be greater than actual input size" error for short audio
        if len(audio_input) < 1600:
            return jsonify({
                'transcription': "(No se detectó voz - el audio es muy corto)",
                'translation': "",
                'audio_url': None
            })
            
        inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        ids = torch.argmax(outputs.logits, dim=-1)[0]
        transcription = processor.decode(ids)
        
        # Convert raw ASR output to ALMG standard K'iche' for display
        display_transcription = to_almg_display(transcription)

        if not transcription:
            return jsonify({
                'transcription': "(No se detectó voz)",
                'translation': "",
                'audio_url': None
            })

        # Translation using NLLB-200 (proper K'iche' support)
        translation = translate_kiche_to_spanish(display_transcription)
        
        # TTS
        output_filename = f"{uuid.uuid4()}.mp3"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        tts = gTTS(text=translation, lang='es')
        tts.save(output_path)
        
        return jsonify({
            'transcription': display_transcription,
            'translation': translation,
            'audio_url': f"/audio/{output_filename}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/training-audio/<path:filepath>')
def get_training_audio(filepath):
    return send_from_directory(TRAINING_FOLDER, filepath)

@app.route('/reload-dict')
def reload_dict():
    load_medical_dict()
    return jsonify({'status': 'ok', 'entries': len(_medical_dict)})

@app.route('/translate-text', methods=['POST'])
def translate_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400
    try:
        # Use NLLB for K'iche'→Spanish retranslation
        translation = translate_kiche_to_spanish(text)
        output_filename = f"{uuid.uuid4()}.mp3"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        tts = gTTS(text=translation, lang='es')
        tts.save(output_path)
        return jsonify({
            'translation': translation,
            'audio_url': f"/audio/{output_filename}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate-to-kiche', methods=['POST'])
def translate_to_kiche():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    spanish_text = data['text'].strip()
    if not spanish_text:
        return jsonify({'error': 'Empty text'}), 400

    try:
        translator = GoogleTranslator(source='es', target='qu')
        kiche_translation = translator.translate(spanish_text)
        
        response_data = {
            'original': spanish_text,
            'translation': kiche_translation
        }

        # Generar TTS si el modelo está disponible
        if tts_model is not None and tts_tokenizer is not None:
            import scipy.io.wavfile as wavfile
            inputs = tts_tokenizer(kiche_translation, return_tensors="pt")
            with torch.no_grad():
                output = tts_model(**inputs)
            waveform = output.waveform[0].cpu().numpy()
            sr = tts_model.config.sampling_rate
            
            output_filename = f"{uuid.uuid4()}.wav"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            wavfile.write(output_path, sr, waveform)
            
            response_data['audio_url'] = f"/audio/{output_filename}"

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── TRAINING MODE ──────────────────────────────────────────────────────────────
TRAINING_FOLDER = 'training_data'
TRAINING_AUDIO_FOLDER = os.path.join(TRAINING_FOLDER, 'audio')
TRAINING_CSV = os.path.join(TRAINING_FOLDER, 'metadata.csv')
os.makedirs(TRAINING_AUDIO_FOLDER, exist_ok=True)

# Suggested phrases for training sessions
TRAINING_PHRASES = [
    ("k'ax waqan", "me duele la pierna / el pie"),
    ("k'ax nu jolom", "me duele la cabeza"),
    ("k'ax nu q'ab'", "me duele la mano"),
    ("k'ax nuk'ux", "me duele el pecho"),
    ("k'o nu q'aq'al", "tengo fiebre"),
    ("maj wuxlab", "no tengo aire / no puedo respirar"),
    ("kin jek' ta wuxlab", "no puedo respirar"),
    ("ka lemlot wanima", "estoy mareado"),
    ("kin xabik", "voy a vomitar"),
    ("xin tzaqik", "me caí"),
    ("xin q'osij", "me golpeé"),
    ("xin jos nu q'ab'", "me raspé la mano"),
    ("k'ax waral", "me duele aquí"),
    ("nim k'ax", "mucho dolor"),
    ("in yowab'", "estoy enfermo"),
    ("na in utz taj", "no estoy bien"),
    ("na toq'abej", "apóyame"),
    ("chi na to o'", "ven a ayudarme"),
    ("na toq'aj", "apóyame / sostenme"),
    ("waqan", "pierna / pie"),
    ("jolom", "cabeza"),
    ("q'ab'", "mano / brazo"),
    ("kik'", "sangre"),
    ("q'aq'al", "fiebre"),
    ("je'", "sí"),
    ("man je' taj", "no"),
    ("utz", "bien"),
    ("jas ab'i'", "¿cómo te llamas?"),
    ("jampa' ajunab'", "¿cuántos años tienes?"),
    ("jawije' at petinaq wi", "¿de dónde vienes?"),
    ("jas k'o awe", "¿qué tienes? / ¿qué te pasa?"),
    ("jas ana'om", "¿cómo te sientes?"),
]

def get_training_count():
    if not os.path.exists(TRAINING_CSV):
        return 0
    with open(TRAINING_CSV, encoding='utf-8') as f:
        return max(0, sum(1 for _ in f) - 1)  # subtract header

@app.route('/train')
def train_page():
    categorized_phrases = {
        "Frases Básicas": list(TRAINING_PHRASES)
    }

    if os.path.exists(DICT_PATH):
        with open(DICT_PATH, encoding='utf-8-sig') as f:
            raw_dict = json.load(f)
            
        aprobada = raw_dict.get('_escritura_aprobada', {})
        
        # Build reverse lookup to find which category the user selected
        cat_lookup = {}
        for cat, items in raw_dict.items():
            if not cat.startswith('_') and isinstance(items, dict):
                for k, v in items.items():
                    cat_lookup[(k, v)] = cat

        for k, v in aprobada.items():
            cat = cat_lookup.get((k, v), "Otras Aprobadas")
            formatted_cat = cat.replace('_', ' ').title()
            
            if formatted_cat not in categorized_phrases:
                categorized_phrases[formatted_cat] = []
                
            categorized_phrases[formatted_cat].append((k, f"✅ [Aprobado] {v}"))
            
    # Calcular cuántas veces se ha grabado cada frase
    phrase_counts = {}
    if os.path.exists(TRAINING_CSV):
        import csv
        with open(TRAINING_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get('transcription', '').strip()
                if t:
                    phrase_counts[t] = phrase_counts.get(t, 0) + 1
                
    return render_template('train.html', categories=categorized_phrases,
                           count=get_training_count(), phrase_counts=phrase_counts)

@app.route('/save-training-sample', methods=['POST'])
def save_training_sample():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio'}), 400
    audio_file = request.files['audio']
    correct_text = request.form.get('correct_text', '').strip()
    asr_raw = request.form.get('asr_raw', '').strip()

    if not correct_text:
        return jsonify({'error': 'No transcription provided'}), 400

    # Generate filename
    count = get_training_count() + 1
    filename = f"sample_{count:04d}.wav"
    audio_path = os.path.join(TRAINING_AUDIO_FOLDER, filename)
    audio_file.save(audio_path)

    # Save to CSV
    write_header = not os.path.exists(TRAINING_CSV)
    with open(TRAINING_CSV, 'a', encoding='utf-8', newline='') as f:
        import csv
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['file_name', 'transcription', 'asr_raw'])
        writer.writerow([f"audio/{filename}", correct_text, asr_raw])

    # Sync to S3 if configured
    if s3_client and S3_BUCKET_NAME:
        try:
            s3_client.upload_file(audio_path, S3_BUCKET_NAME, f"training_data/audio/{filename}")
            s3_client.upload_file(TRAINING_CSV, S3_BUCKET_NAME, "training_data/metadata.csv")
            print(f"[S3] Synced {filename} and metadata.csv to S3")
        except Exception as e:
            print(f"[S3] Upload error: {e}")

    return jsonify({
        'status': 'saved',
        'sample_number': count,
        'file': f"audio/{filename}",
        'total': count
    })

@app.route('/training-stats')
def training_stats():
    count = get_training_count()
    return jsonify({'total_samples': count,
                    'ready_for_training': count >= 50,
                    'progress_pct': min(100, int(count / 50 * 100))})

@app.route('/export-dataset')
def export_dataset():
    from flask import send_file
    if s3_client and S3_BUCKET_NAME:
        try:
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Add metadata.csv
                try:
                    csv_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key="training_data/metadata.csv")
                    zf.writestr("metadata.csv", csv_obj['Body'].read())
                except Exception as e:
                    print(f"No metadata.csv in S3: {e}")
                
                # Add audio files
                paginator = s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix="training_data/audio/"):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            file_key = obj['Key']
                            # Skip if it's just the folder itself
                            if file_key.endswith('/'): continue
                            
                            audio_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
                            file_name = file_key.split('/')[-1]
                            zf.writestr(f"audio/{file_name}", audio_obj['Body'].read())
            
            memory_file.seek(0)
            return send_file(memory_file, as_attachment=True, download_name='kiche_dataset_s3.zip', mimetype='application/zip')
        except Exception as e:
            print(f"[S3] Export error: {e}")
            return jsonify({'error': str(e)}), 500
    else:
        if os.path.exists(TRAINING_CSV):
            memory_file = io.BytesIO()
            with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(TRAINING_CSV, "metadata.csv")
                for f in os.listdir(TRAINING_AUDIO_FOLDER):
                    if f.endswith('.wav'):
                        zf.write(os.path.join(TRAINING_AUDIO_FOLDER, f), f"audio/{f}")
            memory_file.seek(0)
            return send_file(memory_file, as_attachment=True, download_name='kiche_dataset_local.zip', mimetype='application/zip')
        return jsonify({'error': 'No dataset yet'}), 404

@app.route('/generate-summary', methods=['POST'])
def generate_summary():
    data = request.get_json()
    if not data or 'symptoms' not in data:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    symptoms = data['symptoms']
    if not symptoms:
        return jsonify({'summary': 'No hay síntomas registrados.'})

    if not GROQ_API_KEY:
        # Fallback si no hay API KEY
        return jsonify({'summary': 'Síntomas reportados:\n- ' + '\n- '.join(symptoms)})
    
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        system_prompt = "Eres un asistente médico de triaje. Recibirás una lista de síntomas aislados. Tu única tarea es redactar un resumen clínico ejecutivo, estructurado y objetivo en un solo párrafo. No saludes, no des consejos médicos, solo resume los datos."
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Síntomas reportados: {', '.join(symptoms)}"}
            ],
            "temperature": 0.3
        }
        
        resp = http_requests.post(url, headers=headers, json=payload, timeout=15)
        if resp.status_code == 200:
            result = resp.json()
            summary = result['choices'][0]['message']['content'].strip()
            return jsonify({'summary': summary})
        else:
            print(f"[GROQ] Error API: {resp.text}")
            return jsonify({'summary': 'Síntomas reportados:\n- ' + '\n- '.join(symptoms)})
            
    except Exception as e:
        print(f"[GROQ] Exception: {e}")
        return jsonify({'summary': 'Síntomas reportados:\n- ' + '\n- '.join(symptoms)})

@app.route('/actualizar-reglas', methods=['GET', 'POST'])
def actualizar_reglas():
    if request.method == 'GET':
        return render_template('reglas.html', rules=NORMALIZATION_RULES)
    
    data = request.get_json()
    action = data.get('action')
    bad_text = data.get('bad_text', '').strip().lower()
    
    if action == 'add':
        good_text = data.get('good_text', '').strip()
        if bad_text and good_text:
            NORMALIZATION_RULES[bad_text] = good_text
    elif action == 'delete':
        if bad_text in NORMALIZATION_RULES:
            del NORMALIZATION_RULES[bad_text]
            
    with open(RULES_PATH, 'w', encoding='utf-8') as f:
        json.dump(NORMALIZATION_RULES, f, indent=4)
    sync_file_to_s3(RULES_PATH, 'normalization_rules.json')
        
    return jsonify({'status': 'ok'})

@app.route('/generar-vocabulario-ia', methods=['POST'])
def generar_vocabulario_ia():
    api_key = GROQ_API_KEY
    if not api_key:
        return jsonify({'error': 'No hay API Key de Groq configurada en tu archivo .env.'}), 500
        
    try:
        data = request.get_json() or {}
        instruccion_personalizada = data.get('instruccion', 'Genera 5 frases MUY CORTAS sobre síntomas básicos de triaje.').strip()
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        import collections, random, csv
        # Recopilar palabras existentes para evitar duplicados exactos
        palabras_existentes = set()
        with open(DICT_PATH, 'r', encoding='utf-8-sig') as f:
            diccionario_temp = json.load(f, object_pairs_hook=collections.OrderedDict)
            for cat, entries in diccionario_temp.items():
                if isinstance(entries, dict):
                    for kiche, esp in entries.items():
                        base = esp.split('(')[0].strip() if '(' in esp and ')' in esp else esp.strip()
                        palabras_existentes.add(base.lower())
                        
        if os.path.exists(ESCRITURA_CSV):
            with open(ESCRITURA_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    e = row.get('espanol', '').strip()
                    if e:
                        palabras_existentes.add(e.lower())
                    
        existentes_unicas = sorted(list(palabras_existentes))
        # Pasar TODAS las palabras existentes a la IA, ya que Llama-3 tiene suficiente contexto (8k+ tokens)
        existentes_sample = ", ".join(existentes_unicas)
        
        prompt = f"""
        INSTRUCCIÓN DEL USUARIO: {instruccion_personalizada}
        
        REGLAS CRÍTICAS DE NEGOCIO:
        1. NO generes NINGUNA de estas palabras que ya tenemos en nuestra base de datos (lee la lista completa y evítalas): [{existentes_sample}]
        2. Si la instrucción pide "partes del cuerpo" o similares, DEVUELVE ESTRICTAMENTE LA PALABRA EN SINGULAR (ej. "el ojo", "la mano", NO "los ojos", NO "manos").
        3. No uses puntos finales, no des explicaciones.
        
        REGLA CRÍTICA DE FORMATO: Devuelve SOLO un arreglo JSON estrictamente válido. 
        LA ÚNICA CLAVE PERMITIDA EN LOS OBJETOS ES "espanol".
        Bajo ninguna circunstancia uses la palabra generada como clave (no hagas {{"labio": "labio"}}).
        
        FORMATO EXACTO ESPERADO:
        [
            {{"espanol": "palabra uno"}},
            {{"espanol": "palabra dos"}},
            {{"espanol": "palabra tres"}}
        ]
        """
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt.strip()}],
            "temperature": 0.1, # Más bajo para mayor obediencia
            "max_tokens": 1024
        }
        
        response = http_requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        resp_data = response.json()
        content = resp_data['choices'][0]['message']['content'].strip()
        
        # Buscar el bloque JSON dentro de la respuesta (para ignorar texto conversacional)
        start_idx = content.find('[')
        end_idx = content.rfind(']')
        
        if start_idx != -1:
            if end_idx == -1 or end_idx < start_idx:
                # El LLM no terminó el JSON (falta el corchete de cierre). Intentamos repararlo.
                json_str = content[start_idx:]
                # Buscar la última llave de cierre válida
                last_brace = json_str.rfind('}')
                if last_brace != -1:
                    json_str = json_str[:last_brace+1] + ']'
                else:
                    json_str = '[]' # Falla silenciosa si no hay ni un objeto válido
            else:
                json_str = content[start_idx:end_idx+1]
                
            try:
                nuevas_palabras_raw = json.loads(json_str)
                # Filtro estricto: eliminar duplicados contra las palabras_existentes
                nuevas_palabras = []
                for item in nuevas_palabras_raw:
                    e = item.get('espanol', '').strip()
                    if e and e.lower() not in palabras_existentes:
                        nuevas_palabras.append(item)
                        palabras_existentes.add(e.lower()) # Evitar duplicados dentro del mismo batch
            except Exception as e:
                raise ValueError(f"No se pudo parsear el JSON: {json_str}. Error: {e}")
        else:
            raise ValueError("La IA no generó un JSON válido: " + content)
        
        if not nuevas_palabras:
            return jsonify({'error': 'La IA solo generó palabras que ya existen en tu diccionario o que ya descartaste. Intenta cambiar tu instrucción para ser más específico.'}), 400
            
        import collections, uuid
        with open(DICT_PATH, 'r', encoding='utf-8-sig') as f:
            diccionario = json.load(f, object_pairs_hook=collections.OrderedDict)
            
        if '_pendientes_traduccion' not in diccionario:
            diccionario['_pendientes_traduccion'] = collections.OrderedDict()
            
        for item in nuevas_palabras:
            e = item.get('espanol', '').strip()
            if e:
                key = f"IA_{uuid.uuid4().hex[:8]}"
                diccionario['_pendientes_traduccion'][key] = e
                
        with open(DICT_PATH, 'w', encoding='utf-8') as f:
            json.dump(diccionario, f, indent=4, ensure_ascii=False)
        sync_file_to_s3(DICT_PATH, 'medical_dictionary.json')
            
        load_medical_dict()
        
        return jsonify({'status': 'ok', 'palabras': nuevas_palabras})
        
    except Exception as e:
        print(f"[IA-VOCAB] Error: {e}")
        return jsonify({'error': str(e)}), 500

def sync_training_data_from_s3():
    """Download metadata.csv from S3 on startup so we don't start from count=0 on a fresh VPS reboot."""
    if s3_client and S3_BUCKET_NAME:
        try:
            s3_client.download_file(S3_BUCKET_NAME, "training_data/metadata.csv", TRAINING_CSV)
            print(f"[S3] Successfully synced metadata.csv from S3")
        except boto3.exceptions.botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("[S3] No metadata.csv found in bucket, starting fresh.")
            else:
                print(f"[S3] Error syncing metadata.csv: {e}")
        except Exception as e:
            print(f"[S3] Sync error: {e}")

if __name__ == '__main__':
    sync_training_data_from_s3()
    print("Starting server... Model will load on first request.")
    app.run(debug=True, port=5001)
