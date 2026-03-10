import os
import time
import requests as http_requests
from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import librosa
from transformers import Wav2Vec2ForCTC, AutoProcessor
from deep_translator import GoogleTranslator
from gtts import gTTS
import uuid

import json
import unicodedata

# ── HUGGINGFACE CONFIG ─────────────────────────────────────────────────────────
HF_TOKEN = None
_env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            if _line.startswith('HF_TOKEN='):
                HF_TOKEN = _line.strip().split('=', 1)[1]
                break
if HF_TOKEN:
    print(f"[HF] Token loaded ({HF_TOKEN[:8]}...)")
else:
    print("[HF] WARNING: No HF_TOKEN found in .env")

NLLB_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/nllb-200-distilled-600M"

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

load_medical_dict()

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
        # Check if dictionary phrase is contained in the input
        if norm_k and norm_k in norm_input:
            score = len(norm_k.split())
            if score > best_score:
                best_score = score
                best_match = v
    if best_match:
        return best_match, 'partial'

    return None, None

def translate_kiche_to_spanish(text):
    """Translate K'iche' → Spanish. Dictionary first, then NLLB API fallback."""
    if not text or not text.strip():
        return text

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


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global model variables
processor = None
model = None
device = None

def load_model():
    global processor, model, device
    if model is None:
        print("Loading MMS model...")
        model_id = "facebook/mms-1b-all"
        processor = AutoProcessor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        processor.tokenizer.set_target_lang("quc-dialect_central")
        model.load_adapter("quc-dialect_central")
        print("Model loaded.")

@app.route('/')
def index():
    if model is None:
        load_model()
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)
    
    try:
        # ASR
        audio_input, _ = librosa.load(filepath, sr=16000)
        inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        ids = torch.argmax(outputs.logits, dim=-1)[0]
        transcription = processor.decode(ids)
        
        # Convert raw ASR output to ALMG standard K'iche' for display
        display_transcription = to_almg_display(transcription)

        if not transcription:
            transcription = "(No se detectó voz)"

        # Translation using NLLB-200 (proper K'iche' support)
        translation = translate_kiche_to_spanish(transcription)
        
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
        return jsonify({
            'original': spanish_text,
            'translation': kiche_translation
        })
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
]

def get_training_count():
    if not os.path.exists(TRAINING_CSV):
        return 0
    with open(TRAINING_CSV, encoding='utf-8') as f:
        return max(0, sum(1 for _ in f) - 1)  # subtract header

@app.route('/train')
def train_page():
    return render_template('train.html', phrases=TRAINING_PHRASES,
                           count=get_training_count())

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
    if os.path.exists(TRAINING_CSV):
        return send_file(TRAINING_CSV, as_attachment=True,
                         download_name='kiche_dataset.csv')
    return jsonify({'error': 'No dataset yet'}), 404

if __name__ == '__main__':
    print("Starting server... Model will load on first request.")
    app.run(debug=True, port=5000)
