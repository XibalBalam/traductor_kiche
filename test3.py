import json

def format_category_name(key):
    return key.replace('_', ' ').title()

with open('medical_dictionary.json', 'r') as f:
    raw_dict = json.load(f)

categorized_phrases = {}
aprobada = raw_dict.get('_escritura_aprobada', {})

# Build reverse lookup for category
cat_lookup = {}
for cat, items in raw_dict.items():
    if not cat.startswith('_') and isinstance(items, dict):
        for k, v in items.items():
            cat_lookup[(k, v)] = cat

for k, v in aprobada.items():
    cat = cat_lookup.get((k, v), "Otras Aprobadas")
    formatted_cat = format_category_name(cat)
    if formatted_cat not in categorized_phrases:
        categorized_phrases[formatted_cat] = []
    categorized_phrases[formatted_cat].append((k, f"✅ [Aprobado] {v}"))

for cat, items in list(categorized_phrases.items())[:5]:
    print(f"Category: {cat}, Items: {len(items)}")

