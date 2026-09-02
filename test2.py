import json
with open('medical_dictionary.json', 'r') as f:
    d = json.load(f)
aprobada = d.get('_escritura_aprobada', {})
print(f"Total aprobadas: {len(aprobada)}")
for k, v in list(aprobada.items())[:5]:
    found = False
    for cat, items in d.items():
        if cat != '_escritura_aprobada' and not cat.startswith('_') and isinstance(items, dict):
            if k in items and items[k] == v:
                print(f"'{k}' found in '{cat}'")
                found = True
                break
    if not found:
        print(f"'{k}' not found in any category!")
