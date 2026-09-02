#!/bin/bash
# Empaqueta los datos de entrenamiento para subir a Google Colab
# Uso: ./export_training_data.sh

cd "$(dirname "$0")"

TRAINING_DIR="training_data"
OUTPUT_ZIP="training_data.zip"

# Verificar que existen datos
if [ ! -f "$TRAINING_DIR/metadata.csv" ]; then
    echo "Error: No se encontró $TRAINING_DIR/metadata.csv"
    echo "Primero graba muestras usando la interfaz /train de la app."
    exit 1
fi

AUDIO_COUNT=$(find "$TRAINING_DIR/audio" -name "*.wav" 2>/dev/null | wc -l | tr -d ' ')
CSV_LINES=$(wc -l < "$TRAINING_DIR/metadata.csv" | tr -d ' ')
SAMPLE_COUNT=$((CSV_LINES - 1))

echo "=== Exportar datos de entrenamiento ==="
echo "Muestras en CSV: $SAMPLE_COUNT"
echo "Archivos de audio: $AUDIO_COUNT"

if [ "$AUDIO_COUNT" -eq 0 ]; then
    echo ""
    echo "⚠️  No hay archivos de audio en training_data/audio/"
    echo "Los audios se deben grabar desde la interfaz /train de la app."
    echo "Ejecuta la app y ve a http://localhost:5000/train"
    exit 1
fi

# Crear ZIP
echo ""
echo "Creando $OUTPUT_ZIP..."
zip -r "$OUTPUT_ZIP" "$TRAINING_DIR/metadata.csv" "$TRAINING_DIR/audio/"

SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)
echo ""
echo "✅ Archivo creado: $OUTPUT_ZIP ($SIZE)"
echo ""
echo "Próximos pasos:"
echo "1. Sube $OUTPUT_ZIP a Google Drive (carpeta traductor_kiche/)"
echo "2. Abre colab_train_asr.ipynb en Google Colab"
echo "3. Ejecuta todas las celdas"
echo "4. Descarga el modelo entrenado"
echo "5. Colócalo en models/ y ejecuta: python load_trained_model.py"
