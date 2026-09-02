#!/bin/bash
# Script para ejecutar el traductor desde línea de comandos
# Uso: ./run_translator.sh nombre_archivo.wav

if [ "$#" -ne 1 ]; then
    echo "Uso: $0 archivo_audio.wav"
    exit 1
fi

cd "$(dirname "$0")"
source venv/bin/activate
echo "Procesando archivo: $1"
python translator_poc.py "$1"
