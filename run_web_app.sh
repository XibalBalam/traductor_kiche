#!/bin/bash
# Script para ejecutar la aplicación web del traductor K'iche'-Español

cd "$(dirname "$0")"
source venv/bin/activate
echo "Iniciando servidor Flask..."
echo "La aplicación estará disponible en: http://localhost:5000"
python app.py
