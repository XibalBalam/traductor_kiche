# Traductor K'iche' - Español (PoC)

Esta es una prueba de concepto para traducir audio K'iche' a voz en Español.

## Requisitos
- Python 3.8+
- Conexión a Internet (para descargar el modelo MMS y usar Google Translate).
- Un archivo de audio `.wav` con voz en K'iche'.

## Instalación

1.  El entorno virtual ya debería estar configurado. Si no:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```

## Uso

1.  Coloca tu archivo de audio (ej. `mi_audio.wav`) en esta carpeta.
2.  Ejecuta el script:
    ```bash
    venv\Scripts\python translator_poc.py mi_audio.wav
    ```

## Notas
- La primera ejecución tardará unos minutos en descargar el modelo de Meta (1GB+).
- Si el audio no se reconoce, intenta hablar más claro o reducir el ruido de fondo.
