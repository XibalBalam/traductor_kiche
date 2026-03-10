# Traductor K'iche' - Español (PoC)

Esta es una prueba de concepto para traducir audio K'iche' a voz en Español.

## Requisitos
- Python 3.8+
- Un archivo de audio `.wav` con voz en K'iche'.
- **Importante:** Necesitas conexión a internet para la primera ejecución (descarga del modelo MMS ~1GB).

## Instalación en una nueva computadora

1.  **Clonar el repositorio:**
    ```bash
    git clone <URL_DE_TU_REPOSITORIO>
    cd translator_poc
    ```

2.  **Configurar entorno:**
    Ejecuta el script de configuración automática:
    ```bash
    setup_env.bat
    ```
    O manualmente:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configurar credenciales (Opcional):**
    Si usas funciones que requieren token de Hugging Face, crea un archivo `.env` en esta carpeta con el siguiente contenido:
    ```
    HF_TOKEN=tu_token_aqui
    ```

## Uso

### Opción 1: Interfaz Web (Recomendado)
Ejecuta el archivo:
```bash
run_web_app.bat
```
El navegador se abrirá automáticamente en `http://localhost:5000`.

### Opción 2: Línea de Comandos
1.  Coloca tu archivo de audio (ej. `mi_audio.wav`) en esta carpeta.
2.  Ejecuta:
    ```bash
    run_translator.bat mi_audio.wav
    ```

## Notas
- La primera ejecución tardará unos minutos en descargar el modelo de Meta.
- Los audios generados y subidos se guardan en las carpetas `outputs/` y `uploads/` (ignoradas en git).
