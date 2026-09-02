# Traductor K'iche' - Español (Instrucciones para macOS)

## ✅ Instalación Completada

El proyecto ya está listo para usar con todas las dependencias instaladas:
- Python 3.9.6
- PyTorch 2.8.0
- Transformers 4.57.6
- Flask 3.1.3
- Y todas las demás dependencias

## Cómo Usar

### Opción 1: Interfaz Web (Recomendado)
Ejecuta en la terminal:
```bash
./run_web_app.sh
```
El servidor se iniciará en `http://localhost:5000`

### Opción 2: Línea de Comandos
Para traducir un archivo de audio específico:
```bash
./run_translator.sh mi_archivo.wav
```

## Activar el Entorno Virtual Manualmente

Si necesitas trabajar directamente con Python:
```bash
source venv/bin/activate
python app.py
```

Para desactivar el entorno:
```bash
deactivate
```

## Notas Importantes

- **Primera ejecución:** El modelo MMS (~1GB) se descargará automáticamente la primera vez. Necesitas conexión a internet.
- **Formato de audio:** Los archivos deben ser `.wav` con voz en K'iche'.
- **Token de Hugging Face:** Opcional, solo si usas funciones que lo requieran.

## Solución de Problemas

Si encuentras errores al ejecutar:
1. Asegúrate de estar en la carpeta del proyecto
2. Verifica que el entorno virtual esté activado
3. Revisa que los scripts tengan permisos de ejecución:
   ```bash
   chmod +x run_web_app.sh run_translator.sh
   ```
