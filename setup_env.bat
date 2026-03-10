@echo off
echo ==========================================
echo   Configuracion Traductor K'iche'
echo ==========================================

REM Check/Create venv
if not exist venv (
    echo [INFO] Creando entorno virtual...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] No se pudo crear el entorno virtual.
        echo Asegurate de tener Python instalado (https://www.python.org/downloads/)
        pause
        exit /b 1
    )
)

REM Install requirements
echo [INFO] Instalando librerias necesarias...
venv\Scripts\pip install --upgrade pip
venv\Scripts\pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo [ERROR] Fallo al instalar librerias.
    pause
    exit /b 1
)

echo.
echo [EXITO] Todo listo.
echo Ahora puedes ejecutar: run_translator.bat audio.wav
pause
