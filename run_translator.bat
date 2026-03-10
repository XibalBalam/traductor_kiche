@echo off
setlocal
echo ==========================================
echo   Traductor K'iche' - Espanol (PoC)
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no se encuentra. Por favor instala Python 3.8+ y anadelo al PATH.
    echo Descarga: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if venv exists
if not exist venv (
    echo [INFO] Creando entorno virtual...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Fallo al crear venv.
        pause
        exit /b 1
    )
    echo [INFO] Instalando dependencias...
    venv\Scripts\pip install -r requirements.txt
)

REM Run the script
if "%~1"=="" (
    echo Uso: run_translator.bat "archivo_audio.wav"
    echo.
    echo Por favor arrastra un archivo de audio sobre este script o ejecutalo desde consola.
) else (
    echo Procesando archivo: %~1
    venv\Scripts\python translator_poc.py "%~1"
)

pause
