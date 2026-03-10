@echo off
setlocal
echo ==========================================
echo   Traductor K'iche' - Interfaz Web
echo ==========================================

REM Check if venv exists
if not exist venv (
    echo [INFO] Configurando entorno...
    call setup_env.bat
)

REM Install Flask if missing
venv\Scripts\pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Instalando Flask...
    venv\Scripts\pip install flask
)

echo.
echo [INFO] Iniciando servidor web...
echo Abre tu navegador en: http://127.0.0.1:5000
echo (Presiona Ctrl+C para detener)
echo.

venv\Scripts\python app.py

pause
