@echo off
REM Diagnostic startup script

echo ============================================================
echo Streamlit Debug Startup
echo ============================================================
echo.

cd /d "%~dp0.."

echo [1/3] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python not found
    pause
    exit /b 1
)

echo [2/3] Checking Streamlit...
python -c "import streamlit; print(streamlit.__version__)"
if %errorlevel% neq 0 (
    echo [ERROR] Streamlit not found, installing...
    pip install streamlit
)

echo [3/3] Checking app.py...
if not exist "web\app.py" (
    echo [ERROR] web\app.py not found!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Starting Streamlit...
echo URL: http://localhost:18001
echo ============================================================
echo.

set HF_ENDPOINT=https://hf-mirror.com

REM Try different port if 18001 is busy
streamlit run web/app.py --server.port 18001 --server.headless true

pause
