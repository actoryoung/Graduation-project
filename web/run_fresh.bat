@echo off
REM Fresh start - Clear all caches and restart

echo ============================================================
echo Fresh Start - Clearing All Caches
echo ============================================================
echo.

echo [1/3] Stopping any running Streamlit processes...
taskkill /F /IM streamlit.exe >nul 2>&1
timeout /t 2 >nul

echo [2/3] Clearing Streamlit cache...
cd /d "%~dp0"
cd ..
streamlit cache clear

echo [3/3] Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc >nul 2>&1

echo.
echo ============================================================
echo All caches cleared!
echo ============================================================
echo.
echo Now starting fresh instance on port 18001...
echo.

set HF_ENDPOINT=https://hf-mirror.com
streamlit run web/app.py --server.port 18001

pause
