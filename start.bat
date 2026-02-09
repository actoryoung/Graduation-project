@echo off
REM ============================================================
REM Multimodal Sentiment Analysis - Clean Startup
REM ============================================================

echo [1/4] Stopping all Python processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM streamlit.exe >nul 2>&1
timeout /t 2 >nul

echo [2/4] Changing to project directory...
cd /d "%~dp0"

echo [3/4] Clearing Streamlit cache...
streamlit cache clear

echo [4/4] Starting Streamlit on port 18002...
echo.
echo ============================================================
echo Application will start in your browser
echo URL: http://localhost:18002
echo Press Ctrl+C to stop
echo ============================================================
echo.

REM Set environment variables
set HF_ENDPOINT=https://hf-mirror.com

REM Disable Streamlit usage stats and email prompt
set STREAMLIT_SERVER_HEADLESS=true
set STREAMLIT_LOGGER_LEVEL=warning

REM Start Streamlit
python -m streamlit run web/app.py --server.port 18002 --server.headless true

pause
