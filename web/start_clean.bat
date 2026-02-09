@echo off
echo Killing all Python processes...
taskkill /F /IM python.exe >nul 2>&1
timeout /t 2 >nul

echo Clearing Streamlit cache...
cd /d "%~dp0.."
streamlit cache clear

echo Starting on port 18002...
set HF_ENDPOINT=https://hf-mirror.com
streamlit run web/app.py --server.port 18002 --server.headless true

pause
