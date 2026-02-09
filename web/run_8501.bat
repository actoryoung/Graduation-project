@echo off
cd /d "%~dp0.."
set HF_ENDPOINT=https://hf-mirror.com
echo Starting on port 8501...
streamlit run web/app.py --server.port 8501
pause
