@echo off
REM ============================================================
REM Multimodal Sentiment Analysis - Streamlit Web App
REM Port: 18001
REM ============================================================

echo Starting Multimodal Sentiment Analysis System...
echo.

cd /d "%~dp0"

REM Set Hugging Face mirror
set HF_ENDPOINT=https://hf-mirror.com

REM Start Streamlit on port 18001
echo URL: http://localhost:18001
echo.
streamlit run web/app.py

pause
