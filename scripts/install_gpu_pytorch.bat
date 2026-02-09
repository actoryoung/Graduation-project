@echo off
REM GPU PyTorch Installation Script for RTX 4060
REM CUDA 12.8 Driver detected - Installing CUDA 12.1 PyTorch

echo ============================================================
echo Installing GPU PyTorch for NVIDIA RTX 4060
echo ============================================================
echo.

REM Step 1: Uninstall CPU version
echo [1/3] Uninstalling CPU version of PyTorch...
pip uninstall -y torch torchvision torchaudio

REM Step 2: Install GPU version
echo [2/3] Installing GPU version (CUDA 12.1)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Step 3: Verify installation
echo [3/3] Verifying GPU installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Test with mock data: python scripts/train_model.py --fast
echo 2. Download real dataset: python scripts/download_mosei.py
echo 3. Full training: python scripts/train_model.py
echo.
pause
