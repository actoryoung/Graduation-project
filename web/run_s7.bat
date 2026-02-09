@echo off
REM 启动S7-v1集成版本的Web应用

echo ========================================
echo 多模态情感分析系统 (S7-v1集成版本)
echo ========================================
echo.
echo 模型配置:
echo   - 模型1: S3+S4 (权重1.5)
echo   - 模型2: S3 (权重1.0)
echo   - 测试准确率: 59.47%
echo   - Negative准确率: 23.38%
echo.
echo 正在启动Web界面...
echo.

cd web
streamlit run app_s7.py

pause
