@echo off
chcp 65001 >nul
REM ============================================================
REM 多模态情感分析系统 - Streamlit Web应用启动脚本
REM ============================================================

echo.
echo ============================================================
echo 多模态情感分析系统 - Streamlit Web应用
echo ============================================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到Python，请先安装Python 3.10或更高版本
    pause
    exit /b 1
)

echo [信息] Python版本:
python --version
echo.

REM 检查streamlit是否安装
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未安装Streamlit，正在安装...
    echo.
    python -m pip install streamlit matplotlib pandas
    if %errorlevel% neq 0 (
        echo [错误] Streamlit安装失败
        pause
        exit /b 1
    )
    echo [成功] Streamlit安装完成
    echo.
)

echo [信息] 启动Web应用...
echo.
echo ============================================================
echo 应用将在浏览器中自动打开
echo 如果浏览器未自动打开，请访问: http://localhost:18001
echo 按 Ctrl+C 停止服务器
echo ============================================================
echo.

REM 切换到项目根目录
cd /d "%~dp0.."

REM 启动Streamlit应用
streamlit run web/app.py --server.port 18001

pause
