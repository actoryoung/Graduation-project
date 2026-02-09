#!/bin/bash
# 多模态情感分析系统 - Streamlit Web应用启动脚本
#
# 使用方法:
#   chmod +x run.sh
#   ./run.sh

set -e

echo "============================================================"
echo "多模态情感分析系统 - Streamlit Web应用"
echo "============================================================"
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python3，请先安装Python 3.10或更高版本"
    exit 1
fi

echo "[信息] Python版本:"
python3 --version
echo ""

# 检查streamlit是否安装
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "[错误] 未安装Streamlit，正在安装..."
    echo ""
    python3 -m pip install streamlit matplotlib pandas
    if [ $? -ne 0 ]; then
        echo "[错误] Streamlit安装失败"
        exit 1
    fi
    echo "[成功] Streamlit安装完成"
    echo ""
fi

echo "[信息] 启动Web应用..."
echo ""
echo "============================================================"
echo "应用将在浏览器中自动打开"
echo "如果浏览器未自动打开，请访问: http://localhost:8501"
echo "按 Ctrl+C 停止服务器"
echo "============================================================"
echo ""

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 启动Streamlit应用
streamlit run web/app.py
