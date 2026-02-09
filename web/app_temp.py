# -*- coding: utf-8 -*-
"""
多模态情感分析系统 - 临时Web界面（用于流程验证）

注意：当前使用简化版文本特征提取器，预测结果仅供参考
统一端口: 8501
"""

import sys
import os
import tempfile
from typing import Optional, Dict

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.inference import Predictor
from config import Config, EMOTIONS_ZH, EMOTION_COLORS, EMOTIONS, WEB_PORT


# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="多模态情感分析系统 (临时验证版)",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# 临时警告横幅
# =============================================================================

def show_temp_warning():
    """显示临时版本警告"""
    st.warning("""
    【临时验证版本】当前使用简化版文本特征提取器进行流程验证。

    - **不影响**: 代码流程、模型架构、界面功能
    - **影响**: 预测结果不可信（使用伪随机特征）

    完整模型训练完成后将获得准确的预测能力。
    """, icon="⚠️")


# =============================================================================
# 模型加载（缓存）
# =============================================================================

@st.cache_resource
def load_predictor() -> Predictor:
    """加载情感分析预测器（带缓存）"""
    with st.spinner("正在加载模型..."):
        try:
            # 尝试加载训练好的临时模型
            model_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'best_model_temp.pth')

            if os.path.exists(model_path):
                predictor = Predictor(model_path=model_path)
                st.success(f"已加载临时训练模型 (Val Acc: 53.55%)")

                # 显示模型配置信息
                st.info(f"""
                **模型配置:**
                - 数据源: CMU-MOSEI
                - 文本特征: GloVe (300维) - [简化版]
                - 音频特征: COVAREP (74维)
                - 视频特征: OpenFace (25维) - [零向量]
                - 训练准确率: 53.55% (Text + Audio双模态)
                """)
            else:
                predictor = Predictor()
                st.warning("未找到训练模型，使用随机权重")

                # 显示模型配置信息
                st.info(f"""
                **模型配置:**
                - 数据源: {Config.data_source}
                - 文本特征: {Config.text_model} ({Config.text_dim}维)
                - 音频特征: {Config.audio_model} ({Config.audio_dim}维)
                - 视频特征: {Config.video_model} ({Config.video_dim}维)
                """)
                st.warning("注意: 使用未训练模型的预测结果是随机的")

            return predictor

        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            st.info("""
            **错误详情:**

            模型架构可能与训练时不匹配。请检查：
            1. checkpoints/best_model_temp.pth 文件是否存在
            2. 模型架构是否一致

            **解决方案:**
            ```bash
            # 重新训练模型
            python scripts/train_model.py
            ```
            """)
            st.stop()


# =============================================================================
# 主界面
# =============================================================================

def main():
    """主应用入口"""

    # 显示标题
    st.title("多模态情感分析系统")
    st.markdown("---")

    # 显示临时警告
    show_temp_warning()

    # 加载模型
    predictor = load_predictor()

    # 创建两列布局
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("输入")

        # 文本输入
        text_input = st.text_area(
            "输入文本",
            placeholder="请输入要分析情感的文本...",
            height=100,
            help="输入任意英文文本进行情感分析"
        )

        # 分析按钮
        analyze_button = st.button("开始分析", type="primary", use_container_width=True)

    with col2:
        st.header("说明")

        st.markdown("""
        ### 使用说明

        1. 在左侧输入框中输入文本
        2. 点击"开始分析"按钮
        3. 查看右侧的情感分析结果

        ### 当前状态

        - **文本特征**: 简化版 (伪随机)
        - **音频功能**: 暂不支持
        - **视频功能**: 暂不支持

        ### 注意事项

        当前版本仅用于验证技术流程，
        预测结果仅供参考，不反映真实模型性能。
        """)

    # 处理分析请求
    if analyze_button:
        if not text_input.strip():
            st.error("请输入文本！")
        else:
            with st.spinner("正在分析..."):
                try:
                    # 预测
                    result = predictor.predict(text=text_input)

                    # 显示结果
                    st.markdown("---")
                    st.header("分析结果")

                    # 创建结果展示列
                    result_col1, result_col2 = st.columns([1, 1])

                    with result_col1:
                        # 情感标签
                        emotion_zh = EMOTIONS_ZH.get(result['emotion'], result['emotion'])
                        st.markdown(f"### 预测情感: {emotion_zh}")

                        # 置信度
                        confidence = result['confidence']
                        st.metric("置信度", f"{confidence:.2%}")

                        # 使用的模态
                        modalities = result.get('available_modalities', [])
                        st.caption(f"使用模态: {', '.join(modalities) if modalities else '无'}")

                        # 临时提示
                        st.info("""
                        **提示**: 当前使用简化特征，结果仅供参考。
                        完整训练后将提供准确的情感预测。
                        """)

                    with result_col2:
                        # 概率分布图
                        st.subheader("概率分布")

                        probs = result['probabilities']

                        # 按中文标签顺序排列
                        labels = [EMOTIONS_ZH.get(e, e) for e in EMOTIONS]
                        values = [probs.get(e, 0.0) for e in EMOTIONS]
                        colors = [EMOTION_COLORS.get(e, '#808080') for e in EMOTIONS]

                        # 创建DataFrame
                        df = pd.DataFrame({
                            '情感': labels,
                            '概率': values
                        })
                        df['颜色'] = colors

                        # 绘制条形图
                        fig, ax = plt.subplots(figsize=(6, 4))
                        bars = ax.barh(df['情感'], df['概率'], color=df['颜色'])
                        ax.set_xlabel('概率')
                        ax.set_xlim(0, 1)
                        ax.set_ylim(len(df) - 0.5, -0.5)  # 反转y轴

                        # 添加数值标签
                        for i, (bar, prob) in enumerate(zip(bars, df['概率'])):
                            ax.text(prob + 0.02, i, f'{prob:.1%}',
                                   va='center', fontsize=9)

                        plt.tight_layout()
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"分析失败: {str(e)}")
                    st.exception(e)

    # 页脚信息
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        <p><strong>临时验证版本</strong> | 多模态情感分析系统</p>
        <p style='font-size: 0.9em;'>
            使用简化特征进行流程验证 | 完整模型训练中
        </p>
        <p style='font-size: 0.8em; color: #999;'>
            端口: {WEB_PORT} | 统一端口配置
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
