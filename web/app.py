# -*- coding: utf-8 -*-
"""
多模态情感分析系统 - Streamlit Web界面（无emoji版本）
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
from config import Config, EMOTIONS_ZH, EMOTION_COLORS, EMOTIONS


# =============================================================================
# 页面配置
# =============================================================================

st.set_page_config(
    page_title="多模态情感分析系统",
    page_icon=":smile:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# 模型加载（缓存）
# =============================================================================

@st.cache_resource(hash_funcs={Predictor: lambda _: None})
def load_predictor() -> Predictor:
    """加载情感分析预测器（带缓存）

    注意：如果模型文件更新，需要重启Streamlit或清除缓存
    运行: streamlit cache clear
    """
    with st.spinner("正在加载模型..."):
        try:
            # 尝试加载训练好的模型（使用类别权重处理不平衡的模型）
            model_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'bert_hybrid_weighted_best_model.pth')

            if os.path.exists(model_path):
                predictor = Predictor(model_path=model_path)

                # 验证模型类型
                model_type = type(predictor.model).__name__
                if model_type != 'BERTHybridModel':
                    st.warning(f"警告: 加载的模型类型是 {model_type}，期望 BERTHybridModel")
                    st.warning("请运行: streamlit cache clear")

                st.success(f"已加载训练模型 (使用类别权重，预测分布更均衡)")

                # 显示模型配置信息
                st.info(f"""
                **模型配置:**
                - 模型类型: {model_type}
                - 数据源: {Config.data_source}
                - 文本特征: {Config.text_model} ({Config.text_dim}维)
                - 音频特征: {Config.audio_model} ({Config.audio_dim}维)
                - 视频特征: {Config.video_model} ({Config.video_dim}维)
                """)
            else:
                predictor = Predictor()
                st.warning("使用未训练模型（随机权重）")

                # 显示模型配置信息
                st.info(f"""
                **模型配置:**
                - 数据源: {Config.data_source}
                - 文本特征: {Config.text_model} ({Config.text_dim}维)
                - 音频特征: {Config.audio_model} ({Config.audio_dim}维)
                - 视频特征: {Config.video_model} ({Config.video_dim}维)
                """)
                st.warning("注意: 使用未训练模型的预测结果是随机的，仅供测试")

            return predictor

        except RuntimeError as e:
            if 'size mismatch' in str(e).lower():
                st.error("""
                **模型不兼容错误!**

                加载的模型与当前配置不匹配。可能的原因:
                - 模型是用BERT特征训练的(768维)，但当前配置使用GloVe(300维)
                - 模型是用COVAREP特征训练的(74维)，但当前配置使用wav2vec(768维)

                **解决方案:**
                1. 检查模型文件是否存在于 checkpoints/ 目录
                2. 确认config.py中的特征维度与模型匹配
                3. 重新训练模型: `python scripts/train_model.py`
                """)
            else:
                st.error(f"模型加载失败: {str(e)}")
            st.stop()

        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            st.stop()


# =============================================================================
# 辅助函数
# =============================================================================

def save_uploaded_file(uploaded_file, suffix: str) -> Optional[str]:
    """保存上传的文件到临时目录"""
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            dir=Config.temp_dir
        ) as f:
            f.write(uploaded_file.getbuffer())
            return f.name
    except Exception as e:
        st.error(f"文件保存失败: {str(e)}")
        return None


def cleanup_temp_file(file_path: Optional[str]) -> None:
    """清理临时文件"""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass


# =============================================================================
# 侧边栏
# =============================================================================

def render_sidebar() -> None:
    """渲染侧边栏设置"""
    with st.sidebar:
        st.title("设置")

        st.subheader("关于系统")
        st.info("""
        **多模态情感分析系统**

        本系统支持文本、音频、视频三种模态的情感识别。
        """)

        st.subheader("当前配置")
        st.markdown(f"""
        **数据源:** {Config.data_source}

        **特征提取器:**
        - 文本: {Config.text_model} ({Config.text_dim}维)
        - 音频: {Config.audio_model} ({Config.audio_dim}维)
        - 视频: {Config.video_model} ({Config.video_dim}维)

        **模型参数:**
        - 情感类别: {Config.num_classes}
        - 设备: {Config.device}
        """)

        st.subheader("使用说明")
        st.markdown("""
        1. 至少提供一种输入（文本/音频/视频）
        2. 点击"开始分析"按钮进行预测
        3. 查看预测结果和情感分布

        **注意:**
        - 文本输入使用GloVe词向量
        - 音频/视频需要使用官方数据集特征提取器
        - 临时上传的文件可能无法正确提取特征
        """)


# =============================================================================
# 主应用
# =============================================================================

def main():
    """主应用函数"""
    # 页面标题
    st.title("多模态情感分析系统")
    st.markdown("---")

    # 渲染侧边栏
    render_sidebar()

    # 加载模型
    predictor = load_predictor()

    # 输入区域
    st.header("输入区域")

    # 添加配置说明
    st.info(f"""
    当前使用 **{Config.data_source}** 数据源配置的模型。

    **支持的功能:**
    - 文本输入: 使用GloVe词向量 (300维) - 可正常使用
    - 音频/视频上传: 需要COVAREP/OpenFace特征 - 可能无法正常工作

    **建议:** 主要使用文本输入功能进行测试。
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("文本输入")
        text_input = st.text_area(
            "请输入要分析情感的文本",
            placeholder="例如：这部电影非常精彩！",
            height=150,
            help="输入任意英文或中文文本"
        )

        st.caption("提示: 文本使用GloVe词向量，支持中英文")

        st.subheader("音频上传")
        audio_file = st.file_uploader(
            "上传音频文件",
            type=['wav', 'mp3', 'm4a'],
            help="支持WAV, MP3, M4A格式",
            disabled=False
        )
        if audio_file:
            st.warning("注意: 上传的音频文件需要COVAREP特征提取器，当前可能不支持")

    with col2:
        st.subheader("视频上传")
        video_file = st.file_uploader(
            "上传视频文件",
            type=['mp4', 'avi', 'mov'],
            help="支持MP4, AVI, MOV格式",
            disabled=False
        )
        if video_file:
            st.warning("注意: 上传的视频文件需要OpenFace特征提取器，当前可能不支持")

        # 预测按钮
        predict_btn = st.button("开始分析", type="primary")

    # 结果展示区域
    st.header("分析结果")

    # 处理预测请求
    if predict_btn:
        # 验证输入
        if not text_input and not audio_file and not video_file:
            st.warning("请至少提供一种输入（文本/音频/视频）")
            return

        # 显示加载状态
        with st.spinner("正在分析中..."):
            # 初始化文件路径
            audio_path = None
            video_path = None

            try:
                # 保存音频文件
                if audio_file:
                    audio_suffix = os.path.splitext(audio_file.name)[1]
                    audio_path = save_uploaded_file(audio_file, audio_suffix)
                    if not audio_path:
                        st.error("音频文件保存失败")
                        return

                # 保存视频文件
                if video_file:
                    video_suffix = os.path.splitext(video_file.name)[1]
                    video_path = save_uploaded_file(video_file, video_suffix)
                    if not video_path:
                        st.error("视频文件保存失败")
                        return

                # 进行预测
                result = predictor.predict(
                    text=text_input if text_input else None,
                    audio_path=audio_path,
                    video_path=video_path
                )

                # 显示结果
                st.success("分析完成！")
                st.markdown("---")

                # 主要结果
                emotion_zh = EMOTIONS_ZH.get(result['emotion'], result['emotion'])
                st.metric(
                    label="预测情感",
                    value=emotion_zh,
                    delta=f"置信度: {result['confidence']:.2%}"
                )

                # 概率分布
                st.subheader("情感分布")

                # 按概率排序
                sorted_probs = sorted(
                    result['probabilities'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                emotions_zh = [EMOTIONS_ZH.get(k, k) for k, _ in sorted_probs]
                probs = [v for _, v in sorted_probs]

                # 创建DataFrame
                prob_data = []
                for emo_zh, prob in zip(emotions_zh, probs):
                    prob_data.append({
                        '情感': emo_zh,
                        '概率': f"{prob:.4f}",
                        '百分比': f"{prob:.2%}"
                    })

                df_probs = pd.DataFrame(prob_data)
                st.dataframe(
                    df_probs,
                    use_container_width=True,
                    hide_index=True
                )

                # 可用模态
                st.caption(f"使用的模态: {', '.join(result['available_modalities'])}")

            except ValueError as e:
                st.error(f"输入验证失败: {str(e)}")

            except RuntimeError as e:
                st.error(f"特征提取失败: {str(e)}")
                st.info("提示: 请确保上传的文件格式正确且内容有效")

            except Exception as e:
                st.error(f"分析过程中发生错误: {str(e)}")
                st.exception(e)

            finally:
                # 清理临时文件
                cleanup_temp_file(audio_path)
                cleanup_temp_file(video_path)


# =============================================================================
# 应用入口
# =============================================================================

if __name__ == '__main__':
    main()
