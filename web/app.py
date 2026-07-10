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
import torch

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.inference import Predictor
from config import Config, EMOTIONS_ZH, EMOTION_COLORS, EMOTIONS
from ensemble_predictor import S7EnsemblePredictor

# VADER情感分析（用于纯文本输入辅助判断）
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
    vader_analyzer = SentimentIntensityAnalyzer()
except ImportError:
    VADER_AVAILABLE = False
    print("[WARNING] vaderSentiment未安装，纯文本辅助功能不可用")


def get_vader_sentiment(text: str) -> Dict[str, float]:
    """获取VADER情感分数

    Returns:
        {'neg': float, 'neu': float, 'pos': float, 'compound': float}
        compound: -1(最负面) 到 +1(最正面)
    """
    if not VADER_AVAILABLE:
        return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
    return vader_analyzer.polarity_scores(text)


def adjust_prediction_with_vader(result: Dict, text: str, has_audio: bool, has_video: bool) -> Dict:
    """
    使用VADER调整预测结果（主要用于纯文本输入）

    当只有文本输入时，模型的GloVe平均池化会丢失情感信息，
    此时使用VADER来辅助判断。
    """
    # 如果有音频或视频输入，模型应该能正常工作，不需要VADER辅助
    if has_audio or has_video:
        result['vader_adjusted'] = False
        return result

    # 只有纯文本输入时，使用VADER辅助
    vader = get_vader_sentiment(text)
    compound = vader['compound']

    # VADER阈值（可根据效果调整）
    NEGATIVE_THRESHOLD = -0.3   # compound < -0.3 认为是负面
    POSITIVE_THRESHOLD = 0.3    # compound > 0.3 认为是正面

    original_emotion = result['emotion']
    original_probs = result['probabilities']

    # 如果VADER检测到强烈情感，但模型预测为相反类别，进行调整
    if compound < NEGATIVE_THRESHOLD and original_emotion in ['Positive', 'Neutral']:
        # VADER认为是负面，但模型预测为正面或中性
        result['emotion'] = 'Negative'
        result['emotion_zh'] = '负面'
        result['confidence'] = 0.5  # 降低置信度，表示这是VADER辅助判断
        result['probabilities'] = {
            'Negative': 0.6,
            'Neutral': 0.25,
            'Positive': 0.15
        }
        result['vader_adjusted'] = True
        result['vader_compound'] = compound
        result['adjust_reason'] = f"VADER检测到负面情感(compound={compound:.2f})，模型原预测为{original_emotion}"

    elif compound > POSITIVE_THRESHOLD and original_emotion == 'Negative':
        # VADER认为是正面，但模型预测为负面
        result['emotion'] = 'Positive'
        result['emotion_zh'] = '正面'
        result['confidence'] = 0.5
        result['probabilities'] = {
            'Negative': 0.15,
            'Neutral': 0.25,
            'Positive': 0.6
        }
        result['vader_adjusted'] = True
        result['vader_compound'] = compound
        result['adjust_reason'] = f"VADER检测到正面情感(compound={compound:.2f})，模型原预测为{original_emotion}"

    else:
        result['vader_adjusted'] = False

    return result


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
# 侧边栏 - 模型选择
# =============================================================================

st.sidebar.title("模型配置")

MODEL_OPTIONS = {
    "最佳性能 - S7增强集成 (推荐)": {
        "type": "ensemble",
        "models": [
            "02_s4_series/s4_fair_split_lr1e-4_dropout0.35.pth",
            "01_s10_series/s10_fair_split_lr5e-6_dropout0.35.pth"
        ],
        "weights": [0.10, 0.90],
        "name": "S7 增强集成 (S4 0.1 + S10 0.9)",
        "accuracy": "70.00%",
        "macro_f1": "0.7124",
        "neg_f1": "88.54%",
        "description": "网格搜索优化权重，当前项目最优性能"
    },
    "基线对照 - 原始不平衡数据": {
        "type": "single",
        "path": "data_balance_ablation/s10_original_best.pth",
        "name": "S10 - 原始不平衡 (A组)",
        "accuracy": "43.84%",
        "macro_f1": "0.3945",
        "neg_f1": "13.98%",
        "description": "基线: Negative类别仅占10%，严重不平衡"
    },
    "本文改进 - 完全平衡数据": {
        "type": "single",
        "path": "data_balance_ablation/s10_full_best.pth",
        "name": "S10 - 完全平衡 (C组)",
        "accuracy": "65.48%",
        "macro_f1": "0.6626",
        "neg_f1": "82.35%",
        "description": "本文方法: 三类完全平衡，每类~33%，Negative识别能力大幅提升"
    }
}

model_choice = st.sidebar.radio(
    "选择预测模型",
    options=list(MODEL_OPTIONS.keys()),
    index=0,
    help="S10-完全扩充模型性能最优，在平衡数据集上训练，推荐使用"
)

model_info = MODEL_OPTIONS[model_choice]

# 显示模型信息
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{model_info['name']}**")
st.sidebar.markdown(f"**准确率**: {model_info['accuracy']}")
st.sidebar.markdown(f"**Macro F1**: {model_info['macro_f1']}")
st.sidebar.markdown(f"**Negative F1**: {model_info['neg_f1']}")
st.sidebar.caption(model_info['description'])
st.sidebar.markdown("---")


# =============================================================================
# 模型加载（缓存）
# =============================================================================

@st.cache_resource(hash_funcs={Predictor: lambda _: None, S7EnsemblePredictor: lambda _: None})
def load_predictor(model_choice: str):
    """加载情感分析预测器（带缓存）

    Args:
        model_choice: 模型选择 ("S10-CE (推荐)" 或 "S7-集成")

    注意：如果模型文件更新，需要重启Streamlit或清除缓存
    运行: streamlit cache clear
    """
    with st.spinner("正在加载模型..."):
        try:
            model_info = MODEL_OPTIONS[model_choice]
            
            if model_info["type"] == "single":
                # 加载单个S10模型
                model_path = os.path.join(PROJECT_ROOT, 'checkpoints', model_info["path"])
                
                if not os.path.exists(model_path):
                    st.error(f"模型文件不存在: {model_path}")
                    st.stop()
                
                predictor = Predictor(model_path=model_path)
                model_type = type(predictor.model).__name__
                
                st.success(f"已加载 {model_info['name']}")
                # 判断是否是完全扩充模型
                is_augmented = "完全扩充" in model_info['name']
                dataset_desc = "平衡数据集（训练集、验证集、测试集均已扩充负向样本）" if is_augmented else "原始SDK数据集（负向样本较少）"

                st.info(f"""
                **模型配置:**
                - 模型类型: {model_type}
                - 数据集: {dataset_desc}
                - 准确率: {model_info['accuracy']}
                - Macro F1: {model_info['macro_f1']}
                - Negative F1: {model_info['neg_f1']}
                - 文本特征: GloVe ({300}维)
                - 音频特征: COVAREP ({74}维)
                - 视频特征: OpenFace ({710}维)
                """)
                
                st.warning("""
                **当前状态:**

                - 纯文本输入：支持
                - 音频上传：支持（实时74维COVAREP兼容特征）
                - 视频上传：支持（ResNet50实时特征提取710维）
                """)
                
                return predictor
                
            elif model_info["type"] == "ensemble":
                # 加载S7集成模型
                model_paths = [
                    os.path.join(PROJECT_ROOT, 'checkpoints', path)
                    for path in model_info["models"]
                ]
                
                # 检查所有模型文件是否存在
                for path in model_paths:
                    if not os.path.exists(path):
                        st.error(f"模型文件不存在: {path}")
                        st.stop()
                
                predictor = S7EnsemblePredictor(
                    model_paths=model_paths,
                    weights=model_info["weights"],
                    num_classes=3,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                st.success(f"已加载 {model_info['name']}")
                st.info(f"""
                **集成模型配置:**
                - 模型1: baseline_attention + weighted_ce (权重 {model_info['weights'][0]})
                - 模型2: baseline_attention + ce (权重 {model_info['weights'][1]})
                - 准确率: {model_info['accuracy']}
                - Macro F1: {model_info['macro_f1']}
                - Negative F1: {model_info['neg_f1']}
                - 文本特征: GloVe ({300}维)
                - 音频特征: COVAREP ({74}维)
                - 视频特征: OpenFace ({710}维)
                """)

                st.warning("""
                **当前状态:**

                - 纯文本输入：支持
                - 音频上传：支持（实时74维COVAREP兼容特征）
                - 视频上传：支持（ResNet50实时特征提取710维）
                """)
                
                return predictor

        except RuntimeError as e:
            if 'size mismatch' in str(e).lower():
                st.error("""
                **模型不兼容错误!**

                加载的模型与当前配置不匹配。

                **解决方案:**
                1. 检查模型文件是否存在于 checkpoints/ 目录
                2. 确认config.py中的特征维度与模型匹配
                3. 清除缓存: streamlit cache clear
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

        **推荐模型：** S10-完全扩充模型在平衡数据集上训练，
        各类别F1分数更均衡，尤其是Negative类别（F1: 0.8337）。
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

    # 加载模型（根据用户选择）
    predictor = load_predictor(model_choice)

    # 输入区域
    st.header("输入区域")

    # 添加配置说明
    st.info(f"""
    当前使用 **{Config.data_source}** 数据源配置的模型。

    **支持的功能:**
    - 文本输入: 使用GloVe词向量 (300维) - 可正常使用
    - 音频上传: 使用实时COVAREP兼容特征 (74维)
    - 视频上传: ResNet50实时特征提取 (710维)

    **重要提示:**
    - 模型严重依赖多模态融合，单文本输入时负向情感识别能力有限（F1: 0.00）
    - 建议上传音视频以获得更准确的情感识别，特别是负向情感
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("文本输入")
        text_input = st.text_area(
            "请输入要分析情感的文本",
            placeholder="例如：这部电影非常精彩！",
            height=150,
            help="输入任意英文文本"
        )

        st.caption("提示: 文本使用GloVe词向量，支持英文")

        st.subheader("音频上传")
        # S10 和 S7 都支持音频（74维COVAREP）
        audio_disabled = False
        audio_help = "支持实时COVAREP兼容74维音频特征提取（S10和S7模型均可用）"
        
        audio_file = st.file_uploader(
            "上传音频文件",
            type=['wav', 'mp3', 'm4a'],
            help=audio_help,
            disabled=audio_disabled
        )
        st.caption("音频功能已启用（实时COVAREP兼容特征）")

    with col2:
        st.subheader("视频上传")
        video_file = st.file_uploader(
            "上传视频文件",
            type=['mp4', 'avi', 'mov'],
            help="使用ResNet50提取视频特征，输出710维兼容CMU-MOSEI模型",
            disabled=False
        )
        st.caption("视频功能已启用（ResNet50实时特征提取）")

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

                # 使用VADER辅助调整（针对纯文本输入）
                has_audio = audio_file is not None
                has_video = video_file is not None
                result = adjust_prediction_with_vader(result, text_input or "", has_audio, has_video)

                # 显示结果
                st.success("分析完成！")

                # 如果使用VADER调整，显示提示
                if result.get('vader_adjusted', False):
                    st.info(f"ℹ️ **VADER辅助**: {result.get('adjust_reason', '')}")

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
