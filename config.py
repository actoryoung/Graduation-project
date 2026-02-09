"""
多模态情感分析系统 - 全局配置

本模块定义系统中使用的所有配置参数，包括：
- 路径配置
- 模型配置
- 特征维度
- 分类配置
- 设备配置
"""

import os
import torch

# =============================================================================
# 路径配置
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'checkpoints')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# 确保目录存在
for dir_path in [DATA_DIR, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# =============================================================================
# 模型配置
# =============================================================================

# 特征来源
DATA_SOURCE = 'default'  # 使用实时特征提取（BERT + wav2vec + OpenFace）

# 预训练模型名称（用于实时推理）
TEXT_MODEL = 'bert-base-uncased'   # 使用BERT上下文特征
AUDIO_MODEL = 'covarep'            # CMU-MOSEI使用COVAREP声学特征
VIDEO_MODEL = 'openface'           # CMU-MOSEI使用OpenFace视觉特征

# 特征维度（CMU-MOSEI官方数据集）
TEXT_DIM = 768       # BERT特征维度（bert-base-uncased）
AUDIO_DIM = 74      # COVAREP声学特征维度
VIDEO_DIM = 710     # OpenFace2特征维度（移除3个极端异常维度后）
# 注意：如果需要降维到25维，需要在模型中添加降维层

# 融合配置
FUSION_DIM = 256    # 融合层隐藏维度

# =============================================================================
# 分类配置
# =============================================================================

NUM_CLASSES = 7

# 7类情感分类
EMOTIONS_7 = [
    'strong_negative',  # 强烈负面
    'negative',         # 负面
    'weak_negative',    # 弱负面
    'neutral',          # 中性
    'weak_positive',    # 弱正面
    'positive',         # 正面
    'strong_positive'   # 强烈正面
]

# 2分类（简化版）
EMOTIONS_2 = ['negative', 'positive']

# 中文标签映射
EMOTIONS_ZH = {
    'strong_negative': '强烈负面',
    'negative': '负面',
    'weak_negative': '弱负面',
    'neutral': '中性',
    'weak_positive': '弱正面',
    'positive': '正面',
    'strong_positive': '强烈正面'
}

# 情感颜色（用于可视化）
EMOTION_COLORS = {
    'strong_negative': '#8B0000',  # 深红
    'negative': '#DC143C',         # 红色
    'weak_negative': '#FFA07A',    # 浅红
    'neutral': '#808080',          # 灰色
    'weak_positive': '#90EE90',    # 浅绿
    'positive': '#32CD32',         # 绿色
    'strong_positive': '#006400'   # 深绿
}

# 默认使用的情感集合
EMOTIONS = EMOTIONS_7

# =============================================================================
# 数据预处理配置
# =============================================================================

# 文本
MAX_TEXT_LENGTH = 128  # BERT最大序列长度

# 音频
TARGET_SAMPLE_RATE = 16000  # wav2vec要求的采样率
MAX_AUDIO_DURATION = 10     # 最大音频时长（秒）

# 视频
NUM_VIDEO_FRAMES = 16       # 提取的帧数
IMAGE_SIZE = 224            # 图像尺寸

# =============================================================================
# 训练配置
# =============================================================================

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-5

EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP_NORM = 1.0

# 学习率调度
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5

# =============================================================================
# 设备配置
# =============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 如果使用CUDA，设置相关配置
if DEVICE == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 打印设备信息
print(f"使用设备: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# =============================================================================
# Web配置
# =============================================================================

WEB_TITLE = "多模态情感分析系统"
WEB_PORT = 18001  # 统一端口
WEB_HOST = "localhost"

# 禁止随意修改端口 - 统一使用18001端口
WEB_PORT_FIXED = True  # 固定端口标志

# =============================================================================
# 日志配置
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# 其他配置
# =============================================================================

# 随机种子（用于可复现性）
RANDOM_SEED = 42

# 文件上传限制
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200MB

# 临时文件目录
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)


# =============================================================================
# 配置类（方便导入使用）
# =============================================================================

class Config:
    """配置类，提供便捷的属性访问"""

    # 路径
    project_root = PROJECT_ROOT
    data_dir = DATA_DIR
    model_dir = MODEL_DIR
    checkpoint_dir = CHECKPOINT_DIR
    log_dir = LOG_DIR
    temp_dir = TEMP_DIR

    # 数据源
    data_source = DATA_SOURCE

    # 模型
    text_model = TEXT_MODEL
    audio_model = AUDIO_MODEL
    video_model = VIDEO_MODEL

    # 维度
    text_dim = TEXT_DIM
    audio_dim = AUDIO_DIM
    video_dim = VIDEO_DIM
    fusion_dim = FUSION_DIM

    # 分类
    num_classes = NUM_CLASSES
    emotions = EMOTIONS
    emotions_zh = EMOTIONS_ZH
    emotion_colors = EMOTION_COLORS

    # 预处理
    max_text_length = MAX_TEXT_LENGTH
    target_sample_rate = TARGET_SAMPLE_RATE
    max_audio_duration = MAX_AUDIO_DURATION
    num_video_frames = NUM_VIDEO_FRAMES
    image_size = IMAGE_SIZE

    # 训练
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    num_epochs = NUM_EPOCHS

    # 设备
    device = DEVICE

    @classmethod
    def get_emotion_label(cls, emotion: str, lang: str = 'zh') -> str:
        """获取情感标签

        Args:
            emotion: 情感名称（英文）
            lang: 语言 ('zh' 或 'en')

        Returns:
            对应语言的标签
        """
        if lang == 'zh':
            return cls.emotions_zh.get(emotion, emotion)
        return emotion

    @classmethod
    def get_emotion_color(cls, emotion: str) -> str:
        """获取情感对应的颜色

        Args:
            emotion: 情感名称

        Returns:
            颜色代码（十六进制）
        """
        return cls.emotion_colors.get(emotion, '#808080')


# 设置随机种子
def set_seed(seed: int = None):
    """设置随机种子

    Args:
        seed: 随机种子值
    """
    seed = seed or RANDOM_SEED

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# 模块导入时自动设置随机种子
if RANDOM_SEED is not None:
    set_seed(RANDOM_SEED)


if __name__ == '__main__':
    # 测试配置
    print("=" * 50)
    print("配置测试")
    print("=" * 50)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型目录: {MODEL_DIR}")
    print(f"设备: {DEVICE}")
    print(f"情感类别: {EMOTIONS}")
    print(f"特征维度: 文本={TEXT_DIM}, 音频={AUDIO_DIM}, 视频={VIDEO_DIM}")
    print("=" * 50)
