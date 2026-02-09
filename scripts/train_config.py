# -*- coding: utf-8 -*-
"""
训练配置模块

本模块定义多模态情感分析模型训练的所有超参数和配置。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    """训练配置类

    包含所有训练相关的超参数和配置选项。
    """

    # =============================================================================
    # 基本训练参数
    # =============================================================================

    batch_size: int = 16
    """批次大小"""

    learning_rate: float = 1e-4
    """学习率"""

    num_epochs: int = 50
    """训练轮数"""

    weight_decay: float = 1e-5
    """权重衰减（L2正则化）"""

    gradient_clip_norm: float = 1.0
    """梯度裁剪范数"""

    # =============================================================================
    # 优化器配置
    # =============================================================================

    optimizer: str = 'adam'
    """优化器类型: 'adam', 'adamw', 'sgd'"""

    adam_beta1: float = 0.9
    """Adam优化器 beta1 参数"""

    adam_beta2: float = 0.999
    """Adam优化器 beta2 参数"""

    # =============================================================================
    # 学习率调度器配置
    # =============================================================================

    scheduler: str = 'reduce_on_plateau'
    """学习率调度器类型: 'reduce_on_plateau', 'step', 'cosine'"""

    scheduler_patience: int = 3
    """ReduceLROnPlateau调度器的耐心值"""

    scheduler_factor: float = 0.5
    """ReduceLROnPlateau调度器的衰减因子"""

    scheduler_min_lr: float = 1e-7
    """最小学习率"""

    # =============================================================================
    # 早停配置
    # =============================================================================

    early_stopping: bool = True
    """是否启用早停"""

    early_stopping_patience: int = 5
    """早停耐心值（验证损失不再改善的轮数）"""

    early_stopping_min_delta: float = 1e-4
    """早停最小改善阈值"""

    # =============================================================================
    # 模型冻结配置
    # =============================================================================

    freeze_text_encoder: bool = True
    """是否冻结BERT文本编码器"""

    freeze_audio_encoder: bool = True
    """是否冻结wav2vec音频编码器"""

    freeze_video_encoder: bool = True
    """是否冻结视频编码器"""

    # =============================================================================
    # 数据集配置
    # =============================================================================

    data_dir: str = 'data/mosei'
    """CMU-MOSEI数据集根目录"""

    train_split: float = 0.8
    """训练集比例"""

    val_split: float = 0.1
    """验证集比例"""

    test_split: float = 0.1
    """测试集比例"""

    num_workers: int = 4
    """数据加载器工作进程数"""

    pin_memory: bool = True
    """是否使用锁页内存（加速GPU数据传输）"""

    # =============================================================================
    # 特征配置
    # =============================================================================

    text_dim: int = 768
    """文本特征维度（BERT输出）"""

    audio_dim: int = 768
    """音频特征维度（wav2vec输出）"""

    video_dim: int = 25
    """视频特征维度（OpenFace AU特征）"""

    fusion_dim: int = 256
    """融合层隐藏维度"""

    num_classes: int = 7
    """情感类别数"""

    dropout_rate: float = 0.3
    """Dropout比率"""

    # =============================================================================
    # 检查点配置
    # =============================================================================

    checkpoint_dir: str = 'checkpoints'
    """检查点保存目录"""

    save_best_only: bool = True
    """是否只保存最佳模型"""

    save_frequency: int = 5
    """保存频率（每N轮保存一次）"""

    resume_from_checkpoint: Optional[str] = None
    """从检查点恢复训练的路径"""

    # =============================================================================
    # 日志配置
    # =============================================================================

    log_dir: str = 'logs'
    """日志保存目录"""

    tensorboard_log_dir: str = 'logs/tensorboard'
    """TensorBoard日志目录"""

    log_interval: int = 10
    """日志记录间隔（每N个batch记录一次）"""

    # =============================================================================
    # 验证配置
    # =============================================================================

    val_interval: int = 1
    """验证间隔（每N轮验证一次）"""

    # =============================================================================
    # 随机种子
    # =============================================================================

    seed: int = 42
    """随机种子（确保可复现性）"""

    # =============================================================================
    # 设备配置
    # =============================================================================

    device: str = 'cuda'
    """训练设备: 'cuda' 或 'cpu'"""

    # =============================================================================
    # 混合精度训练
    # =============================================================================

    use_amp: bool = False
    """是否使用自动混合精度训练（需要GPU）"""

    # =============================================================================
    # 方法
    # =============================================================================

    def __post_init__(self):
        """初始化后处理"""
        import os
        import sys

        # 添加项目根目录到路径
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # 创建必要的目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)

    def get_optimizer_params(self):
        """获取优化器参数字典"""
        return {
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
            'betas': (self.adam_beta1, self.adam_beta2)
        }

    def get_scheduler_params(self):
        """获取学习率调度器参数字典"""
        return {
            'mode': 'min',
            'factor': self.scheduler_factor,
            'patience': self.scheduler_patience,
            'min_lr': self.scheduler_min_lr,
            'verbose': True
        }

    def get_splits_info(self):
        """获取数据集划分信息"""
        return {
            'train': self.train_split,
            'val': self.val_split,
            'test': self.test_split
        }

    def validate(self):
        """验证配置的有效性"""
        # 验证数据集划分比例
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"数据集划分比例之和应为1.0，当前为: {total}"
            )

        # 验证学习率
        if self.learning_rate <= 0:
            raise ValueError(f"学习率必须为正数，当前为: {self.learning_rate}")

        # 验证批次大小
        if self.batch_size <= 0:
            raise ValueError(f"批次大小必须为正数，当前为: {self.batch_size}")

        # 验证训练轮数
        if self.num_epochs <= 0:
            raise ValueError(f"训练轮数必须为正数，当前为: {self.num_epochs}")

        return True

    def __str__(self):
        """配置信息字符串表示"""
        lines = [
            "=" * 70,
            "训练配置",
            "=" * 70,
            "",
            "基本训练参数:",
            f"  批次大小: {self.batch_size}",
            f"  学习率: {self.learning_rate}",
            f"  训练轮数: {self.num_epochs}",
            f"  权重衰减: {self.weight_decay}",
            f"  梯度裁剪: {self.gradient_clip_norm}",
            "",
            "优化器:",
            f"  类型: {self.optimizer}",
            "",
            "学习率调度:",
            f"  调度器: {self.scheduler}",
            f"  耐心值: {self.scheduler_patience}",
            f"  衰减因子: {self.scheduler_factor}",
            "",
            "早停:",
            f"  启用: {self.early_stopping}",
            f"  耐心值: {self.early_stopping_patience}",
            "",
            "模型冻结:",
            f"  冻结BERT: {self.freeze_text_encoder}",
            f"  冻结wav2vec: {self.freeze_audio_encoder}",
            f"  冻结视频编码器: {self.freeze_video_encoder}",
            "",
            "数据集:",
            f"  数据目录: {self.data_dir}",
            f"  训练/验证/测试比例: {self.train_split}/{self.val_split}/{self.test_split}",
            f"  数据加载进程数: {self.num_workers}",
            "",
            "特征维度:",
            f"  文本: {self.text_dim}, 音频: {self.audio_dim}, 视频: {self.video_dim}",
            f"  融合维度: {self.fusion_dim}",
            f"  类别数: {self.num_classes}",
            "",
            "检查点:",
            f"  保存目录: {self.checkpoint_dir}",
            f"  只保存最佳: {self.save_best_only}",
            "",
            "日志:",
            f"  日志目录: {self.log_dir}",
            f"  TensorBoard: {self.tensorboard_log_dir}",
            "",
            "其他:",
            f"  随机种子: {self.seed}",
            f"  设备: {self.device}",
            f"  混合精度: {self.use_amp}",
            "",
            "=" * 70
        ]
        return "\n".join(lines)


# =============================================================================
# 预定义配置
# =============================================================================

def get_default_config() -> TrainConfig:
    """获取默认训练配置"""
    return TrainConfig()


def get_fast_config() -> TrainConfig:
    """获取快速训练配置（用于调试）"""
    return TrainConfig(
        batch_size=8,
        num_epochs=5,
        early_stopping_patience=2,
        val_interval=1,
        log_interval=5
    )


def get_small_config() -> TrainConfig:
    """获取小数据集训练配置"""
    return TrainConfig(
        batch_size=8,
        num_epochs=30,
        early_stopping_patience=5
    )


# =============================================================================
# 测试
# =============================================================================

if __name__ == '__main__':
    # 设置Windows控制台编码
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print(get_default_config())
    print("\n配置验证:")
    config = get_default_config()
    config.validate()
    print("✓ 配置有效")
