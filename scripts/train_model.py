# -*- coding: utf-8 -*-
"""
多模态情感分析模型训练脚本

本脚本实现完整的训练流程，包括：
- 数据加载
- 模型初始化和冻结
- 训练循环
- 验证和测试
- 检查点保存
- TensorBoard日志
- 早停策略

使用方法:
    # 基本训练
    python scripts/train_model.py

    # 指定数据目录
    python scripts/train_model.py --data-dir data/mosei

    # 快速测试（少epoch）
    python scripts/train_model.py --fast

    # 从检查点恢复
    python scripts/train_model.py --resume checkpoints/checkpoint.pth
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config, DEVICE, EMOTIONS, TEXT_DIM, AUDIO_DIM, VIDEO_DIM, NUM_CLASSES
from scripts.train_config import TrainConfig, get_default_config
from src.models.fusion_module import MultimodalFusionModule
from src.data.cmu_mosei_dataset import create_data_loaders


# =============================================================================
# 早停类
# =============================================================================

class EarlyStopping:
    """早停机制"""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        """
        初始化早停

        Args:
            patience: 耐心值（等待改善的轮数）
            min_delta: 最小改善阈值
            mode: 'min'表示指标越小越好，'max'表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停

        Args:
            score: 当前指标值

        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# =============================================================================
# 指标计算
# =============================================================================

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 7
) -> Dict[str, float]:
    """
    计算评估指标

    Args:
        predictions: 预测类别索引，形状为(N,)
        targets: 目标类别索引，形状为(N,)
        num_classes: 类别数

    Returns:
        包含accuracy, precision, recall, f1的字典
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # 计算准确率
    accuracy = accuracy_score(targets, predictions)

    # 计算precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions,
        average='weighted',
        zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# =============================================================================
# 训练和验证函数
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_clip_norm: float = 1.0,
    use_amp: bool = False,
    log_interval: int = 10
) -> Dict[str, float]:
    """
    训练一个epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        gradient_clip_norm: 梯度裁剪范数
        use_amp: 是否使用混合精度
        log_interval: 日志间隔

    Returns:
        包含loss和accuracy的字典
    """
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, batch in enumerate(progress_bar):
        # 从字典中提取数据
        text_feats = batch['text'].to(device)
        audio_feats = batch['audio'].to(device)
        video_feats = batch['video'].to(device)
        labels = batch['label'].to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(text_feats, audio_feats, video_feats)
                loss = criterion(outputs, labels)
        else:
            outputs = model(text_feats, audio_feats, video_feats)
            loss = criterion(outputs, labels)

        # 反向传播
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        # 统计
        total_loss += loss.item()
        predictions = outputs.argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        all_predictions.extend(predictions)
        all_targets.extend(targets)

        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })

    # 计算指标
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    phase: str = 'Val'
) -> Dict[str, float]:
    """
    验证模型

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch
        phase: 阶段名称（Val或Test）

    Returns:
        包含loss和accuracy的字典
    """
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [{phase}]')

    with torch.no_grad():
        for batch in progress_bar:
            # 从字典中提取数据
            text_feats = batch['text'].to(device)
            audio_feats = batch['audio'].to(device)
            video_feats = batch['video'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            outputs = model(text_feats, audio_feats, video_feats)
            loss = criterion(outputs, labels)

            # 统计
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(targets)

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })

    # 计算指标
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))
    metrics['loss'] = total_loss / len(dataloader)

    return metrics


# =============================================================================
# 检查点管理
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: TrainConfig,
    filepath: str,
    is_best: bool = False
) -> None:
    """
    保存检查点

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        metrics: 评估指标
        config: 训练配置
        filepath: 保存路径
        is_best: 是否是最佳模型
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'text_dim': config.text_dim,
            'audio_dim': config.audio_dim,
            'video_dim': config.video_dim,
            'fusion_dim': config.fusion_dim,
            'num_classes': config.num_classes,
            'dropout_rate': config.dropout_rate
        }
    }

    torch.save(checkpoint, filepath)
    print(f"检查点已保存: {filepath}")

    if is_best:
        best_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"最佳模型已保存: {best_path}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None
) -> Tuple[int, Dict[str, float]]:
    """
    加载检查点

    Args:
        filepath: 检查点文件路径
        model: 模型
        optimizer: 优化器（可选）

    Returns:
        (epoch, metrics) 元组
    """
    checkpoint = torch.load(filepath, map_location=DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    print(f"检查点已加载: {filepath}")
    print(f"  - Epoch: {epoch}")
    print(f"  - Metrics: {metrics}")

    return epoch, metrics


# =============================================================================
# 主训练函数
# =============================================================================

def train(config: TrainConfig) -> None:
    """
    主训练函数

    Args:
        config: 训练配置
    """
    print("=" * 70)
    print("多模态情感分析模型训练")
    print("=" * 70)
    print(config)

    # 验证配置
    config.validate()

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 创建TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.tensorboard_log_dir, timestamp)
    writer = SummaryWriter(log_dir)

    # 创建数据加载器
    print("\n加载数据集...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_dir=config.data_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            max_samples=None  # 使用全部数据
        )
    except FileNotFoundError as e:
        print(f"[ERROR] 数据加载失败: {e}")
        print("\n请确保已完成以下步骤:")
        print("1. 运行 python scripts/download_mosei_data.py 下载并处理数据")
        print("2. 数据应保存在 data/mosei/processed/ 目录下")
        sys.exit(1)

    # 创建模型
    print("\n创建模型...")
    model = MultimodalFusionModule(
        text_dim=TEXT_DIM,      # 300 for GloVe
        audio_dim=AUDIO_DIM,    # 74 for COVAREP
        video_dim=VIDEO_DIM,    # 25 for OpenFace
        fusion_dim=config.fusion_dim,
        num_classes=NUM_CLASSES,
        dropout_rate=config.dropout_rate
    )
    model = model.to(DEVICE)
    print(f"模型参数数量: {model.get_num_parameters():,}")
    print(f"输入维度: 文本={TEXT_DIM}, 音频={AUDIO_DIM}, 视频={VIDEO_DIM}")

    # 冻结预训练编码器（如果配置要求）
    # 注意: 在这个实现中，编码器是独立的，融合层是可训练的
    # 如果未来集成了编码器，可以在这里冻结
    print("\n冻结配置:")
    print(f"  - 冻结文本编码器: {config.freeze_text_encoder}")
    print(f"  - 冻结音频编码器: {config.freeze_audio_encoder}")
    print(f"  - 冻结视频编码器: {config.freeze_video_encoder}")

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器（只优化融合层参数）
    # 过滤出需要梯度的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n可训练参数数量: {sum(p.numel() for p in trainable_params):,}")

    if config.optimizer == 'adam':
        optimizer = optim.Adam(trainable_params, **config.get_optimizer_params())
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW(trainable_params, **config.get_optimizer_params())
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(trainable_params, lr=config.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"不支持的优化器: {config.optimizer}")

    # 定义学习率调度器
    if config.scheduler == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **config.get_scheduler_params()
        )
    else:
        scheduler = None

    # 早停
    early_stopping = None
    if config.early_stopping:
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode='min'  # 监控验证损失
        )

    # 从检查点恢复（如果指定）
    start_epoch = 0
    best_val_loss = float('inf')
    if config.resume_from_checkpoint:
        if os.path.exists(config.resume_from_checkpoint):
            start_epoch, _ = load_checkpoint(
                config.resume_from_checkpoint,
                model,
                optimizer
            )
            start_epoch += 1
        else:
            print(f"[WARN] 检查点文件不存在: {config.resume_from_checkpoint}")

    # 训练循环
    print("\n开始训练...")
    print("=" * 70)

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 70)

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE,
            epoch, config.gradient_clip_norm, config.use_amp, config.log_interval
        )

        # 验证
        if (epoch + 1) % config.val_interval == 0:
            val_metrics = validate(model, val_loader, criterion, DEVICE, epoch, 'Val')

            # 打印指标
            print(f"\n训练 - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            print(f"验证 - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")

            # 记录到TensorBoard
            writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
            writer.add_scalar('Train/F1', train_metrics['f1'], epoch)
            writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
            writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            # 更新学习率
            if scheduler is not None:
                scheduler.step(val_metrics['loss'])

            # 保存检查点
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']

            if not config.save_best_only or is_best:
                if (epoch + 1) % config.save_frequency == 0 or is_best:
                    checkpoint_path = os.path.join(
                        config.checkpoint_dir,
                        f'checkpoint_epoch_{epoch + 1}.pth'
                    )
                    save_checkpoint(
                        model, optimizer, epoch, val_metrics, config,
                        checkpoint_path, is_best
                    )

            # 早停检查
            if early_stopping is not None:
                if early_stopping(val_metrics['loss']):
                    print(f"\n早停触发！在epoch {epoch + 1}停止训练")
                    break

    # 测试
    print("\n" + "=" * 70)
    print("训练完成，在测试集上评估...")
    print("=" * 70)

    # 加载最佳模型
    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model)
        print("已加载最佳模型进行测试")

    test_metrics = validate(model, test_loader, criterion, DEVICE, config.num_epochs, 'Test')

    print(f"\n测试集结果:")
    print(f"  - Loss: {test_metrics['loss']:.4f}")
    print(f"  - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  - Precision: {test_metrics['precision']:.4f}")
    print(f"  - Recall: {test_metrics['recall']:.4f}")
    print(f"  - F1: {test_metrics['f1']:.4f}")

    # 记录测试结果到TensorBoard
    writer.add_hparams(
        {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
        {
            'hparam/test_accuracy': test_metrics['accuracy'],
            'hparam/test_f1': test_metrics['f1']
        }
    )

    writer.close()

    # 保存最终结果
    results_path = os.path.join(config.log_dir, f'training_results_{timestamp}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
            'test_metrics': test_metrics,
            'timestamp': timestamp
        }, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {results_path}")
    print("\n训练完成!")


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    """命令行入口"""
    # 设置Windows控制台编码
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description='训练多模态情感分析模型'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/mosei',
        help='数据集目录'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='批次大小（覆盖配置）'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数（覆盖配置）'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='学习率（覆盖配置）'
    )

    parser.add_argument(
        '--fast',
        action='store_true',
        help='快速训练模式（用于调试）'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从检查点恢复训练'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='训练设备（cuda或cpu）'
    )

    args = parser.parse_args()

    # 获取配置
    if args.fast:
        config = get_fast_config()
    else:
        config = get_default_config()

    # 覆盖配置
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.resume:
        config.resume_from_checkpoint = args.resume
    if args.device:
        config.device = args.device

    # 开始训练
    try:
        train(config)
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
