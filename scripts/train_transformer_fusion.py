# -*- coding: utf-8 -*-
"""
Transformer融合模型训练脚本

使用BERT(768维) + COVAREP(74维) + OpenFace(710维) + Transformer融合进行3类情感分类

预期性能:
    - Accuracy: 68-75%
    - Macro F1: 0.62-0.70
    - Negative F1: 0.45-0.55

使用方法:
    python scripts/train_transformer_fusion.py
    python scripts/train_transformer_fusion.py --epochs 50
    python scripts/train_transformer_fusion.py --use-class-weights --weight-method sqrt_inv
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config, DEVICE


# =============================================================================
# 数据集类
# =============================================================================

class BERT3ClassDataset(Dataset):
    """BERT 3类多模态数据集"""

    def __init__(self, data_path: str):
        """
        初始化数据集

        Args:
            data_path: NPZ文件路径，包含text_features, audio_features, video_features, labels
        """
        data = np.load(data_path, allow_pickle=True)

        self.text_features = data['text_features'].astype(np.float32)  # [N, 768] BERT
        self.audio_features = data['audio_features'].astype(np.float32)  # [N, 74] COVAREP
        self.video_features = data['video_features'].astype(np.float32)  # [N, 710] OpenFace
        self.labels = data['labels'].astype(np.int64)  # [N] 3类标签

        print(f"加载数据: {data_path}")
        print(f"  样本数: {len(self.labels)}")
        print(f"  BERT特征: {self.text_features.shape}")
        print(f"  COVAREP特征: {self.audio_features.shape}")
        print(f"  OpenFace特征: {self.video_features.shape}")

        # 显示标签分布
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"  标签分布:")
        class_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        for cls, count in zip(unique, counts):
            print(f"    {class_names[cls]} ({cls}): {count} ({count/len(self.labels)*100:.1f}%)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': torch.from_numpy(self.text_features[idx]),
            'audio': torch.from_numpy(self.audio_features[idx]),
            'video': torch.from_numpy(self.video_features[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# =============================================================================
# 早停类
# =============================================================================

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=5, min_delta=1e-4, mode='max', verbose=True):
        """
        Args:
            patience: 耐心值
            min_delta: 最小改善阈值
            mode: 'max'表示指标越大越好
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch, score):
        """
        检查是否应该早停

        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  [早停] 指标改善: {score:.4f} (epoch {epoch})")
        else:
            self.counter += 1
            if self.verbose and self.counter >= self.patience - 1:
                print(f"  [早停] 指标未改善 {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"  [早停] 触发早停! 最佳epoch: {self.best_epoch}, 最佳分数: {self.best_score:.4f}")
            return True

        return False


# =============================================================================
# 类别权重计算
# =============================================================================

def compute_class_weights(labels, num_classes=3, method='sqrt_inv'):
    """
    计算类别权重以处理不平衡问题

    Args:
        labels: 标签数组
        num_classes: 类别数
        method: 'sqrt_inv' 使用sqrt(逆频率), 'inv' 使用直接逆频率

    Returns:
        类别权重tensor
    """
    class_counts = Counter(labels)
    total_samples = len(labels)

    print(f"\n类别权重计算 (方法: {method}):")
    class_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    weights = []
    for cls in range(num_classes):
        count = class_counts.get(cls, 1)  # 避免除以0
        if method == 'sqrt_inv':
            # 使用sqrt的逆频率，权重差异更温和
            weight = np.sqrt(total_samples / (num_classes * count))
        elif method == 'inv':
            # 直接逆频率
            weight = total_samples / (num_classes * count)
        else:
            raise ValueError(f"Unknown method: {method}")
        weights.append(weight)
        print(f"  {class_names[cls]} ({cls}): {count} 样本 ({count/total_samples*100:.1f}%), 权重 = {weight:.4f}")

    # 显示权重比例
    print(f"\n权重比例: Negative={weights[0]/weights[1]:.2f}x Neutral, Positive={weights[2]/weights[1]:.2f}x Neutral")

    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# 指标计算函数
# =============================================================================

def compute_metrics(predictions, labels, num_classes=3):
    """
    计算评估指标

    Args:
        predictions: 预测标签 [N]
        labels: 真实标签 [N]
        num_classes: 类别数

    Returns:
        包含accuracy, macro_f1, per_class_f1的字典
    """
    accuracy = (predictions == labels).float().mean().item()

    # 计算各类别F1
    f1_scores = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(),
                       labels=[0, 1, 2], average=None, zero_division=0)

    # 计算Macro F1
    macro_f1 = np.mean(f1_scores)

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'negative_f1': f1_scores[0] if len(f1_scores) > 0 else 0.0,
        'neutral_f1': f1_scores[1] if len(f1_scores) > 1 else 0.0,
        'positive_f1': f1_scores[2] if len(f1_scores) > 2 else 0.0,
    }


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    pbar = tqdm(dataloader, desc="  训练")
    for batch in pbar:
        text = batch['text'].to(device)
        audio = batch['audio'].to(device)
        video = batch['video'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        optimizer.zero_grad()
        logits = model(text, audio, video)
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录
        total_loss += loss.item()
        _, predictions = torch.max(logits, dim=1)

        all_predictions.append(predictions)
        all_labels.append(labels)

        pbar.set_postfix(loss=loss.item())

    # 计算指标
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)

    metrics['loss'] = total_loss / len(dataloader)
    return metrics


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="  验证")
        for batch in pbar:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['label'].to(device)

            logits = model(text, audio, video)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predictions = torch.max(logits, dim=1)

            all_predictions.append(predictions)
            all_labels.append(labels)

            pbar.set_postfix(loss=loss.item())

    # 计算指标
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)

    metrics['loss'] = total_loss / len(dataloader)
    return metrics


# =============================================================================
# 主训练函数
# =============================================================================

def train_transformer_fusion(args):
    """训练Transformer融合模型"""

    print("=" * 70)
    print("Transformer融合多模态情感分析模型训练")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.lr}")
    print(f"  使用类别权重: {args.use_class_weights}")
    print(f"  权重方法: {args.weight_method}")
    print(f"  设备: {DEVICE}")

    # 创建数据加载器
    print(f"\n加载数据...")
    train_dataset = BERT3ClassDataset(os.path.join(args.data_dir, 'train.npz'))
    val_dataset = BERT3ClassDataset(os.path.join(args.data_dir, 'val.npz'))
    test_dataset = BERT3ClassDataset(os.path.join(args.data_dir, 'test.npz'))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 计算类别权重
    if args.use_class_weights:
        class_weights = compute_class_weights(train_dataset.labels, num_classes=3, method=args.weight_method)
        class_weights = class_weights.to(DEVICE)
    else:
        class_weights = None

    # 创建模型
    print(f"\n创建模型...")
    from src.models.transformer_fusion import TransformerFusionModel

    model = TransformerFusionModel(
        text_dim=768,
        audio_dim=74,
        video_dim=710,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=3,
        dropout_rate=args.dropout
    ).to(DEVICE)

    info = model.get_model_info()
    print(f"  参数量: {info['num_parameters']:,}")
    print(f"  模型大小: {info['model_size_mb']:.2f} MB")

    # 损失函数和优化器
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\n使用加权损失函数:")
        print(f"  权重: Negative={class_weights[0]:.4f}, "
              f"Neutral={class_weights[1]:.4f}, Positive={class_weights[2]:.4f}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # 早停
    early_stopping = EarlyStopping(patience=args.patience, mode='max', verbose=True)

    # 创建保存目录
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    print(f"\n开始训练...")
    print(f"{'='*70}")

    best_val_macro_f1 = 0.0
    train_history = {'loss': [], 'accuracy': [], 'macro_f1': []}
    val_history = {'loss': [], 'accuracy': [], 'macro_f1': []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"{'-'*70}")

        # 训练
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"  训练 - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"Macro F1: {train_metrics['macro_f1']:.4f}")

        # 验证
        val_metrics = validate(model, val_loader, criterion, DEVICE)
        print(f"  验证 - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"Macro F1: {val_metrics['macro_f1']:.4f}")

        # 记录历史
        for key in train_history:
            train_history[key].append(train_metrics[key])
            val_history[key].append(val_metrics[key])

        # 学习率调度
        scheduler.step()

        # 保存最佳模型
        if val_metrics['macro_f1'] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics['macro_f1']
            checkpoint_path = checkpoint_dir / 'transformer_fusion_best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': best_val_macro_f1,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"  [保存] 最佳模型: Macro F1 = {best_val_macro_f1:.4f}")

        # 早停检查
        if early_stopping(epoch, val_metrics['macro_f1']):
            print(f"\n训练提前停止于epoch {epoch}")
            break

    # 测试
    print(f"\n{'='*70}")
    print(f"在测试集上评估最佳模型...")
    print(f"{'='*70}")

    # 加载最佳模型
    checkpoint_path = checkpoint_dir / 'transformer_fusion_best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 在测试集上评估
    test_metrics = validate(model, test_loader, criterion, DEVICE)

    print(f"\n测试集结果:")
    print(f"  准确率: {test_metrics['accuracy']:.4f}")
    print(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Negative F1: {test_metrics['negative_f1']:.4f}")
    print(f"  Neutral F1: {test_metrics['neutral_f1']:.4f}")
    print(f"  Positive F1: {test_metrics['positive_f1']:.4f}")

    # 保存结果
    results = {
        'test_metrics': test_metrics,
        'best_val_macro_f1': best_val_macro_f1,
        'best_epoch': checkpoint['epoch'],
        'train_history': train_history,
        'val_history': val_history,
    }

    results_path = checkpoint_dir / 'transformer_fusion_results.json'
    import json
    with open(results_path, 'w') as f:
        # 转换numpy类型为Python类型
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            return obj

        json.dump(results, f, default=convert, indent=2)

    print(f"\n结果已保存: {results_path}")
    print(f"模型已保存: {checkpoint_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='训练Transformer融合多模态情感分析模型')

    # 数据参数
    parser.add_argument('--data-dir', type=str, default='data/mosei/bert_3class_full',
                       help='数据目录')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/transformer_fusion',
                       help='检查点保存目录')

    # 模型参数
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='隐藏层维度')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Transformer层数')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout率')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--patience', type=int, default=8,
                       help='早停耐心值')
    parser.add_argument('--use-class-weights', action='store_true',
                       help='使用类别权重处理不平衡问题')
    parser.add_argument('--weight-method', type=str, default='sqrt_inv',
                       choices=['sqrt_inv', 'inv'],
                       help='权重计算方法: sqrt_inv(温和) 或 inv(激进)')

    args = parser.parse_args()

    # 开始训练
    train_transformer_fusion(args)


if __name__ == '__main__':
    main()
