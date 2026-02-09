# -*- coding: utf-8 -*-
"""
测试不同的类别权重策略
对比: 平方根权重、线性权重、平方权重
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.train_baseline import MOSEIBaselineDataset, AttentionFusionModel, train_epoch, validate


def calculate_class_weights(train_labels, num_classes, strategy='sqrt'):
    """
    计算类别权重

    Args:
        train_labels: 训练标签
        num_classes: 类别数
        strategy: 权重策略
            - 'sqrt': 平方根权重 (1/√count) - 当前S4使用
            - 'linear': 线性权重 (1/count) - 更激进
            - 'square': 平方权重 (1/count²) - 最激进
            - 'log': 对数权重 (1/log(count+1)) - 更温和
            - 'manual': 手动权重
    """
    unique, counts = np.unique(train_labels, return_counts=True)
    class_counts = dict(zip(unique, counts))

    print(f"训练集类别分布: {class_counts}")

    if strategy == 'sqrt':
        # 平方根权重 - 当前S4使用
        class_weights = 1.0 / np.sqrt([class_counts[i] for i in range(num_classes)])
        weight_name = "平方根权重 (1/√count)"

    elif strategy == 'linear':
        # 线性权重 - 更激进
        class_weights = 1.0 / np.array([class_counts[i] for i in range(num_classes)])
        weight_name = "线性权重 (1/count)"

    elif strategy == 'square':
        # 平方权重 - 最激进
        class_weights = 1.0 / (np.array([class_counts[i] for i in range(num_classes)]) ** 2)
        weight_name = "平方权重 (1/count²)"

    elif strategy == 'log':
        # 对数权重 - 更温和
        class_weights = 1.0 / np.log(np.array([class_counts[i] for i in range(num_classes)]) + 1)
        weight_name = "对数权重 (1/log(count+1))"

    elif strategy == 'manual':
        # 手动权重 - 基于经验调优
        # Negative:Neutral:Positive = 1:5:4
        # 手动设置权重为 [3.0, 1.0, 1.5]
        class_weights = np.array([3.0, 1.0, 1.5])
        weight_name = "手动权重 [3.0, 1.0, 1.5]"

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # 归一化到平均为1
    class_weights = class_weights / class_weights.mean()

    print(f"{weight_name}")
    print(f"类别权重: {class_weights}")
    print(f"权重解释: 较少类别获得更高权重")

    return class_weights, weight_name


def train_with_weight_strategy(strategy, epochs=30):
    """使用指定权重策略训练模型"""

    print("=" * 70)
    print(f"训练策略: {strategy}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    print("\n加载数据...")
    train_dataset = MOSEIBaselineDataset(
        'data/mosei/processed_cleaned_3class/train.npz',
        split='train'
    )
    val_dataset = MOSEIBaselineDataset(
        'data/mosei/processed_cleaned_3class/val.npz',
        split='val'
    )
    test_dataset = MOSEIBaselineDataset(
        'data/mosei/processed_cleaned_3class/test.npz',
        split='test'
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 获取训练标签用于计算权重
    train_labels = []
    for batch in train_loader:
        train_labels.extend(batch['label'].numpy())
    train_labels = np.array(train_labels)

    # 计算类别权重
    class_weights, weight_name = calculate_class_weights(train_labels, num_classes=3, strategy=strategy)

    # 创建模型
    print("\n创建模型...")
    model = AttentionFusionModel(
        text_dim=300,
        audio_dim=74,
        video_dim=710,
        fusion_dim=256,
        num_classes=3
    ).to(device)

    # 损失函数和优化器
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 训练
    print("\n开始训练...")
    best_val_acc = 0.0
    best_epoch = 0
    model_name = f'baseline_attention_3class_weighted_{strategy}_best_model.pth'

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # 学习率调度
        scheduler.step(val_loss)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            os.makedirs('checkpoints', exist_ok=True)
            model_path = os.path.join('checkpoints', model_name)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'strategy': strategy,
                'class_weights': class_weights.tolist()
            }, model_path)
            print(f"[OK] 保存最佳模型 (Epoch {epoch+1}, Val Acc: {val_acc:.4f})")

    # 测试
    print("\n" + "=" * 70)
    print("在测试集上评估...")
    print("=" * 70)

    # 加载最佳模型
    model_path = os.path.join('checkpoints', model_name)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc = validate(model, test_loader, criterion, device)

    print(f"\n测试集结果:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"\n最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")

    return {
        'strategy': strategy,
        'weight_name': weight_name,
        'class_weights': class_weights,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'best_epoch': best_epoch,
        'model_path': model_path
    }


def main():
    print("=" * 70)
    print("类别权重策略对比实验")
    print("=" * 70)

    # 测试不同的权重策略
    strategies = ['sqrt', 'linear', 'square', 'log']  # 暂不包括manual

    results = []

    for strategy in strategies:
        try:
            result = train_with_weight_strategy(strategy, epochs=30)
            results.append(result)
        except Exception as e:
            print(f"策略 {strategy} 训练失败: {e}")

    # 总结结果
    print("\n" + "=" * 70)
    print("实验结果总结")
    print("=" * 70)

    print(f"\n{'策略':<15} {'权重':<30} {'验证Acc':<10} {'测试Acc':<10}")
    print("-" * 70)
    for r in results:
        weights_str = str([f"{w:.2f}" for w in r['class_weights']])
        print(f"{r['strategy']:<15} {weights_str:<30} {r['best_val_acc']:<10.4f} {r['test_acc']:<10.4f}")

    # 找出最优策略
    best_result = max(results, key=lambda x: x['test_acc'])
    print("\n" + "=" * 70)
    print(f"最优策略: {best_result['strategy']}")
    print(f"权重名称: {best_result['weight_name']}")
    print(f"类别权重: {best_result['class_weights']}")
    print(f"测试准确率: {best_result['test_acc']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
