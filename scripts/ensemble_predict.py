# -*- coding: utf-8 -*-
"""
S7: 集成学习脚本
支持多种集成策略: checkpoint集成、多模型集成、投票/加权
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from scripts.train_baseline import MOSEIBaselineDataset, BaselineFusionModel, AttentionFusionModel


def load_model(model_path, model_type='attention', num_classes=3, device='cuda'):
    """加载单个模型"""
    if model_type == 'baseline':
        model = BaselineFusionModel(
            text_dim=300,
            audio_dim=74,
            video_dim=710,
            num_classes=num_classes
        ).to(device)
    elif model_type == 'attention':
        model = AttentionFusionModel(
            text_dim=300,
            audio_dim=74,
            video_dim=710,
            fusion_dim=256,
            num_classes=num_classes
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_with_model(model, dataloader, device):
    """使用模型预测，返回概率和预测类别"""
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)

            outputs = model(text, audio, video)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    return all_probs, all_preds


def soft_voting(probs_list):
    """软投票: 平均概率"""
    avg_probs = np.mean(probs_list, axis=0)
    return np.argmax(avg_probs, axis=1)


def hard_voting(preds_list):
    """硬投票: 多数投票"""
    preds_array = np.array(preds_list)  # (n_models, n_samples)
    # 使用mode找每个样本的多数类
    from scipy.stats import mode
    majority_votes, _ = mode(preds_array, axis=0)
    return majority_votes.flatten()


def weighted_voting(probs_list, weights):
    """加权软投票"""
    weighted_probs = np.average(probs_list, axis=0, weights=weights)
    return np.argmax(weighted_probs, axis=1)


class CheckpointEnsemble:
    """Checkpoint集成: 同一模型不同epoch的集成"""

    def __init__(self, checkpoint_dir, model_type='attention', num_classes=3, device='cuda'):
        self.checkpoint_dir = checkpoint_dir
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = device
        self.models = []

    def load_checkpoints(self, checkpoint_paths):
        """加载多个checkpoint"""
        print(f"Loading {len(checkpoint_paths)} checkpoints...")
        for path in checkpoint_paths:
            if os.path.exists(path):
                model = load_model(path, self.model_type, self.num_classes, self.device)
                self.models.append(model)
                print(f"  Loaded: {path}")
            else:
                print(f"  Warning: {path} not found")

        if len(self.models) == 0:
            raise ValueError("No checkpoints loaded!")

    def predict(self, dataloader, method='soft'):
        """集成预测"""
        probs_list = []
        preds_list = []

        for i, model in enumerate(self.models):
            print(f"Predicting with model {i+1}/{len(self.models)}...")
            probs, preds = predict_with_model(model, dataloader, self.device)
            probs_list.append(probs)
            preds_list.append(preds)

        print(f"\nUsing {method} voting...")
        if method == 'soft':
            ensemble_preds = soft_voting(probs_list)
        elif method == 'hard':
            ensemble_preds = hard_voting(preds_list)
        else:
            raise ValueError(f"Unknown voting method: {method}")

        return ensemble_preds, probs_list, preds_list


class MultiModelEnsemble:
    """多模型集成: 不同架构模型的集成"""

    def __init__(self, model_configs, num_classes=3, device='cuda'):
        """
        model_configs: list of dict
            [{'path': 'model1.pth', 'type': 'attention', 'weight': 1.0}, ...]
        """
        self.model_configs = model_configs
        self.num_classes = num_classes
        self.device = device
        self.models = []

    def load_models(self):
        """加载多个模型"""
        print(f"Loading {len(self.model_configs)} models...")
        for config in self.model_configs:
            path = config['path']
            model_type = config.get('type', 'attention')

            if os.path.exists(path):
                model = load_model(path, model_type, self.num_classes, self.device)
                self.models.append((model, config.get('weight', 1.0)))
                print(f"  Loaded: {path} (weight={config.get('weight', 1.0)})")
            else:
                print(f"  Warning: {path} not found")

        if len(self.models) == 0:
            raise ValueError("No models loaded!")

    def predict(self, dataloader, method='weighted'):
        """集成预测"""
        probs_list = []
        preds_list = []
        weights = []

        for i, (model, weight) in enumerate(self.models):
            print(f"Predicting with model {i+1}/{len(self.models)}...")
            probs, preds = predict_with_model(model, dataloader, self.device)
            probs_list.append(probs)
            preds_list.append(preds)
            weights.append(weight)

        print(f"\nUsing {method} voting...")
        if method == 'soft':
            # 等权重软投票
            ensemble_preds = soft_voting(probs_list)
        elif method == 'hard':
            # 硬投票
            ensemble_preds = hard_voting(preds_list)
        elif method == 'weighted':
            # 加权软投票
            weights = np.array(weights)
            weights = weights / weights.sum()  # 归一化
            ensemble_preds = weighted_voting(probs_list, weights)
        else:
            raise ValueError(f"Unknown voting method: {method}")

        return ensemble_preds, probs_list, preds_list


def evaluate_ensemble(ensemble_preds, labels, class_names=None):
    """评估集成结果"""
    if class_names is None:
        class_names = ['Negative', 'Neutral', 'Positive']

    accuracy = accuracy_score(labels, ensemble_preds)

    print("\n" + "=" * 70)
    print("集成模型评估结果")
    print("=" * 70)
    print(f"\n总体准确率: {accuracy*100:.2f}%")

    print("\n" + "-" * 70)
    print("分类报告")
    print("-" * 70)
    report = classification_report(labels, ensemble_preds, target_names=class_names, digits=4)
    print(report)

    print("\n" + "-" * 70)
    print("混淆矩阵")
    print("-" * 70)
    cm = confusion_matrix(labels, ensemble_preds)
    print(cm)
    print("\n混淆矩阵说明 (行=真实, 列=预测):")
    print("                Negative  Neutral  Positive")
    for i, name in enumerate(class_names):
        print(f"{name:>10}:  {cm[i, 0]:>8}  {cm[i, 1]:>8}  {cm[i, 2]:>8}")

    # 各类别准确率
    print("\n" + "-" * 70)
    print("各类别准确率")
    print("-" * 70)
    for i, name in enumerate(class_names):
        if cm[i, :].sum() > 0:
            acc = cm[i, i] / cm[i, :].sum()
            print(f"{name}: {acc*100:.2f}% ({cm[i, i]}/{cm[i, :].sum()})")

    return accuracy, cm


def main():
    parser = argparse.ArgumentParser(description='S7: 集成学习')
    parser.add_argument('--mode', choices=['checkpoint', 'multi_model', 'single'], default='multi_model',
                        help='Ensemble mode')
    parser.add_argument('--models', nargs='+', default=[
        'checkpoints/baseline_attention_3class_weighted_best_model.pth',
        'checkpoints/baseline_attention_3class_ce_best_model.pth',
        'checkpoints/baseline_3class_ce_best_model.pth'
    ], help='Model paths for ensemble')
    parser.add_argument('--voting', choices=['soft', 'hard', 'weighted'], default='soft',
                        help='Voting method')
    parser.add_argument('--data-dir', default='data/mosei/processed_cleaned_3class',
                        help='Test data directory')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='Model weights for weighted voting')

    args = parser.parse_args()

    print("=" * 70)
    print("S7: 集成学习实验")
    print("=" * 70)
    print(f"\n模式: {args.mode}")
    print(f"投票方法: {args.voting}")

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载测试数据
    print("\n加载测试数据...")
    test_dataset = MOSEIBaselineDataset(
        os.path.join(args.data_dir, 'test.npz'),
        split='test'
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 获取真实标签
    labels = []
    for batch in test_loader:
        labels.extend(batch['label'].numpy())
    labels = np.array(labels)

    print(f"测试样本数: {len(labels)}")

    # 构建模型配置
    if args.weights is None:
        # 默认等权重
        weights = [1.0] * len(args.models)
    else:
        weights = args.weights

    model_configs = []
    for path, weight in zip(args.models, weights):
        model_type = 'attention' if 'attention' in path else 'baseline'
        model_configs.append({
            'path': path,
            'type': model_type,
            'weight': weight
        })

    print("\n模型配置:")
    for config in model_configs:
        print(f"  - {config['path']}")
        print(f"    Type: {config['type']}, Weight: {config['weight']}")

    # 创建集成
    if args.mode == 'multi_model':
        ensemble = MultiModelEnsemble(model_configs, num_classes=3, device=device)
        ensemble.load_models()
        ensemble_preds, probs_list, preds_list = ensemble.predict(test_loader, method=args.voting)

    elif args.mode == 'checkpoint':
        # Checkpoint集成 (使用同一模型的不同epoch)
        ensemble = CheckpointEnsemble(
            'checkpoints/',
            model_type='attention',
            num_classes=3,
            device=device
        )
        ensemble.load_checkpoints(args.models)
        ensemble_preds, probs_list, preds_list = ensemble.predict(test_loader, method=args.voting)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # 评估
    accuracy, cm = evaluate_ensemble(ensemble_preds, labels)

    # 保存结果
    results = {
        'mode': args.mode,
        'voting': args.voting,
        'models': args.models,
        'weights': weights,
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist()
    }

    output_file = 'ensemble_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"结果已保存到: {output_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
