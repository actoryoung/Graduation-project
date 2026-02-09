# -*- coding: utf-8 -*-
"""
将7类数据转换为3类数据
7类: [-3,-2,-1,0,1,2,3]
3类: [0=negative, 1=neutral, 2=positive]
"""

import numpy as np
import os


def simplify_labels(labels):
    """7类 -> 3类"""
    simplified = []
    for label in labels:
        if label <= 2:  # strong_neg, neg, weak_neg -> 0
            simplified.append(0)
        elif label == 3:  # neutral -> 1
            simplified.append(1)
        else:  # weak_pos, pos, strong_pos -> 2
            simplified.append(2)
    return np.array(simplified)


def main():
    data_dir = 'data/mosei/processed_cleaned_correct'
    output_dir = 'data/mosei/processed_cleaned_3class'

    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'val', 'test']:
        print(f'处理 {split} 数据...')
        data = np.load(os.path.join(data_dir, f'{split}.npz'), allow_pickle=True)

        # 简化标签
        labels_3class = simplify_labels(data['labels'])

        # 统计分布
        unique, counts = np.unique(labels_3class, return_counts=True)
        total = len(labels_3class)
        print(f'  标签分布:')
        for cls, count in zip(unique, counts):
            print(f'    类别{cls}: {count} ({count/total*100:.1f}%)')

        # 保存
        np.savez(
            os.path.join(output_dir, f'{split}.npz'),
            text_features=data['text_features'],
            audio_features=data['audio_features'],
            video_features=data['video_features'],
            labels=labels_3class
        )
        print(f'  已保存到: {output_dir}/{split}.npz')

    print('\n3类数据创建完成!')
    print(f'输出目录: {output_dir}')


if __name__ == '__main__':
    main()
