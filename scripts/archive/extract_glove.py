# -*- coding: utf-8 -*-
"""
解压GloVe词向量文件

下载完成后，自动解压到指定目录。
"""

import os
import sys
import zipfile

# 项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows兼容
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def extract_glove():
    """解压GloVe词向量文件"""
    zip_path = os.path.join(project_root, 'data', 'glove', 'glove.6B.zip')
    extract_dir = os.path.join(project_root, 'data', 'glove')
    txt_path = os.path.join(extract_dir, 'glove.6B.300d.txt')

    print("=" * 50)
    print("解压GloVe词向量文件")
    print("=" * 50)

    # 检查zip文件
    if not os.path.exists(zip_path):
        print(f"[错误] ZIP文件不存在: {zip_path}")
        print(f"\n请先运行下载脚本:")
        print("  python scripts/download_glove.py")
        return False

    # 检查是否已解压
    if os.path.exists(txt_path):
        file_size = os.path.getsize(txt_path) / (1024 * 1024)
        print(f"[INFO] 词向量文件已存在: {txt_path}")
        print(f"[INFO] 文件大小: {file_size:.1f}MB")
        return True

    print(f"\n正在解压: {zip_path}")
    print(f"目标目录: {extract_dir}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取文件列表
            file_list = zip_ref.namelist()
            print(f"压缩包包含 {len(file_list)} 个文件")

            # 解压
            zip_ref.extractall(extract_dir)
            print("[成功] 解压完成!")

        # 验证
        if os.path.exists(txt_path):
            file_size = os.path.getsize(txt_path) / (1024 * 1024)
            print(f"[验证] 词向量文件: {txt_path}")
            print(f"[验证] 文件大小: {file_size:.1f}MB")

            # 快速检查格式
            with open(txt_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if 'the' in first_line.lower():
                    print("[验证] 文件格式正确 (看到'the'单词)")
                    return True

        print("\n" + "=" * 50)
        print("解压完成!")
        print("=" * 50)
        return True

    except Exception as e:
        print(f"[错误] 解压失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = extract_glove()
    sys.exit(0 if success else 1)
