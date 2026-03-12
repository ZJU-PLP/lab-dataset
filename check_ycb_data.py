#!/usr/bin/env python3
import os
import sys
import scipy.io as scio
from tqdm import tqdm
import argparse

# 尝试导入项目配置，确保路径与训练时一致
try:
    from common import Config
    from utils.basic_utils import Basic_Utils
except ImportError:
    print("错误: 请将此脚本放在 'ffb6d' 项目的根目录下 (与 train_ycb.py 同级).")
    sys.exit(1)


def check_dataset():
    parser = argparse.ArgumentParser(description="检查 YCB 数据集是否有损坏的 .mat 文件")
    parser.add_argument('--split', type=str, default='all', choices=['train', 'test', 'all'], help='要检查的数据集划分')
    args = parser.parse_args()

    # 初始化配置
    try:
        config = Config(ds_name='ycb')
        bs_utils = Basic_Utils(config)
    except Exception as e:
        print(f"初始化配置失败: {e}")
        return

    root = config.ycb_root
    print(f"--> 数据集根目录: {root}")

    # 获取要检查的文件列表
    file_lists = {}

    if args.split in ['train', 'all']:
        train_path = 'datasets/ycb/dataset_config/train_data_list.txt'
        print(f"--> 读取训练列表: {train_path}")
        file_lists['train'] = bs_utils.read_lines(train_path)

    if args.split in ['test', 'all']:
        test_path = 'datasets/ycb/dataset_config/test_data_list.txt'
        print(f"--> 读取测试列表: {test_path}")
        file_lists['test'] = bs_utils.read_lines(test_path)

    corrupted_files = []

    for split_name, item_list in file_lists.items():
        print(f"\n正在检查 {split_name} 集 ({len(item_list)} 个文件)...")

        # 使用 tqdm 显示进度条
        for item_name in tqdm(item_list):
            # 构造完整路径
            meta_path = os.path.join(root, item_name + '-meta.mat')

            # 1. 检查文件是否存在
            if not os.path.exists(meta_path):
                msg = f"[MISSING] {meta_path}"
                tqdm.write(msg)
                corrupted_files.append(msg)
                continue

            # 2. 尝试加载 .mat 文件 (这是你报错的核心位置)
            try:
                # 只读取头部或尝试完整读取以触发 I/O 错误
                scio.loadmat(meta_path)
            except Exception as e:
                # 捕获 OSError, MatReadError 等
                msg = f"[CORRUPT MAT] {meta_path} | Error: {str(e)}"
                tqdm.write(msg)
                corrupted_files.append(msg)
                continue

            # (可选) 3. 如果你想检查图片是否损坏，可以取消下面注释
            # 注意：这会显著增加检查时间
            # """
            try:
                from PIL import Image
                # Check Depth
                with Image.open(os.path.join(root, item_name + '-depth.png')) as img:
                    img.verify()
                # Check Color
                with Image.open(os.path.join(root, item_name + '-color.png')) as img:
                    img.verify()
                # Check Label
                with Image.open(os.path.join(root, item_name + '-label.png')) as img:
                    img.verify()
            except Exception as e:
                msg = f"[CORRUPT IMG] {item_name} | Error: {str(e)}"
                tqdm.write(msg)
                corrupted_files.append(msg)
            # """

    print("\n" + "=" * 50)
    print("检查完成。")
    if len(corrupted_files) > 0:
        print(f"发现 {len(corrupted_files)} 个损坏或丢失的文件:")
        save_path = "corrupted_files.txt"
        with open(save_path, "w") as f:
            for line in corrupted_files:
                print(line)
                f.write(line + "\n")
        print(f"\n损坏文件列表已保存至: {os.path.abspath(save_path)}")
        print("建议: 删除这些文件，或者重新从原始数据集中复制覆盖它们。")
    else:
        print("未发现损坏的 .mat 文件。")
    print("=" * 50)


if __name__ == "__main__":
    check_dataset()