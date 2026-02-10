#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成DOTA评估所需的testset.txt文件
扫描CODrone/test/images目录下的所有图像文件，生成图像名列表
"""

import os
import glob
from pathlib import Path

def generate_testset_txt():
    """
    生成testset.txt文件
    """
    # 定义路径
    script_dir = Path(__file__).parent
    images_dir = script_dir / "CODrone" / "test" / "images"
    output_file = script_dir / "testset.txt"
    
    print(f"扫描图像目录: {images_dir}")
    print(f"输出文件: {output_file}")
    
    # 检查目录是否存在
    if not images_dir.exists():
        print(f"错误: 图像目录不存在 - {images_dir}")
        return False
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        pattern = str(images_dir / f"*{ext}")
        image_files.extend(glob.glob(pattern))
        # 也检查大写扩展名
        pattern = str(images_dir / f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))
    
    # 去重并排序
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"警告: 在 {images_dir} 中未找到任何图像文件")
        return False
    
    # 提取文件名（不包含扩展名）
    image_names = []
    for img_path in image_files:
        img_name = Path(img_path).stem  # 获取不含扩展名的文件名
        image_names.append(img_name)
    
    # 写入testset.txt文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for img_name in image_names:
                f.write(f"{img_name}\n")
        
        print(f"✅ 成功生成testset.txt!")
        print(f"   - 图像数量: {len(image_names)}")
        print(f"   - 输出路径: {output_file}")
        print(f"   - 图像列表:")
        for i, name in enumerate(image_names, 1):
            print(f"     {i:2d}. {name}")
        
        return True
        
    except Exception as e:
        print(f"错误: 写入文件失败 - {e}")
        return False

def verify_annotations():
    """
    验证对应的标注文件是否存在
    """
    script_dir = Path(__file__).parent
    testset_file = script_dir / "testset.txt"
    annot_dir = script_dir / "dataset" / "CODrone" / "test" / "annfile"
    
    if not testset_file.exists():
        print("❌ testset.txt 文件不存在，请先运行生成函数")
        return False
    
    if not annot_dir.exists():
        print(f"❌ 标注目录不存在: {annot_dir}")
        return False
    
    # 读取testset.txt
    with open(testset_file, 'r', encoding='utf-8') as f:
        image_names = [line.strip() for line in f if line.strip()]
    
    print(f"\n🔍 验证标注文件...")
    missing_annotations = []
    
    for img_name in image_names:
        annot_file = annot_dir / f"{img_name}.txt"
        if not annot_file.exists():
            missing_annotations.append(img_name)
        else:
            print(f"   ✅ {img_name}.txt")
    
    if missing_annotations:
        print(f"\n❌ 缺失的标注文件:")
        for name in missing_annotations:
            print(f"   - {name}.txt")
        return False
    else:
        print(f"\n✅ 所有标注文件都存在!")
        return True

def main():
    """
    主函数
    """
    print("=" * 60)
    print("CODrone testset.txt 生成工具")
    print("=" * 60)
    
    # 生成testset.txt
    success = generate_testset_txt()
    
    if success:
        # 验证标注文件
        verify_annotations()
        
        print("\n" + "=" * 60)
        print("📋 使用说明:")
        print("1. 生成的testset.txt可直接用于DOTA评估脚本")
        print("2. 确保检测结果文件按以下格式命名:")
        print("   - Task1_car.txt")
        print("   - Task1_truck.txt") 
        print("   - Task1_traffic-sign.txt")
        print("   - ... (其他类别)")
        print("3. 修改dota_evaluation_task1.py中的路径配置:")
        print("   - imagesetfile: 指向生成的testset.txt")
        print("   - annopath: 指向dataset/CODrone/test/annfile/{:s}.txt")
        print("   - detpath: 指向检测结果目录/Task1_{:s}.txt")
        print("=" * 60)
    
if __name__ == "__main__":
    main() 