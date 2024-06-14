import argparse
import os
import random
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入目录')
    parser.add_argument('--output', required=True, help='输出目录名')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path('datasets') / args.output

    output_dir.mkdir(parents=True, exist_ok=True)

    ext = {'.JPG', '.JPEG', '.PNG', '.TIF', '.TIFF'}

    images = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].upper() in ext:
                images.append(Path(root) / file)

    if len(images) < 1000:
        raise ValueError("输入目录中图像数量不足1000张")

    selected_images = random.sample(images, 1000)

    for img in selected_images:
        shutil.copy(img, output_dir / img.name)

    print(f"成功抽取1000张图片到 {output_dir}")

if __name__ == '__main__':
    main()
