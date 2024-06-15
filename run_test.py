import os
from loguru import logger
from PIL import Image
import numpy as np
from model import generate_image
from tqdm import tqdm

def process_images_and_masks(image_directory, mask_directory, output_directory, num_images=None):
    image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))]
    
    if num_images:
        image_files = image_files[:num_images]
    image_files = image_files[210:]
    logger.info(f"Number of images to process: {len(image_files)}")

    logger.info("starting...")
    for image_file in tqdm(image_files, desc="Processing images"):
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        image = Image.open(image_file)
        if image.mode == 'L':
                image = image.convert('RGB')
        image_np = np.array(image)
        
        for subdir in os.listdir(mask_directory):
            subdir_path = os.path.join(mask_directory, subdir)
            if os.path.isdir(subdir_path):
                mask_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))]

                for mask_file in mask_files:
                    mask_name = os.path.splitext(os.path.basename(mask_file))[0]
                    mask_image = Image.open(mask_file).convert("L")  # 转换为灰度图像
                    mask_np = np.array(mask_image)
                    
                    # 将mask图片转化成uint8，黑色部分为1，白色部分为0
                    mask_np = np.where(mask_np == 0, 1, 0).astype(np.uint8)
                    
                    # 调用generate_image函数
                    result_image = generate_image(image_np, mask_np)
                    
                    # 构建输出路径
                    percentage = subdir  # 使用子目录名作为百分比信息
                    output_path = os.path.join(output_directory, percentage)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    
                    # 保存结果图像
                    result_image_path = os.path.join(output_path, f"{image_name}.png")
                    result_image.save(result_image_path)
                    logger.info(f"Saved generated image to {result_image_path}")

# image_directory = "datasets/celebA"
# mask_directory = "datasets/mask"
# output_directory = "outputs/celebA"
# num_images = 2000

# process_images_and_masks(image_directory, mask_directory, output_directory, num_images)

image_directory = "datasets/places"
mask_directory = "datasets/mask"
output_directory = "outputs/places"
num_images = 2000

process_images_and_masks(image_directory, mask_directory, output_directory, num_images)