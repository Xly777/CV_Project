import os
from PIL import Image
import numpy as np
from collections import defaultdict
import shutil

def is_black_and_white(image):
    image_np = np.array(image)
    return np.all(np.logical_or(image_np == 0, image_np == 255))

def calculate_black_ratio(image):
    image_np = np.array(image)
    black_pixels = np.sum(image_np == 0)
    total_pixels = image_np.size
    return (black_pixels / total_pixels) * 100

def categorize_and_select_images(image_files, num_per_category=2):
    categories = defaultdict(list)
    
    for image_file in image_files:
        with Image.open(image_file) as img:
            img = img.convert("L")
            if is_black_and_white(img):
                black_ratio = calculate_black_ratio(img)
                category = int(black_ratio // 10) * 10
                if 10 <= category < 50:
                    categories[f"{category}_{category+10}"].append(image_file)
    
    selected_images = {}
    for category in categories:
        selected_images[category] = categories[category][:num_per_category]
    
    return selected_images

def process_and_save_images(input_directory, output_directory, num_per_category=2):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    image_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'))]
    
    selected_images = categorize_and_select_images(image_files, num_per_category)
    
    for category, images in selected_images.items():
        category_dir = os.path.join(output_directory, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        
        for image_file in images:
            shutil.copy(image_file, category_dir)

input_directory = "datasets/qd_imd/test"
output_directory = "datasets/mask"

process_and_save_images(input_directory, output_directory, num_per_category=2)

print(f"Images saved to {output_directory}")
