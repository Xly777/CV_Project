from PIL import Image

def apply_mask(image_path, mask_path, output_path):
    # 打开图片和mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    # 获取图片和mask的像素数据
    image_data = image.load()
    mask_data = mask.load()
    
    # 遍历所有像素，根据mask调整图片
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            if mask_data[x, y] == 0:  # mask中为黑色的部分
                image_data[x, y] = (255, 255, 255)  # 图片上表示为白色

    # 保存处理后的图片
    image.save(output_path)

# 使用示例
image_path = "../source.png"
mask_path = "../mask.png"
output_path = "../source_mask.png"

apply_mask(image_path, mask_path, output_path)
