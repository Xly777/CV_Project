import os
import csv
import numpy as np
from PIL import Image
from utils.criteria import calculate_psnr, calculate_fid, calculate_ssim
from tqdm import tqdm
# 指定原图片目录和处理后的图片目录父目录
origin_directory = 'datasets/celeb'
outputs_directory = 'outputs/celeb'

# 获取处理方法目录列表
methods = [d for d in os.listdir(outputs_directory) if os.path.isdir(os.path.join(outputs_directory, d))]
def get_pms():
# 处理每个方法
    for method in tqdm(methods):
        method_dir = os.path.join(outputs_directory, method)
        output_file = f'{method}_metrics.csv'
        
        results = []
        
        for origin_file in tqdm(os.listdir(origin_directory)):
            origin_path = os.path.join(origin_directory, origin_file)
            # 寻找处理后的文件
            processed_file = None
            for ext in ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff']:
                potential_file = os.path.splitext(origin_file)[0] + '.' + ext
                if os.path.exists(os.path.join(method_dir, potential_file)):
                    processed_file = potential_file
                    break
            if processed_file is None:
                continue  # 如果处理后的文件没有找到，跳过
            
            processed_path = os.path.join(method_dir, processed_file)
            
            # 打开原图片和处理后的图片
            origin_img = np.array(Image.open(origin_path).convert('RGB'))
            processed_img = np.array(Image.open(processed_path).convert('RGB'))
            
            # 计算 PSNR, SSIM 和 FID
            psnr_value = calculate_psnr(origin_img, processed_img)
            ssim_value = calculate_ssim(origin_img, processed_img)
            # fid_value = calculate_fid(origin_img, processed_img)
            
            results.append([origin_file, psnr_value, ssim_value])
        
        # 计算平均值
        avg_psnr = np.mean([r[1] for r in results])
        avg_ssim = np.mean([r[2] for r in results])
        
        # 保存到 CSV
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['Image', 'PSNR', 'SSIM', 'FID']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in results:
                writer.writerow({'Image': row[0], 'PSNR': row[1], 'SSIM': row[2]})
            writer.writerow({'Image': 'Average', 'PSNR': avg_psnr, 'SSIM': avg_ssim})
        
        print(f'{output_file} has been created.')

def get_fid(batch_size=50, device='cuda'):
    from torchvision import transforms, models
    from scipy import linalg
    import torch

    # 定义图像转换
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 定义函数来加载图像
    def load_images(image_paths):
        images = []
        for path in image_paths:
            with Image.open(path) as img:
                img = img.convert('RGB')
                images.append(transform(img))
        return torch.stack(images)

    # 加载InceptionV3模型
    inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()

    def get_activations(images):
        n_batches = len(images) // batch_size + 1
        activations = []
        with torch.no_grad():
            for i in range(n_batches):
                batch_images = images[i*batch_size:(i+1)*batch_size].to(device)
                if batch_images.size(0) == 0:
                    continue
                pred = inception(batch_images)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = pred.view(pred.size(0), -1)  # Flatten the output
                activations.append(pred.cpu().numpy())
        activations = np.concatenate(activations, axis=0)
        print(f"Activations shape: {activations.shape}")
        return activations

    def calculate_statistics(images):
        act = get_activations(images)
        print(f"act {act.shape}")
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        print(f"len{mu.shape} len({sigma.shape})")
        return mu, sigma

    def calculate_fid(mu1, sigma1, mu2, sigma2):
        print("Mu1 shape:", mu1.shape)
        print("Sigma1 shape:", sigma1.shape)
        print("Mu2 shape:", mu2.shape)
        print("Sigma2 shape:", sigma2.shape)

        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    fid_results = {}

    for method in tqdm(methods):
        method_path = os.path.join(outputs_directory, method)
        generated_images = {f: os.path.join(method_path, f) for f in os.listdir(method_path) if os.path.isfile(os.path.join(method_path, f))}
        
        common_images = list(set(os.listdir(origin_directory)).intersection(generated_images.keys()))
        print(f"Common images for method {method}: {len(common_images)}")
        print(common_images)

        if not common_images:
            print(f"No common images found for method: {method}")
            continue
        
        origin_image_paths = [os.path.join(origin_directory, f) for f in common_images]
        generated_image_paths = [generated_images[f] for f in common_images]
        
        origin_images = load_images(origin_image_paths).to(device)
        generated_images = load_images(generated_image_paths).to(device)

        mu1, sigma1 = calculate_statistics(origin_images)
        
        mu2, sigma2 = calculate_statistics(generated_images)
        
        fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)
        fid_results[method] = fid_value/len(origin_images)

    for method, fid_value in fid_results.items():
        print(f"Method: {method}, FID: {fid_value}")

get_pms()
# get_fid()