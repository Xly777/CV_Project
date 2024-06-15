import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from skimage.color import rgb2gray
from PIL import Image

def calculate_psnr(origin_img, gen_img):
    mse = np.mean((origin_img - gen_img) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(origin_img, gen_img):
    # Ensure the images are grayscale
    if origin_img.ndim == 3:
        origin_img = origin_img.mean(axis=2)
    if gen_img.ndim == 3:
        gen_img = gen_img.mean(axis=2)
    
    ssim_value = ssim(origin_img, gen_img, data_range=gen_img.max() - gen_img.min())
    return ssim_value
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(Image.fromarray(image))
    image = image.unsqueeze(0)
    return image

# 定义函数来计算图片特征
def extract_features(image, model, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        features = model(image)
    return features.cpu().numpy().squeeze()

# 计算两个高斯分布之间的FID
def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = covmean.real
    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# 定义主函数
def calculate_fid(origin_img, gen_img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.fc = nn.Identity()

    # 预处理图片
    image1 = preprocess_image(origin_img)
    image2 = preprocess_image(gen_img)

    # 获取图片特征
    features1 = extract_features(image1, inception_model, device)
    features2 = extract_features(image2, inception_model, device)

    # 检查特征维度
    print(f'Features1 shape: {features1.shape}')
    print(f'Features2 shape: {features2.shape}')

    # 计算特征均值和协方差
    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    # 检查均值和协方差矩阵的维度
    print(f'mu1 shape: {mu1.shape}, sigma1 shape: {sigma1.shape}')
    print(f'mu2 shape: {mu2.shape}, sigma2 shape: {sigma2.shape}')

    # 计算FID
    fid = compute_fid(mu1, sigma1, mu2, sigma2)
    return fid