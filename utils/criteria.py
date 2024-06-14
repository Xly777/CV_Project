import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.linalg import sqrtm
from skimage.color import rgb2gray

def calculate_psnr(origin_img, gen_img):
    if origin_img.ndim == 3:
        origin_img = rgb2gray(origin_img)
    if gen_img.ndim == 3:
        gen_img = rgb2gray(gen_img)
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

def calculate_fid(origin_img, gen_img):
    # Ensure images are 3D
    if origin_img.ndim == 2:
        origin_img = np.stack([origin_img] * 3, axis=-1)
    if gen_img.ndim == 2:
        gen_img = np.stack([gen_img] * 3, axis=-1)

    # Convert numpy arrays to tensors
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    origin_img = preprocess(origin_img).unsqueeze(0)
    gen_img = preprocess(gen_img).unsqueeze(0)

    # Load pre-trained InceptionV3 model
    model = models.inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()  # Remove final fully connected layer
    model.eval()

    # Extract features
    with torch.no_grad():
        origin_features = model(origin_img).numpy()
        gen_features = model(gen_img).numpy()

    # Calculate mean and covariance statistics
    mu1, sigma1 = np.mean(origin_features, axis=0), np.cov(origin_features, rowvar=False)
    mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)

    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
