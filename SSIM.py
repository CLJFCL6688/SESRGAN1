# 开发者：吴承志
# 开发时间： 10:29
import numpy as np
import cv2


def ssim(img1, img2):
    """计算两个图像的结构相似性指标(SSIM)"""

    # 将图像转换为浮点数类型，并确保它们具有相同的尺寸
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if img1.shape != img2.shape:
        # 将img2的尺寸调整为与img1相同
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 计算C1和C2常数，这些常数用于稳定SSIM计算
    k1 = 0.01
    k2 = 0.03
    L = 255  # 像素值的最大范围
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    # 计算均值和方差
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    # 计算SSIM
    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_num / ssim_den
    mssim = np.mean(ssim_map)

    return mssim
img1 = cv2.imread(r"E:\programe\pythonodc\SRGAN\srgan\results\L_black_core_2.jpg")
img2 = cv2.imread(r"E:\programe\pythonodc\SRGAN\srgan\other_SR\Bicubic_Interpolation\Bilinear_Interpolation_L_black_core_2.jpg")
ssim_val = ssim(img1, img2)
print("SSIM指标为:", ssim_val)
