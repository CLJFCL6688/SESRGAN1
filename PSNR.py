# 开发者：吴承志
# 开发时间： 20:46

import math
import cv2
import numpy as np

def psnr(original_image, sr_image):
    sr_image = cv2.resize(sr_image, (original_image.shape[1], original_image.shape[0]))

    mse = np.mean((original_image - sr_image) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# 读入图像并转换为灰度
original_image = cv2.imread(r"E:\programe\pythonodc\SRGAN\srgan\results\L_finger_2.jpg", cv2.IMREAD_GRAYSCALE)
sr_image = cv2.imread(r"E:\programe\pythonodc\SRGAN\srgan\other_SR\Bicubic_Interpolation\H_finger_2.jpg", cv2.IMREAD_GRAYSCALE)

# 将数据类型从unit8转换为float64
original_image = original_image.astype(np.float64)
sr_image = sr_image.astype(np.float64)
psnr_value = psnr(original_image,sr_image)
print("PSNR: {:.4f}".format(psnr_value))



