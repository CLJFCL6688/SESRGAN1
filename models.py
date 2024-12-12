# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
# '''
# @文件        :models.py
# @说明        :模型定义文件
# @时间        :2020/02/13 11:42:33
# @作者        :钱彬
# @版本        :1.0
# '''


import torch
from torch import nn
import torchvision
import math
from torchvision.models import vgg19
from torchvision.models.vgg import VGG19_Weights


class ConvolutionalBlock(nn.Module):
    """
    卷积模块,由卷积层, BN归一化层, 激活层构成.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        """
        :参数 in_channels: 输入通道数
        :参数 out_channels: 输出通道数
        :参数 kernel_size: 核大小
        :参数 stride: 步长
        :参数 batch_norm: 是否包含BN层
        :参数 activation: 激活层类型; 如果没有则为None
        """
        super(ConvolutionalBlock, self).__init__()
        print("ConvolutionalBlock - In Channels:", in_channels, "Out Channels:", out_channels)

        if activation is not None:
            activation = activation.lower()
            assert activation in {'PRelu', 'LeakyRelu', 'tanh'}

        # 层列表
        layers = list()

        # 1个卷积层
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # 1个BN归一化层
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # 1个激活层
        if activation == 'PRelu':
            layers.append(nn.PReLU())
        elif activation == 'LeakyRelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        # 合并层
        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        """
        前向传播

        :参数 input: 输入图像集，张量表示，大小为 (N, in_channels, w, h)
        :返回: 输出图像集，张量表示，大小为(N, out_channels, w, h)
        """
        print("ConvolutionalBlock - Input Shape:", input.shape)
        output = self.conv_block(input)
        print("ConvolutionalBlock - Output Shape:", output.shape)
        return output


class SubPixelConvolutionalBlock(nn.Module):
    """
    子像素卷积模块, 包含卷积, 像素清洗和激活层.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
        """
        :参数 kernel_size: 卷积核大小
        :参数 n_channels: 输入和输出通道数
        :参数 scaling_factor: 放大比例
        """
        super(SubPixelConvolutionalBlock, self).__init__()
        print("SubPixelConvolutionalBlock - Input and Output Channels:", n_channels)
        # 首先通过卷积将通道数扩展为 scaling factor^2 倍
        self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # 进行像素清洗，合并相关通道数据
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        # 最后添加激活层
        self.PRelu = nn.PReLU()

    def forward(self, input):
        """
        前向传播.

        :参数 input: 输入图像数据集，张量表示，大小为(N, n_channels, w, h)
        :返回: 输出图像数据集，张量表示，大小为 (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class ResidualBlock(nn.Module):
    """
    残差模块, 包含两个卷积模块和一个跳连.
    """

    def __init__(self, kernel_size=3, n_channels=64):
        """
        :参数 kernel_size: 核大小
        :参数 n_channels: 输入和输出通道数（由于是ResNet网络，需要做跳连，因此输入和输出通道数是一致的）
        """
        super(ResidualBlock, self).__init__()

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation='PReLu')

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                              batch_norm=True, activation=None)

    def forward(self, input):
        """
        前向传播.

        :参数 input: 输入图像集，张量表示，大小为 (N, n_channels, w, h)
        :返回: 输出图像集，张量表示，大小为 (N, n_channels, w, h)
        """
        residual = input  # (N, n_channels, w, h)
        output = self.conv_block1(input)  # (N, n_channels, w, h)
        output = self.conv_block2(output)  # (N, n_channels, w, h)
        output = output + residual  # (N, n_channels, w, h)

        return output

class RFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFB, self).__init__()
        # 1x1卷积进行通道分配
        self.channel_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 不同感受野的支路卷积，可以根据具体需求设置不同的空洞率等参数
        self.branches = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=d, padding=d)
            for d in [1, 3, 5]
        ])
        # 通道调整后的卷积
        self.fusion_conv = nn.Conv2d(len(self.branches) * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        adjusted_x = self.channel_adjust(x)
        branch_outputs = [branch(adjusted_x) for branch in self.branches]
        concatenated_output = torch.cat(branch_outputs, dim=1)
        fused_output = self.fusion_conv(concatenated_output)
        return fused_output


class OSA_RFB(nn.Module):
    def __init__(self):
        super(OSA_RFB, self).__init__()
        # 定义多尺度卷积层，例如不同大小的卷积核进行卷积操作
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        # 定义高效通道注意力机制（ECA）模块
        self.eca_module = ECA_Module()

    def forward(self, x):
        # 检测输入通道数
        if x.shape[1]!= 64:
            raise ValueError(f"Unexpected input channels in OSA_RFB: {x.shape[1]}")
        # 多尺度卷积的输出融合
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        fused_out = torch.cat([out1, out2], dim=1)
        # 通过 ECA 模块
        output = self.eca_module(fused_out)
        return output


class ECA_Module(nn.Module):
    def __init__(self):
        super(ECA_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), 1, -1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.view(x.size(0), -1, 1, 1)
        return x * y


class ROSA_RFB(nn.Module):
    def __init__(self):
        super(ROSA_RFB, self).__init__()
        self.osarfb_modules = nn.ModuleList([OSA_RFB() for _ in range(16)])

    def forward(self, x):
        for module in self.osarfb_modules:
            x = module(x)
        return x

class SRResNet(nn.Module):
    """
    SRResNet模型
    """
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        :参数 large_kernel_size: 第一层卷积和最后一层卷积核大小
        :参数 small_kernel_size: 中间层卷积核大小
        :参数 n_channels: 中间层通道数
        :参数 n_blocks: 残差模块数
        :参数 scaling_factor: 放大比例
        """
        super(SRResNet, self).__init__()

        # 放大比例必须为 2、 4 或 8
        scaling_factor = int(scaling_factor)
        assert scaling_factor in {2, 4, 8}, "放大比例必须为 2、 4 或 8!"

        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        # 一系列残差模块, 每个残差模块包含一个跳连接
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        # 放大通过子像素卷积模块实现, 每个模块放大两倍
        n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        # 最后一个卷积模块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        print (lr_imgs.shape)
        """
        前向传播.

        :参数 lr_imgs: 低分辨率输入图像集, 张量表示，大小为 (N, 3, w, h)
        :返回: 高分辨率输出图像集, 张量表示， 大小为 (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)  # (16, 3, 24, 24)
        residual = output  # (16, 64, 24, 24)
        output = self.residual_blocks(output)  # (16, 64, 24, 24)
        output = self.conv_block2(output)  # (16, 64, 24, 24)
        output = output + residual  # (16, 64, 24, 24)
        output = self.subpixel_convolutional_blocks(output)  # (16, 64, 24 * 4, 24 * 4)
        sr_imgs = self.conv_block3(output)  # (16, 3, 24 * 4, 24 * 4)

        return sr_imgs


class Generator(nn.Module):
    """
    生成器模型，其结构与SRResNet完全一致.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        参数 large_kernel_size：第一层和最后一层卷积核大小
        参数 small_kernel_size：中间层卷积核大小
        参数 n_channels：中间层卷积通道数
        参数 n_blocks: 残差模块数量
        参数 scaling_factor: 放大比例
        """
        super(Generator, self).__init__()
        # 第一个卷积块
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        # 使用新的多级残差模块
        self.residual_blocks = ROSA_RFB()

        # 上采样部分修改
        self.upconv1 = nn.Conv2d(n_channels, 256, kernel_size=3, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.upconv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)

        # 最后一个卷积块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        """
        前向传播.

        参数 lr_imgs: 低精度图像 (N, 3, w, h)
        返回: 超分重建图像 (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_imgs)
        print(f"After conv_block1, shape: {output.shape}")
        if output.shape[1] == 128:
            output = nn.Conv2d(128, 64, kernel_size=1)(output)
        residual = output
        output = self.residual_blocks(output)
        print(f"After residual_blocks, shape: {output.shape}")
        if output.shape[1] == 128:
            output = nn.Conv2d(128, 64, kernel_size=1)(output)
        sr_imgs = self.conv_block3(output)
        print(f"Before conv_block3, shape: {output.shape}")
        return sr_imgs


class Discriminator(nn.Module):
    """
    SRGAN判别器
    """

    def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
        """
        参数 kernel_size: 所有卷积层的核大小
        参数 n_channels: 初始卷积层输出通道数, 后面每隔一个卷积层通道数翻倍
        参数 n_blocks: 卷积块数量
        参数 fc_size: 全连接层连接数
        """
        super(Discriminator, self).__init__()

        in_channels = 3

        # 卷积系列，使用新的结构和 SN 归一化层
        conv_blocks = list()
        for i in range(n_blocks):
            out_channels = (n_channels if i == 0 else in_channels * 2) if i % 2 == 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 == 0 else 2, batch_norm=False, activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # 固定输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.fc2 = nn.Linear(1024, 1)

    def forward(self, imgs):
        """
        前向传播.

        参数 imgs: 用于作判别的原始高清图或超分重建图，张量表示，大小为(N, 3, w * scaling factor, h * scaling factor)
        返回: 一个评分值， 用于判断一副图像是否是高清图, 张量表示，大小为 (N)
        """
        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)

        return logit


class TruncatedVGG19(nn.Module):
    """
    truncated VGG19网络，用于计算VGG特征空间的MSE损失
    """

    def __init__(self, i, j):
        """
        :参数 i: 第 i 个池化层
        :参数 j: 第 j 个卷积层
        """
        super(TruncatedVGG19, self).__init__()

        # 加载预训练的VGG模型
        vgg19_model = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)

        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        # 迭代搜索
        for layer in vgg19_model.features.children():
            truncate_at += 1

            # 统计
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0

            # 截断位置在第(i-1)个池化层之后（第 i 个池化层之前）的第 j 个卷积层
            if maxpool_counter == i - 1 and conv_counter == j:
                break

        # 检查是否满足条件
        assert maxpool_counter == i - 1 and conv_counter == j, "当前 i=%d 、 j=%d 不满足 VGG19 模型结构" % (
            i, j)

        # 截取网络
        self.truncated_vgg19 = nn.Sequential(*list(vgg19_model.features.children())[:truncate_at + 1])

    def forward(self, input):
        """
        前向传播
        参数 input: 高清原始图或超分重建图，张量表示，大小为 (N, 3, w * scaling factor, h * scaling factor)
        返回: VGG19特征图，张量表示，大小为 (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.truncated_vgg19(input)

        return output
# 假设的高效通道注意力机制（ECA_Module）类
class ECA_Module(nn.Module):
    def __init__(self):
        super(ECA_Module, self).__init__()
        # 这里可以根据实际的 ECA 模块实现进行定义

    def forward(self, input):
        # 这里实现 ECA 模块的前向传播逻辑
        return input