import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
        :参数 batch_norm: 是否包含 BN 层
        :参数 activation: 激活层类型; 如果没有则为 None
        """
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'PRelu', 'LeakyRelu', 'tanh'}

        # 层列表
        layers = list()

        # 1 个卷积层
        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        # 1 个 BN 归一化层
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        # 1 个激活层
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
        :返回: 输出图像集，张量表示，大小为 (N, out_channels, w, h)
        """
        output = self.conv_block(input)

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

        :参数 input: 输入图像数据集，张量表示，大小为 (N, n_channels, w, h)
        :返回: 输出图像数据集，张量表示，大小为 (N, n_channels, w * scaling factor, h * scaling factor)
        """
        output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.PRelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output


class RFB(nn.Module):
    def __init__(self, in_channels):
        super(RFB, self).__init__()
        self.conv1x1_channel_distribution = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        branches = []
        for dilation in [1, 3, 5]:
            branches.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation))
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        distributed_channels = self.conv1x1_channel_distribution(x)
        branch_outputs = [branch(distributed_channels) for branch in self.branches]
        fused_branches = torch.cat(branch_outputs, dim=1)
        return fused_branches


class OSA_RFB(nn.Module):
    def __init__(self):
        super(OSA_RFB, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        rfb_modules = []
        for _ in range(3):
            rfb_modules.append(RFB(64))
        self.rfb_modules = nn.ModuleList(rfb_modules)
        self.eca_module = ECA_Module()

    def forward(self, x):
        adjusted_x = self.conv1x1(x)
        output = adjusted_x
        for rfb_module in self.rfb_modules:
            next_output = rfb_module(output)
            output = next_output
        fused_output = torch.cat([output, adjusted_x], dim=1)
        return self.eca_module(fused_output)


class ROSA_RFB(nn.Module):
    def __init__(self):
        super(ROSA_RFB, self).__init__()
        osa_rfb_modules = [OSA_RFB() for _ in range(16)]
        self.osa_rfb_modules = nn.ModuleList(osa_rfb_modules)

    def forward(self, x):
        for module in self.osa_rfb_modules:
            x = module(x)
        return x


class Generator(nn.Module):
    def __init__(self, scaling_factor=4):
        super(Generator, self).__init__()
        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=64, kernel_size=9, activation='PReLu')
        self.residual_blocks = ROSA_RFB()
        self.upconv1 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.upconv2 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)
        residual = output
        output = self.residual_blocks(output)
        output = self.upconv1(output)
        output = self.pixel_shuffle1(output)
        output = self.upconv2(output)
        output = self.pixel_shuffle2(output)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        conv_blocks = []
        in_channels = 3
        for i in range(8):
            out_channels = in_channels * 2 if i % 2 == 0 else in_channels
            conv_blocks.append(
                ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1 if i % 2 == 0 else 2,
                                   batch_norm=False, activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(out_channels * 6 * 6, 1024)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, imgs):
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(output.size(0), -1))
        output = self.leaky_relu(output)
        logit = self.fc2(output)
        return logit


class TruncatedVGG19(nn.Module):
    def __init__(self, i, j):
        super(TruncatedVGG19, self).__init__()
        vgg19_model = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
        maxpool_counter = 0
        conv_counter = 0
        truncate_at = 0
        for layer in vgg19_model.features.children():
            truncate_at += 1
            if isinstance(layer, nn.Conv2d):
                conv_counter += 1
            if isinstance(layer, nn.MaxPool2d):
                maxpool_counter += 1
                conv_counter = 0
            if maxpool_counter == i - 1 and conv_counter == j:
                break
        self.truncated_vgg19 = nn.Sequential(*list(vgg19_model.features.children())[:truncate_at + 1])

    def forward(self, input):
        return self.truncated_vgg19(input)


class ECA_Module(nn.Module):
    def __init__(self):
        super(ECA_Module, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.adaptive_kernel_size(64), padding=(self.adaptive_kernel_size(64) - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        batch_size, channels, height, width = input.size()
        avg_pool = self.gap(input).view(batch_size, channels)
        eca_output = self.conv(avg_pool.unsqueeze(-1)).squeeze(-1)
        return self.sigmoid(eca_output).view(batch_size, channels, 1, 1) * input

    def adaptive_kernel_size(self, channels):
        k = int(abs(math.log2(channels) / 2 + 0.5 / 2))
        k = k if k % 2 == 1 else k + 1
        return k