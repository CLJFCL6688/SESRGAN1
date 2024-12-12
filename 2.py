import torch
import math

# RFB 模块
class RFBModule(torch.nn.Module):
    def __init__(self, in_channels):
        super(RFBModule, self).__init__()
        self.channel_distribution = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        branches = []
        for dilation in [1, 3, 5]:
            branches.append(torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation))
        self.branches = torch.nn.ModuleList(branches)

    def forward(self, x):
        distributed_channels = self.channel_distribution(x)
        branch_outputs = [branch(distributed_channels) for branch in self.branches]
        fused_branches = torch.cat(branch_outputs, dim=1)
        return fused_branches

# OSA-RFB 模块
class OSARFBModule(torch.nn.Module):
    def __init__(self):
        super(OSARFBModule, self).__init__()
        self.channel_adjust = torch.nn.Conv2d(64, 64, kernel_size=1)
        rfb_modules = [RFBModule(64) for _ in range(3)]
        self.rfb_modules = torch.nn.ModuleList(rfb_modules)
        self.eca_module = ECAModule()

    def forward(self, x):
        adjusted_x = self.channel_adjust(x)
        output = adjusted_x
        for rfb_module in self.rfb_modules:
            next_output = rfb_module(output)
            output = next_output
        fused_output = torch.cat([output, adjusted_x], dim=1)
        return self.eca_module(fused_output)

# ROSA-RFB 模块
class ROSARFBModule(torch.nn.Module):
    def __init__(self):
        super(ROSARFBModule, self).__init__()
        osa_rfb_modules = [OSARFBModule() for _ in range(16)]
        self.osa_rfb_modules = torch.nn.ModuleList(osa_rfb_modules)

    def forward(self, x):
        for module in self.osa_rfb_modules:
            x = module(x)
        return x

# ECA 模块
class ECAModule(torch.nn.Module):
    def __init__(self):
        super(ECAModule, self).__init__()
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=self.adaptive_kernel_size(64), padding=(self.adaptive_kernel_size(64) - 1) // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        batch_size, channels, height, width = input.size()
        avg_pool = self.gap(input).view(batch_size, channels)
        eca_output = self.conv(avg_pool.unsqueeze(-1)).squeeze(-1)
        return self.sigmoid(eca_output).view(batch_size, channels, 1, 1) * input

    def adaptive_kernel_size(self, channels):
        k = int(abs(math.log2(channels) / 2 + 0.5 / 2))
        k = k if k % 2 == 1 else k + 1
        return k

# 生成器
class GeneratorModule(torch.nn.Module):
    def __init__(self, scaling_factor=4):
        super(GeneratorModule, self).__init__()
        self.conv_block1 = torch.nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.residual_blocks = ROSARFBModule()
        self.up_conv1 = torch.nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pixel_shuffle1 = torch.nn.PixelShuffle(2)
        self.up_conv2 = torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.pixel_shuffle2 = torch.nn.PixelShuffle(2)

    def forward(self, lr_imges):
        output = self.conv_block1(lr_imges)
        residual = output
        output = self.residual_blocks(output)
        output = self.upconv1(output)
        output = self.pixel_shuffle1(output)
        output = self.upconv2(output)
        output = self.pixel_shuffle2(output)
        return output

# 判别器
class DiscriminatorModule(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorModule, self).__init__()
        conv_blocks = []
        in_channels = 3
        for i in range(8):
            out_channels = in_channels * 2 if i % 2 == 0 else in_channels
            conv_blocks.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1 if i % 2 == 0 else 2, padding=1))
            conv_blocks.append(torch.nn.LeakyReLU(0.2))
            in_channels = out_channels
        self.conv_blocks = torch.nn.Sequential(*conv_blocks)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = torch.nn.Linear(out_channels * 6 * 6,1024)
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.fc2 = torch.nn.Linear(1024, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_imges):
        output = self.conv_blocks(input_imges)
        output = self.adaptive_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.leaky_relu(output)
        output = self.fc2(output)
        return self.sigmoid(output)