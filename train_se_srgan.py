import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from models import Generator, Discriminator, TruncatedVGG19
from datasets import SRDataset
from utils import *

# 数据集参数
data_folder = 'D:/Cfbl/RE-SRGAN/data/PCB_picture/PCB'  # 数据存放路径
crop_size = 96  # 高分辨率图像裁剪尺寸
scaling_factor = 4  # 放大比例

# 生成器模型参数(与SRResNet相同)
large_kernel_size_g = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size_g = 3  # 中间层卷积的核大小
n_channels_g = 64  # 中间层通道数
n_blocks_g = 16  # 残差模块数量
srresnet_checkpoint = "./results/checkpoint_srresnet.pth"  # 预训练的SRResNet模型，用来初始化

# 判别器模型参数
kernel_size_d = 3  # 所有卷积模块的核大小
n_channels_d = 64  # 第1层卷积模块的通道数, 后续每隔1个模块通道数翻倍
n_blocks_d = 8  # 卷积模块数量
fc_size_d = 1024  # 全连接层连接数

# 学习参数
batch_size = 128  # 批大小
start_epoch = 1  # 迭代起始位置
epochs = 50  # 迭代轮数
checkpoint = None  # SRGAN预训练模型, 如果没有则填None
workers = 4  # 加载数据线程数量
vgg19_i = 5  # VGG19网络第i个池化层
vgg19_j = 4  # VGG19网络第j个卷积层
beta = 1e-3  # 判别损失乘子
lr = 1e-4  # 学习率

# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = 1  # 用来运行的gpu数量
cudnn.benchmark = True  # 对卷积进行加速
writer = SummaryWriter('D:/Cfbl/RE-SRGAN/my_se_srgan_log_paths')  # 实时监控     使用命令tensorboard --logdir runs进行查看


def main():
    """
    训练.
    """
    global checkpoint, start_epoch, writer

    # 模型初始化
    # 使用新的生成器结构
    generator = Generator(large_kernel_size=large_kernel_size_g,
                          small_kernel_size=small_kernel_size_g,
                          n_channels=n_channels_g,
                          n_blocks=n_blocks_g,
                          scaling_factor=scaling_factor)

    # 使用新的判别器结构
    discriminator = Discriminator(kernel_size=kernel_size_d,
                                  n_channels=n_channels_d,
                                  n_blocks=n_blocks_d,
                                  fc_size=fc_size_d)

    # 初始化优化器
    optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()), lr=lr)
    optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr)

    # 截断的VGG19网络用于计算损失函数
    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    truncated_vgg19.eval()

    # 损失函数
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    # 将数据移至默认设备
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    truncated_vgg19 = truncated_vgg19.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # 加载预训练模型
    srresnetcheckpoint = torch.load(srresnet_checkpoint)
    pretrained_dict = srresnetcheckpoint['model']
    model_dict = generator.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    # 根据新模型结构调整加载逻辑
    generator.load_state_dict(model_dict)

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])

    # 单机多GPU训练
    if torch.cuda.is_available() and ngpu > 1:
        generator = nn.DataParallel(generator, device_ids=list(range(ngpu)))
        discriminator = nn.DataParallel(discriminator, device_ids=list(range(ngpu)))

    # 定制化的dataloaders
    train_dataset = SRDataset(data_folder, split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='imagenet-norm')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)

    # 开始逐轮训练
    for epoch in range(start_epoch, epochs + 1):

        if epoch == int(epochs / 2):  # 执行到一半时降低学习率
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        generator.train()  # 开启训练模式：允许使用批样本归一化
        discriminator.train()

        losses_c = AverageMeter()  # 内容损失
        losses_a = AverageMeter()  # 生成损失
        losses_d = AverageMeter()  # 判别损失

        n_iter = len(train_loader)

        # 按批处理
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):

            # 数据移至默认设备进行训练
            lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24),  imagenet-normed 格式
            hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96),  imagenet-normed 格式

            # -----------------------1. 生成器更新----------------------------
            # 生成
            sr_imgs = generator(lr_imgs)
            # 根据新生成器输出可能需要的调整
            if hasattr(generator, 'output_range') and generator.output_range == '[-1, 1]':
                sr_imgs = convert_image(
                    sr_imgs, source='[-1, 1]',
                    target='imagenet-norm')
             # 检查并调整尺寸
            if sr_imgs.shape != hr_imgs.shape:
                    sr_imgs = torch.nn.functional.interpolate(sr_imgs, size=hr_imgs.shape[2:], mode='bilinear',
                                                              align_corners=False)
            # 计算 VGG 特征图
            sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
            hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

            # 计算内容损失
            content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)

            # 计算生成损失
            sr_discriminated = discriminator(sr_imgs)
            adversarial_loss = adversarial_loss_criterion(
                sr_discriminated, torch.ones_like(sr_discriminated))

            # 计算总的感知损失
            perceptual_loss = content_loss + beta * adversarial_loss

            # 后向传播.
            optimizer_g.zero_grad()
            perceptual_loss.backward()

            # 更新生成器参数
            optimizer_g.step()

            # 记录损失值
            losses_c.update(content_loss.item(), lr_imgs.size(0))
            losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

            # -----------------------2. 判别器更新----------------------------
            # 判别器判断
            hr_discriminated = discriminator(hr_imgs)
            sr_discriminated = discriminator(sr_imgs.detach())

            # 二值交叉熵损失
            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                               adversarial_loss_criterion(hr_discriminated, torch.ones_like(
                                   hr_discriminated))

            # 后向传播
            optimizer_d.zero_grad()
            adversarial_loss.backward()

            # 更新判别器
            optimizer_d.step()

            # 记录损失
            losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

            # 监控图像变化
            if i == (n_iter - 2):
                writer.add_image('SE_SRGAN/epoch_' + str(epoch) + '_1',
                                 make_grid(lr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
                writer.add_image('SE_SRGAN/epoch_' + str(epoch) + '_2',
                                 make_grid(sr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
                writer.add_image('SE_SRGAN/epoch_' + str(epoch) + '_3',
                                 make_grid(hr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)

            # 打印结果
            print("第 " + str(i) + " 个batch结束")

        # 手动释放内存
        del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated

        # 监控损失值变化
        writer.add_scalar('SE_SRGAN/Loss_c', losses_c.val, epoch)
        writer.add_scalar('SE_SRGAN/Loss_a', losses_a.val, epoch)
        writer.add_scalar('SE_SRGAN/Loss_d', losses_d.val, epoch)

        # 保存预训练模型
        torch.save({
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
        }, 'results/checkpoint_se_srgan.pth')

    # 训练结束关闭监控
    writer.close()


if __name__ == '__main__':
    main()