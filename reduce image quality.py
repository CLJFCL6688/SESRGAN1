import os  # 导入os模块主要用于文件的读写
import argparse  # 导入argpase主要是用来命令行运行时参数的配置
import cv2  # 图像处理模块

# parse args
parser = argparse.ArgumentParser(
    description='Downsize images at 2x using bicubic interpolation')  # 创建一个参数解析对象，为解析对象添加描述语句，而这个描述语句是当调用parser.print_help()或者运行程序时由于参数不正确(此时python解释器其实也是调用了pring_help()方法)时，会打印这些描述信息
parser.add_argument("-k", "--keepdims", help="keep original image dimensions in downsampled images",
                    action="store_true")  # 为函数添加参数k和参数keepdims，并且设置action="store_true"，也就是当命令行提及到这两个参数的时候，参数设置为true，如果没提及那就是默认值（如果用了default制定了默认值的话）
parser.add_argument('--hr_img_dir', type=str, default='C:/Users/ZhuanZ/Desktop/me/SRGAN/data/PCB_picture/my_hr_images_folder',  # 设置高分辨率图片路径参数
                    help='path to high resolution image dir')
parser.add_argument('--lr_img_dir', type=str, default='C:/Users/ZhuanZ/Desktop/me/SRGAN/data/PCB_picture/my_lr_output_folder',  # 设置低分辨率路径参数
                    help='path to desired output dir for downsampled images')
args = parser.parse_args()  # 调用parse_args()方法对参数进行解析；解析成功之后即可使用

hr_image_dir = args.hr_img_dir  # 从参数列表中取出高分辨率图像路径
lr_image_dir = args.lr_img_dir  # 从参数列表中取出低分辨率图像路径

print(args.hr_img_dir)  # 将高分辨率图像路径打印出来
print(args.lr_img_dir)  # 将低分辨率图像路径打印出来

# create LR image dirs
# 在低分辨率图像路径中创建每个下采样倍率的文件夹
os.makedirs(lr_image_dir + "/X2", exist_ok=True)  # 创建下采样2倍率的文件夹，exist_ok：只有在目录不存在时创建目录，目录已存在时不会抛出异常
os.makedirs(lr_image_dir + "/X3", exist_ok=True)
os.makedirs(lr_image_dir + "/X4", exist_ok=True)
os.makedirs(lr_image_dir + "/X6", exist_ok=True)
os.makedirs(lr_image_dir + "/X8", exist_ok=True)

supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                         ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                         ".tiff")  # 在这里用一个元组保存支持进行下采样的图像的后缀格式

# Downsample HR images
# 对高分辨率图像进行下采样
for filename in os.listdir(hr_image_dir):  # 遍历高分辨率图像文件夹中的每一个文件
    if not filename.endswith(supported_img_formats):  # 如果文件的后缀名不是支持下采样的图片格式，那么就跳过这张图片
        continue

    name, ext = os.path.splitext(filename)  # os.path.splitext(“文件路径”)：分离文件名与扩展名；默认返回(fname,fextension)元组
    # 在这里，我们将遍历的每个文件的文件名存在变量name中，后缀存在ext中

    # Read HR image
    hr_img = cv2.imread(os.path.join(hr_image_dir, filename))
    """
    os.path.join()函数用于路径拼接文件路径,在这里也就是将文件所在目录和文件名拼接在一起，获得文件完整的路径
    cv2.imread:为 opencv-python 包的读取图片的函数,cv2.imread()有两个参数,第一个参数filename是图片路径,第二个参数flag表示图片读取模式,共有三种
    cv2.IMREAD_COLOR:加载彩色图片,这个是默认参数,可以直接写1。
    cv2.IMREAD_GRAYSCALE:以灰度模式加载图片,可以直接写0。
    cv2.IMREAD_UNCHANGED:包括alpha(包括透明度通道)，可以直接写-1
    cv2.imread()读取图片后以多维数组的形式保存图片信息，前两维表示图片的像素坐标,最后一维表示图片的通道索引,具体图像的通道数由图片的格式来决定
    """

    hr_img_dims = (hr_img.shape[1], hr_img.shape[0])  # 获得高清图片的分辨率，分辨率值为一个(high,wide)的元组

    # Blur with Gaussian kernel of width sigma = 1
    # 设置一个宽度为1的高斯模糊核
    hr_img = cv2.GaussianBlur(hr_img, (0, 0), 1, 1)
    """
    高斯滤波是对整幅图像进行加权平均的过程，每一个像素点的值都由其本身和邻域内的其他像素值经过加权平均后得到。高斯滤波的具体操作是：用一个模板（或称卷积、掩模）扫描图像中的每一个像素，
    用模板确定的邻域内像素的加权平均灰度值去替代模板中心像素点的值,基于二维高斯函数，构建权重矩阵，进而构建高斯核，最终对每个像素点进行滤波处理（平滑、去噪）
    dst=cv2.GaussianBlur(src,ksize,sigmaX,sigmaY,borderType)
    dst是返回值,表示进行高斯滤波后得到的处理结果
    src 是需要处理的图像，即原始图像。它能够有任意数量的通道,并能对各个通道独立处理。图像深度应该是CV_8U、CV_16U、CV_16S、CV_32F 或者 CV_64F中的一 种
    ksize 是滤波核的大小。滤波核大小是指在滤波处理过程中其邻域图像的高度和宽度。需要注意，滤波核的值必须是奇数
    sigmaX 是卷积核在水平方向上(X 轴方向）的标准差，其控制的是权重比例
    sigmaY 是卷积核在垂直方向上(Y轴方向)的标准差。如果将该值设置为0,则只采用sigmaX的值
    如果sigmaX和sigmaY都是0，则通过ksize.width和ksize.height计算得到。其中:sigmaX=0.3*[(ksize.width-1)*0.5-1] +0.8,sigmaY=0.3*[(ksize.height-1)*0.5-1]+0.8

    """
    # cv2.GaussianBlur(hr_img, (0,0), 1, 1)   其中模糊核这里用的0。两个1分别表示x、y方向的标准差。 可以具体查看该函数的官方文档。
    # Downsample image 2x
    lr_image_2x = cv2.resize(hr_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    """
    resize是opencv库中的一个函数，主要起到对图片进行缩放的作用
    resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR )
    InputArray src:要进行缩放的图片
    OutputArray dst： 输出，改变后的图像。这个图像和原图像具有相同的内容，只是大小和原图像不一样而已
    dsize：输出图像的大小，如果是(0,0)我认为意思就是在这里不指定输出图像的大小
    fx：width方向的缩放比例
    fy：height方向的缩放比例
    interpolation（插值）：这个是指定插值的方式，图像缩放之后，肯定像素要进行重新计算的，就靠这个参数来指定重新计算像素的方式，有以下几种：
        INTER_NEAREST - 最邻近插值
        INTER_LINEAR - 双线性插值，如果最后一个参数你不指定，默认使用这种方法
        INTER_CUBIC - 4x4像素邻域内的双立方插值
        INTER_LANCZOS4 - 8x8像素邻域内的Lanczos插值
        在本代码中使用双线性插值
    """
    if args.keepdims:  # 如果要保持原来分辨率的话
        lr_image_2x = cv2.resize(lr_image_2x, hr_img_dims,
                                 interpolation=cv2.INTER_CUBIC)  # 就将缩小成两倍的图片再用双线性插值法重新放大到原来的高清的图像分辨率

    cv2.imwrite(os.path.join(lr_image_dir + "/X2", filename.split('.')[0] + 'x2' + ext),
                lr_image_2x)  # 将文件写入准备好的低分辨率中的"x2"文件夹中，并在原有的文件名基础之上加上字符串'x2'

    # Downsample image 3x
    lr_img_3x = cv2.resize(hr_img, (0, 0), fx=(1 / 3), fy=(1 / 3),
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_3x = cv2.resize(lr_img_3x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X3", filename.split('.')[0] + 'x3' + ext), lr_img_3x)

    # Downsample image 4x
    lr_img_4x = cv2.resize(hr_img, (0, 0), fx=0.25, fy=0.25,
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_4x = cv2.resize(lr_img_4x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X4", filename.split('.')[0] + 'x4' + ext), lr_img_4x)

    # Downsample image 6x
    lr_img_6x = cv2.resize(hr_img, (0, 0), fx=1 / 6, fy=1 / 6,
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_4x = cv2.resize(lr_img_6x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X6", filename.split('.')[0] + 'x6' + ext), lr_img_6x)

    # Downsample image 8x
    lr_img_8x = cv2.resize(hr_img, (0, 0), fx=1 / 8, fy=1 / 8,
                           interpolation=cv2.INTER_CUBIC)
    if args.keepdims:
        lr_img_8x = cv2.resize(lr_img_8x, hr_img_dims,
                               interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(lr_image_dir + "/X8", filename.split('.')[0] + 'x8' + ext), lr_img_8x)

print("下采样成功")
