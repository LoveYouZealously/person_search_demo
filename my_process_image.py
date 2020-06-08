import numpy as np
import cv2


def letterbox(img, new_shape=416, color=(128, 128, 128), mode='auto'):
    """
    求得较长边缩放到416的比例，然后对图片wh按这个比例缩放，使得较长边达到416,
    再对较短边进行填充使得较短边满足32的倍数
    :param img: 需要处理的原始图片CHW
    :param new_shape: 网络的输入分辨率
    :param color: 进行pad时，填充的颜色(值)
    :param mode:需要进行填充的模式
    :return: 返回填充后wh都为32倍数的图片
    """
    # Resize a rectangular image to a 32 pixel multiple rectangle
    # https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width] (1080, 810)

    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape) # 416.0 / 1080 = 0.3851851851851852
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    # round() 方法返回浮点数x的四舍五入值。
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio))) # WH:(312, 416)

    # Compute padding https://github.com/ultralytics/yolov3/issues/232
    if mode is 'auto':
        # 填充为符合条件的最小矩形minimum rectangle
        # 使得较长边达到416, 再对较短边进行填充使得较短边满足32的倍数
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding  4.0
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding 0.0
    elif mode is 'square':  # square
        # 直接填充为416x416的正方形
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode is 'rect':  # square
        # 填充为指定形状new_shape=(320, 416)的矩形
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode is 'scaleFill':
        # resize到指定的416x416
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    if shape[::-1] != new_unpad:  # new_unpad: (312, 416)
        # 进行resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1)) # 0, 0
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1)) # 4, 4
    # 为图像扩边（填充）
    # 想为图像周围建一个边可以使用cv2.copyMakeBorder()函数。这经常在卷积运算或0填充时被用到。具体参数如下：
    # 5.1 src输入图像
    # 5.2 top,bottom,left,right对应边界的像素数目
    # 5.3 borderType要添加哪种类型的边界：
    # 5.3.1	cv2.BORDER_CONSTANT添加有颜色的常数值边界，还需要下一个参数（value）
    # 5.3.2	cv2.BORDER_REFLIECT边界元素的镜像。例如：fedcba | abcdefgh | hgfedcb
    # 5.3.3	cv2.BORDER_101或者cv2.BORDER_DEFAULT跟上面一样，但稍作改动，例如：gfedcb | abcdefgh | gfedcba
    # 5.3.4	cv2.BORDER_REPLICATE复后一个元素。例如: aaaaaa| abcdefgh|hhhhhhh
    # 5.3.5	cv2.BORDER_WRAP 不知怎么了, 就像样: cdefgh| abcdefgh|abcdefg
    # 5.3.6	value边界颜色
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratiow, ratioh, dw, dh


def process_img(img0):
    img_size = 416
    half = False

    img0 = cv2.flip(img0, 1)  # flip left-right

    # Padded resize
    img, *_ = letterbox(img0, new_shape=img_size)

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    return img, img0