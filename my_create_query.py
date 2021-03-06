import time

from models import *
from utils import torch_utils
from utils.datasets import *


cfg='cfg/yolov3.cfg'
data='data/coco.data'
weights='weights/yolo/yolov3.weights'
# parser.add_argument('--images', type=str, default='data/samples', help='需要进行检测的图片文件夹')
# parser.add_argument('-q', '--query', default=r'query', help='查询图片的读取路径.')
img_size=416
# parser.add_argument('--conf-thres', type=float, default=0.1, help='物体置信度阈值')
# parser.add_argument('--nms-thres', type=float, default=0.4, help='NMS阈值')
# parser.add_argument('--dist_thres', type=float, default=1.0, help='行人图片距离阈值，小于这个距离，就认为是该行人')
# parser.add_argument('--fourcc', type=str, default='mp4v', help='fourcc output video codec (verify ffmpeg support)')
# parser.add_argument('--output', type=str, default='output', help='检测后的图片或视频保存的路径')
half=False

device = torch_utils.select_device(force_cpu=False)
conf_thres=0.5
nms_thres=0.5


############# 行人检测模型初始化 #############
model = Darknet(cfg, img_size)

# Load weights
_ = load_darknet_weights(model, weights)

# Eval mode
model.to(device).eval()
# Half precision
half = half and device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.half()

# Set Dataloader
vid_path, vid_writer = None, None
dataloader = LoadWebcam(img_size=img_size, half=half)

# Get classes and colors
# parse_data_cfg(data)['names']:得到类别名称文件路径 names=data/coco.names
classes = load_classes(parse_data_cfg(data)['names']) # 得到类别名列表: ['person', 'bicycle'...]
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]


query_index = 0
for i, (path, img, im0, vid_cap) in enumerate(dataloader):
    # print(img.shape)
    # (3, 320, 416)

    # Get detections shape: (3, 416, 320)
    img = torch.from_numpy(img).unsqueeze(0).to(device) # torch.Size([1, 3, 416, 320])
    pred, _ = model(img) # 经过处理的网络预测，和原始的
    det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0] # torch.Size([5, 7])

    if det is not None and len(det) > 0:
        # Rescale boxes from 416 to true image size 映射到原图
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Print results to screen image 1/3 data\samples\000493.jpg: 288x416 5 persons, Done. (0.869s)
        print('%gx%g ' % img.shape[2:], end='')  # print image size '288x416'
        for c in det[:, -1].unique():   # 对图片的所有类进行遍历循环
            n = (det[:, -1] == c).sum() # 得到了当前类别的个数，也可以用来统计数目
            if classes[int(c)] == 'person':
                print('%g %ss' % (n, classes[int(c)]), end=', ') # 打印个数和类别'5 persons'

        # Draw bounding boxes and labels of detections
        # (x1y1x2y2, obj_conf, class_conf, class_pred)
        gallery_img = []
        gallery_loc = []
        for *xyxy, conf, cls_conf, cls in det: # 对于最后的预测框进行遍历
            # Add bbox to the image
            label = '%s %.2f' % (classes[int(cls)], conf) # 'person 1.00'
            if classes[int(cls)] == 'person':
                #plot_one_bo x(xyxy, im0, label=label, color=colors[int(cls)])
                xmin = int(xyxy[0])
                ymin = int(xyxy[1])
                xmax = int(xyxy[2])
                ymax = int(xyxy[3])
                w = xmax - xmin # 233
                h = ymax - ymin # 602
                # 如果检测到的行人太小了，感觉意义也不大
                # 这里需要根据实际情况稍微设置下
                if h>2*w and h*w > 300*150:
                    print(h, w)
                    crop_img = im0[ymin:ymax, xmin:xmax] # HWC (602, 233, 3
                    query_index+=1
                    query_index = query_index % 5

                    cv2.imshow('cutted_img'.format(query_index), crop_img)
                    cv2.imwrite('query/9001_c9s1_00000{}_00.jpg'.format(query_index), crop_img)
                    time.sleep(0.5)
    print('') 