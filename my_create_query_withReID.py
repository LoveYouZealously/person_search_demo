import time

from models import *
from utils import torch_utils
from utils.datasets import *

from reid.data import make_data_loader
from reid.data.transforms import build_transforms
from reid.modeling import build_model
from reid.config import cfg as reidCfg


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

############# 行人重识别模型初始化 #############
query_loader, num_query = make_data_loader(reidCfg)
reidModel = build_model(reidCfg, num_classes=10126)
reidModel.load_param(reidCfg.TEST.WEIGHT)
reidModel.to(device).eval()

query_feats = []
query_pids  = []

for i, batch in enumerate(query_loader):
    with torch.no_grad():
        img, pid, camid = batch
        img = img.to(device)
        feat = reidModel(img)         # 一共2张待查询图片，每张图片特征向量2048 torch.Size([2, 2048])
        query_feats.append(feat)
        query_pids.extend(np.asarray(pid))  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。

query_feats = torch.cat(query_feats, dim=0)  # torch.Size([2, 2048])
print("The query feature is normalized")
query_feats = torch.nn.functional.normalize(query_feats, dim=1, p=2) # 计算出查询图片的特征向量

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
                    crop_img = im0[ymin:ymax, xmin:xmax] # HWC (602, 233, 3)
                    crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                    crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                    gallery_img.append(crop_img)


                if gallery_img:
                    gallery_img = torch.cat(gallery_img, dim=0)  # torch.Size([7, 3, 256, 128])
                    gallery_img = gallery_img.to(device)
                    gallery_feats = reidModel(gallery_img) # torch.Size([7, 2048])
                    print("The gallery feature is normalized")
                    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量

                    # m: 2
                    # n: 7
                    m, n = query_feats.shape[0], gallery_feats.shape[0]
                    distmat = torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                            torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
                    # out=(beta∗M)+(alpha∗mat1@mat2)
                    # qf^2 + gf^2 - 2 * qf@gf.t()
                    # distmat - 2 * qf@gf.t()
                    # distmat: qf^2 + gf^2
                    # qf: torch.Size([2, 2048])
                    # gf: torch.Size([7, 2048])
                    distmat.addmm_(1, -2, query_feats, gallery_feats.t())
                    # distmat = (qf - gf)^2
                    # distmat = np.array([[1.79536, 2.00926, 0.52790, 1.98851, 2.15138, 1.75929, 1.99410],
                    #                     [1.78843, 1.96036, 0.53674, 1.98929, 1.99490, 1.84878, 1.98575]])
                    distmat = distmat.cpu().numpy()  # <class 'tuple'>: (3, 12)
                    distmat = distmat.sum(axis=0) / len(query_feats) # 平均一下query中同一行人的多个结果
                    index = distmat.argmin()
                    if distmat[index] < dist_thres:
                        print('距离：%s'%distmat[index])
                        plot_one_box(gallery_loc[index], im0, label='find!', color=colors[int(cls)])
                        # cv2.imshow('person search', im0)
                        # cv2.waitKey()


                        query_index+=1
                        query_index = query_index % 5

                        cv2.imshow('9001_c9s1_00000{}_00.jpg'.format(query_index), crop_img)
                        cv2.imwrite('query/9001_c9s1_00000{}_00.jpg'.format(query_index), crop_img)
                        time.sleep(0.5)
    print('') 