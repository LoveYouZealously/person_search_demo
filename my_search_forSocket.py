import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *

from reid.data import make_data_loader
from reid.data.transforms import build_transforms
from reid.modeling import build_model
from reid.config import cfg as reidCfg
reidCfg.DATASETS.ROOT_DIR = 'query'

query_time_last = 0
query_time_now = 0
query_feats = []


def search_init():
    cfg='cfg/yolov3.cfg' # 模型配置文件路径
    data = 'data/coco.data'
    weights = 'weights/yolo/yolov3.weights'
    images='data/samples'
    img_size=416
    half=False

    # Initialize
    device = torch_utils.select_device(force_cpu=False)
    torch.backends.cudnn.benchmark = False  # set False for reproducible results

    ############# 行人重识别模型初始化 #############
    reidModel = build_model(reidCfg, num_classes=10126)
    reidModel.load_param(reidCfg.TEST.WEIGHT)
    reidModel.to(device).eval()

    ############# 行人检测模型初始化 #############
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()
    # Half precision
    if half and device.type != 'cpu':  # half precision only supported on CUDA
        model.half()

    # Set Dataloader
    # dataloader = LoadWebcam(pipe=1, img_size=img_size, half=half)
    # dataloader = LoadImages(images, img_size=img_size, half=half)
    dataloader = None

    # Get classes and colors
    # parse_data_cfg(data)['names']:得到类别名称文件路径 names=data/coco.names
    classes = load_classes(parse_data_cfg(data)['names']) # 得到类别名列表: ['person', 'bicycle'...]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))] # 对于每种类别随机使用一种颜色画框

    return dataloader, model, reidModel, device, classes, colors, weights


def search_detect(dataloader_item, model, reidModel, device, classes, colors, weights):
    global query_time_last
    global query_time_now
    global query_feats

    conf_thres=0.1
    nms_thres=0.4
    dist_thres=3.0
    output='output'
    fourcc='mp4v'

    t = time.time()
    path, img, im0, vid_cap = dataloader_item
    # print(path, img.shape, im0.shape, vid_cap)
    # print(aaa)
    # data/samples/c1s1_001051.jpg (3, 320, 416) (480, 640, 3) None
    vid_path, vid_writer = None, None

    ############# query初始化 #############
    if len(os.listdir('query'))<1:
        print('not enough query')
        return
    else:
        if query_time_now==query_time_last or query_time_now-query_time_last>=1:
            query_loader, num_query = make_data_loader(reidCfg)
            
            query_feats = []
            for i, batch in enumerate(query_loader):
                with torch.no_grad():
                    img_q, pid, camid = batch
                    img_q = img_q.to(device)
                    feat = reidModel(img_q)         # 一共2张待查询图片，每张图片特征向量2048 torch.Size([2, 2048])
                    query_feats.append(feat)

            query_feats = torch.cat(query_feats, dim=0)  # torch.Size([2, 2048])
            query_feats = torch.nn.functional.normalize(query_feats, dim=1, p=2) # 计算出查询图片的特征向量
            print("The query feature is normalized")

            query_time_last = query_time_now
        elif len(query_feats)==0:
            print('no query_feats')
            return
    query_time_now = time.time()
    ############# query初始化 END #############

    if not os.path.exists(output):
        os.makedirs(output)
    save_path = str(Path(output) / Path(path).name) # 保存的路径

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
        count = 0
        gallery_img = []
        gallery_loc = []
        for *xyxy, conf, cls_conf, cls in det: # 对于最后的预测框进行遍历
            # *xyxy: 对于原图来说的左上角右下角坐标: [tensor(349.), tensor(26.), tensor(468.), tensor(341.)]
            '''Write to file'''
            # with open(save_path + '.txt', 'a') as file:
                # file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

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
                # if h>2*w and h*w > 100*50:
                if h > 100 and w > 50:
                    gallery_loc.append((xmin, ymin, xmax, ymax))
                    crop_img = im0[ymin:ymax, xmin:xmax] # HWC (602, 233, 3)
                    crop_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))  # PIL: (233, 602)
                    crop_img = build_transforms(reidCfg)(crop_img).unsqueeze(0)  # torch.Size([1, 3, 256, 128])
                    gallery_img.append(crop_img)


        '''flip image and box'''
        im0 = cv2.flip(im0, 1)

        if gallery_img:
            gallery_img = torch.cat(gallery_img, dim=0)  # torch.Size([7, 3, 256, 128])
            gallery_img = gallery_img.to(device)
            gallery_feats = reidModel(gallery_img) # torch.Size([7, 2048])
            gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1, p=2)  # 计算出查询图片的特征向量
            print("The gallery feature is normalized")

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
                
                # print(gallery_loc[index])
                xmin = im0.shape[1] - gallery_loc[index][2]
                ymin = im0.shape[0] - gallery_loc[index][3]
                xmax = im0.shape[1] - gallery_loc[index][0]
                ymax = im0.shape[0] - gallery_loc[index][1]

                # plot_one_box(gallery_loc[index], im0, label='find!', color=colors[int(cls)])
                plot_one_box((xmin, ymin, xmax, ymax), im0, label='find!', color=colors[int(cls)])
                # cv2.imshow('person search', im0)
                # cv2.waitKey()

    print('Done. (%.3fs)' % (time.time() - t))


    '''show image'''
    # cv2.imshow(weights, im0)

    '''save image'''
    # cv2.imwrite(save_path, im0)

    '''save webcam'''
#     if vid_path != save_path:  # new video
#         vid_path = save_path
#         if isinstance(vid_writer, cv2.VideoWriter):
#             vid_writer.release()  # release previous video writer

#         fps = vid_cap.get(cv2.CAP_PROP_FPS)
#         width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
#     vid_writer.write(im0)

    return im0



def main():
    with torch.no_grad():
        dataloader, model, reidModel, device, classes, colors, weights = search_init()
            
        # Run inference
        global query_time_last
        global query_time_now
        global query_feats
        query_time_last = time.time()
        query_time_now = query_time_last
        query_feats = []

        t0 = time.time()
        iii = 0
        for i, dataloader_item in enumerate(dataloader):
            search_detect(dataloader_item, model, reidModel, device, classes, colors, weights)
            iii+=1
            print(iii)

        print('All Done. (%.3fs)' % (time.time() - t0))
        print(iii)

if __name__ == '__main__':
    main()
