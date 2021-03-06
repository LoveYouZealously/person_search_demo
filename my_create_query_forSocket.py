import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

from models import *
from utils import torch_utils
from utils.datasets import *


query_index = 0


def query_init():
    cfg='cfg/yolov3.cfg'
    data='data/coco.data'
    weights='weights/yolo/yolov3.weights'
    img_size=416
    half=False
    device = torch_utils.select_device(force_cpu=False)


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
    # dataloader = LoadWebcam(img_size=img_size, half=half)
    # dataloader = LoadImages('data/samples', img_size=img_size, half=half)
    dataloader = None

    # Get classes and colors
    # parse_data_cfg(data)['names']:得到类别名称文件路径 names=data/coco.names
    classes = load_classes(parse_data_cfg(data)['names']) # 得到类别名列表: ['person', 'bicycle'...]
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    colors_hex_list = ['#000000', '#CCFF00', '#666699', '#FF9999', '#99CC66', '#FF0033', '#FFCC00', '#FF6600', '#CC3399', '#666633', '#FFFF66', '#99CC33', '#FF9900',
             '#FF99CC', '#993366', '#CCFF66', '#66CCCC', '#FFCC99', '#CCCC33', '#FF9966', '#FF6666']
    colors_rgb_list = []
    for hex_num in colors_hex_list:
        # print(hex_num)
        r_int = int('0x' + hex_num[1:3], 16)
        g_int = int('0x' + hex_num[3:5], 16)
        b_int = int('0x' + hex_num[5:7], 16)
        colors_rgb_list.append((r_int, g_int, b_int))

    return dataloader, device, model, classes, colors_rgb_list


def query_detect(dataloader_item, device, model, classes, colors_rgb_list):
    # cropImg_pid_camid_list = []

    global query_index
    conf_thres=0.5
    nms_thres=0.5
    query_dirpath = 'query'
    path, img, im0, vid_cap = dataloader_item

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
        im0_forDrawing = np.copy(im0)
        im0_forDrawing = cv2.flip(im0_forDrawing, 1)
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
                # if h>2*w and h*w > 100*50:
                if h > 100 and w > 50:
                    print(h, w)
                    crop_img = im0[ymin:ymax, xmin:xmax] # HWC (602, 233, 3)
                    crop_img = cv2.flip(crop_img, 1)

                    query_index += 1
                    print('query_index', query_index)
                    
                    xmin_f = im0_forDrawing.shape[1] - xmax
                    xmax_f = im0_forDrawing.shape[1] - xmin

                    cv2.imwrite(os.path.join(query_dirpath, '900{}_c9s1_000001_01.jpg'.format(query_index)), crop_img)
                    plot_one_box(
                        (xmin_f, ymin, xmax_f, ymax), im0_forDrawing,
                        label='{}'.format(query_index),
                        color=colors_rgb_list[int(query_index)]
                    )

        # cv2.imwrite(os.path.join(r'C:\Users\10156672\Desktop', '9001_c9s1_000001_01_raw.jpg'), im0_forDrawing)
        print('---------- query created ----------')
        return im0_forDrawing


if __name__=='__main__':
    dataloader, device, model, classes, colors = query_init()

    for i, dataloader_item in enumerate(dataloader):
        query_detect(dataloader_item, device, model, classes, colors)






