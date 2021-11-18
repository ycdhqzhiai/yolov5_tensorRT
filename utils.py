import math
import numpy as np
import cv2
import random
import colorsys
coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

def gen_colors(classes):
    """
        generate unique hues for each class and convert to bgr
        classes -- list -- class names (80 for coco dataset)
        -> list
    """
    hsvs = []
    for x in range(len(classes)):
        hsvs.append([float(x) / len(classes), 1., 0.7])
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = []
    for hsv in hsvs:
        h, s, v = hsv
        rgb = colorsys.hsv_to_rgb(h, s, v)
        rgbs.append(rgb)

    bgrs = []
    for rgb in rgbs:
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        bgrs.append(bgr)
    return bgrs

color_list = gen_colors(coco)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_v(array):
    return np.reciprocal(np.exp(-array) + 1.0)

def make_grid(nx, ny):
    """
    Create scaling tensor based on box location
    Source: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
    Arguments
        nx: x-axis num boxes
        ny: y-axis num boxes
    Returns
        grid: tensor of shape (1, 1, nx, ny, 80)
    """
    nx_vec = np.arange(nx)
    ny_vec = np.arange(ny)
    yv, xv = np.meshgrid(ny_vec, nx_vec)
    grid = np.stack((yv, xv), axis=2)
    grid = grid.reshape(1, 1, ny, nx, 2)
    return grid

def xywh2xyxy(x, origin_w=0, origin_h=0, INPUT_W=640, INPUT_H=640):
    """
    description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    param:
        origin_h:   height of original image
        origin_w:   width of original image
        x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
    return:
        y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
    """
    y = np.zeros_like(x)
    r_w = INPUT_W / origin_w
    r_h = INPUT_H / origin_h
    if r_h > r_w:
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
        y /= r_w
    else:
        y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        y /= r_h

    return y


def non_max_suppression(boxes, confs, classes, iou_thres=0.6):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    order = confs.flatten().argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    boxes = boxes[keep]
    confs = confs[keep]
    classes = classes[keep]
    return boxes, confs, classes

def nms(pred, iou_thres=0.6, origin_w=0, origin_h=0):
    boxes = xywh2xyxy(pred[..., 0:4], origin_w, origin_h)
    # best class only
    confs = np.amax(pred[:, 5:], 1, keepdims=True)
    classes = np.argmax(pred[:, 5:], axis=-1)
    return non_max_suppression(boxes, confs, classes, iou_thres)


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img

def GiB(val):
    return val * 1 << 30

def draw_results(img, boxes, confs, classes):
    window_name = 'final results'
    cv2.namedWindow(window_name)
    overlay = img.copy()
    final = img.copy()
    for box, conf, cls in zip(boxes, confs, classes):
        # draw rectangle
        x1, y1, x2, y2 = box
        x1, y1, x2, y2=(int)(x1),(int)(y1),(int)(x2),(int)(y2)
        conf = conf[0]
        cls_name = coco[cls]
        color = color_list[cls]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=1, lineType=cv2.LINE_AA)
        # draw text
        cv2.putText(overlay, '%s %f' % (cls_name, conf), org=(x1, int(y1-10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=color)
    # cv2.addWeighted(overlay, 0.5, final, 1 - 0.5, 0, final)
    cv2.imshow(window_name, overlay)
    cv2.waitKey(0)
