import os
import random
import glob
import cv2
import argparse
from yolov5 import Yolov5
from utils import draw_results

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.rectangle(img, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        #cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [0, 255, 255],
            #thickness=tf,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
def main():
    # setup processor and visualizer
    yolov5 = Yolov5(model=args.model)
    img_list = glob.glob(os.path.join(args.image_path, '*.jpg'))
    for i in range(100):
        for img_path in img_list:
            img = cv2.imread(img_path)
            output = yolov5.detect(img)
            # final results
            boxes, confs, classes = yolov5.post_process(output, conf_thres=0.3, iou_thres=0.5, origin_w=img.shape[1], origin_h=img.shape[0])
            #draw_results(img, boxes, confs, classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='weights/yolov5s_int8.trt', help='tensorrt engine file', required=False)
    parser.add_argument('--image_path', default='images', help='image file path', required=False)
    args = parser.parse_args()
    main()
