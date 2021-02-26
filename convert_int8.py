import argparse
import numpy as np
import torch
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import torch.nn as nn
import glob,os,cv2
from calibrator import Calibrator
from utils import GiB

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) # ** engine可视化 **

def build_int8_engine(onnx_file, calib, batch_size=32):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        if not os.path.exists(onnx_file):
            print('Onnx model not exists!!!')
            return None
        with open(onnx_file, 'rb') as model:
            parser.parse(model.read())
            assert network.num_layers > 0, 'Failed to parse ONNX model. Please check if the ONNX model is compatible '

        builder.max_batch_size = batch_size
        builder.max_workspace_size = GiB(1)
        builder.int8_mode = True
        builder.int8_calibrator = calib
        engine = builder.build_cuda_engine(network) 
    return engine

def main():
    calib = Calibrator(args.data_path, args.batch_size, args.calibration_table)
    engine_int8 = build_int8_engine(args.onnx_file, calib, args.batch_size)
    assert engine_int8, 'convert model filad'
    with open(args.engine_file, "wb") as f:
        f.write(engine_int8.serialize())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_file', default='weights/yolov5s.onnx', help='onnx file location')
    parser.add_argument('--calibration_table', default='yolov5s_calibration.cache', help='calibration cache')
    parser.add_argument('--engine_file', default='weights/yolov5s_int8.trt', help='name of trt output file')
    parser.add_argument('--data_path', default='/opt/sda5/BL01_Data/Object_Detect_Data/COCO/images/test2017', help='calibration images')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--height', type=int, default=640, help='model input h')
    parser.add_argument('--width', type=int, default=640, help='model input w')
    
    args = parser.parse_args()

    

    main()