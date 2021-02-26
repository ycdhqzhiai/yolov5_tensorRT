import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import ctypes
import glob
import cv2

class DataLoader:
    def __init__(self, data_path, batch_size):
        self.index = 0
        self.batch_size = batch_size
        self.img_list = glob.glob(os.path.join(data_path, "*.jpg"))
        self.width = 640
        self.height = 640
        self.calibration_data = np.zeros((self.batch_size,3,self.width,self.height), dtype=np.float32)
        
    def reset(self):
        self.index = 0

    def next_batch(self, index):
        print(self.calibration_data.shape)
        for i in range(self.batch_size):
            assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
            img = cv2.imread(self.img_list[i + self.index * self.batch_size])
            img = self.preprocess_v1(img)
            self.calibration_data[i] = img
            # example only
        return np.ascontiguousarray(self.calibration_data, dtype=np.float32)

    def preprocess_v1(self, image_raw):
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.width / w
        r_h = self.height / h
        if r_h > r_w:
            tw = self.width
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.height - th) / 2)
            ty2 = self.height - th - ty1
        else:
            tw = int(r_h * w)
            th = self.height
            tx1 = int((self.width - tw) / 2)
            tx2 = self.width - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        #image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        #image = np.ascontiguousarray(image)
        return image

class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_path, batch_size, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data = DataLoader(data_path, batch_size)
        self.device_input = cuda.mem_alloc(self.data.calibration_data.nbytes)
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):

        if self.current_index + self.batch_size > len(self.data.img_list):
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        batch = self.data.next_batch(self.current_index)
        self.current_index += 1

        cuda.memcpy_htod(self.device_input, batch)

        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)