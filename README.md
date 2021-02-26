# yolov5_tensorRT

## 1.Export tensorrt model
```shell
git clone https://github.com/ycdhqzhiai/yolov5_tensorRT
cd yolov5_tensorRT
pip install -r requirements.txt
```
复制`export.py`到原始[yolov5](https://github.com/ultralytics/yolov5)仓库`models`目录下</br>
`python models/export.py` 生成yolov5s.onnx,拷贝到weights目录下

## 2.run demo
`python main.py`

## 3.tensorrt int8 convert
```
python  convert_int8.py --onnx_file  weights/yolov5s.onnx  --data_path $your coustom datasets
```
