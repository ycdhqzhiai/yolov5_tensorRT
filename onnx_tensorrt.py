import tensorrt as trt
import sys
import argparse

logger = trt.Logger(trt.Logger.WARNING)

def convert():
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # trt7
    with trt.Builder(logger) as builder, builder.create_network(explicit_batch) as network, trt.OnnxParser(network, logger) as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        if args.floatingpoint == 16:
            builder.fp16_mode = True
        with open(args.model, 'rb') as f:
            print('Beginning ONNX file parsing')
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print("ERROR", parser.get_error(error))
        print("num layers:", network.num_layers)
        network.get_input(0).shape = [1, 3, 640, 640]  # trt7
        # last_layer = network.get_layer(network.num_layers - 1)
        # network.mark_output(last_layer.get_output(0))
        # reshape input from 32 to 1
        engine = builder.build_cuda_engine(network)
        with open(args.output, 'wb') as f:
            f.write(engine.serialize())
        print("Completed creating Engine")
if __name__ == '__main__':
    desc = 'Compile Onnx model to TensorRT'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-m', '--model', default='yolov5s.onnx', help='onnx file location')
    parser.add_argument('-fp', '--floatingpoint', type=int, default=16, help='floating point precision. 16 or 32')
    parser.add_argument('-o', '--output', default='yolov5.trt', help='name of trt output file')
    args = parser.parse_args()
    convert()