import argparse
import tensorrt as trt
import os

"""
tensorrt 10 ~
"""

def build_engine(onnx_path: str, trt_path: str,
                 fp16: bool, max_batch: int = 1,
                 input_h: int = 352, input_w: int = 352):

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        1 << 30  # 1 GiB
    )
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,
        min=(1, 3, input_h, input_w),
        opt=(1, 3, input_h, input_w),
        max=(max_batch, 3, input_h, input_w)
    )
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine:
        os.makedirs(os.path.dirname(trt_path), exist_ok=True)
        with open(trt_path, "wb") as out_file:
            out_file.write(serialized_engine)
        print(f"Serialized TensorRT engine saved to: {trt_path}")
    else:
        print("Engine build failed")
    return serialized_engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine (with dynamic shape support)")
    parser.add_argument("--onnx", type=str, default="weight/PRMWNet/PRMWNet.onnx", help="input onnx model path")
    parser.add_argument("--trt", type=str, default="weight/PRMWNet/PRMWNet.trt", help="output tensorRT engine path")
    parser.add_argument("--fp16", action="store_true", help="The precision mode to build in 'FP16'")
    parser.add_argument("--max_batch", type=int, default=1, help="max batch size for optimization profile")
    parser.add_argument("--input_h", type=int, default=256, help="model input height")
    parser.add_argument("--input_w", type=int, default=256, help="model input width")
    args = parser.parse_args()

    build_engine(args.onnx, args.trt, args.fp16,
                 max_batch=args.max_batch, input_h=args.input_h, input_w=args.input_w)


# python export_tensorrt10.py --onnx weight/PRMWNet/PRMWNet.onnx --trt weight/PRMWNet/PRMWNet.trt
# python export_tensorrt10.py --onnx weight/Melanoma/U-Net/U-Net.onnx --trt weight/Melanoma/U-Net/U-Net.trt
# python export_tensorrt10.py --onnx weight/Melanoma/U-Net_plusplus/U-Net_plusplus.onnx --trt weight/Melanoma/U-Net_plusplus/U-Net_plusplus.trt
# python export_tensorrt10.py --onnx weight/Melanoma/MSFNet/MSFNet.onnx --trt weight/Melanoma/MSFNet/MSFNet.trt
# python export_tensorrt10.py --onnx weight/Melanoma/DSU-Net/DS-UNet.onnx --trt weight/Melanoma/DSU-Net/DS-UNet.trt
# python export_tensorrt10.py --onnx weight/Melanoma/CTMANet/CTMANet.onnx --trt weight/Melanoma/CTMANet/CTMANet.trt


