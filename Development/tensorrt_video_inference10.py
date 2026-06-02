import argparse
import os
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
from typing import Tuple, List, Dict
from tqdm import tqdm

"""
tensorrt 10 ~
"""

def preprocess(frame: np.ndarray, input_size: Tuple[int,int]) -> np.ndarray:
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size[::-1], interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    chw = img.transpose(2, 0, 1)
    return np.expand_dims(chw, axis=0).copy()

def postprocess(logits: np.ndarray, orig_size: Tuple[int,int], threshold: float=0.5) -> np.ndarray:
    prob = logits.squeeze()
    prob = cv2.resize(prob, orig_size, interpolation=cv2.INTER_LINEAR)
    return (prob >= threshold).astype(np.uint8) * 255

def load_engine(trt_path: str) -> trt.ICudaEngine:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(trt_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine from {trt_path}")
    return engine

def allocate_buffers(engine: trt.ICudaEngine):

    inputs: List[Dict] = []
    outputs: List[Dict] = []
    bindings: List[int] = []
    stream = cuda.Stream()

    nb_tensors = engine.num_io_tensors

    for idx in range(nb_tensors):

        name = engine.get_tensor_name(idx)
        mode = engine.get_tensor_mode(name)

        dims = engine.get_tensor_shape(name)
        size = trt.volume(dims)
        dtype = engine.get_tensor_dtype(name)
        np_dtype = trt.nptype(dtype)

        host_mem   = cuda.pagelocked_empty(size, np_dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        buf = {
            'name':   name,
            'index':  idx,
            'shape':  tuple(dims),
            'host':   host_mem,
            'device': device_mem
        }


        if mode == trt.TensorIOMode.INPUT:
            inputs.append(buf)
        else:
            outputs.append(buf)

    return inputs, outputs, bindings, stream


def infer(context: trt.IExecutionContext,
          inputs: List[Dict],
          outputs: List[Dict],
          bindings: List[int],
          stream: cuda.Stream,
          inp: np.ndarray) -> np.ndarray:

    in_buf = inputs[0]
    np.copyto(in_buf['host'], inp.ravel())
    cuda.memcpy_htod_async(in_buf['device'], in_buf['host'], stream)

    context.execute_v2(bindings)

    out_buf = outputs[0]
    cuda.memcpy_dtoh_async(out_buf['host'], out_buf['device'], stream)
    stream.synchronize()

    out_shape = (1,) + tuple(out_buf['shape'][1:])
    return out_buf['host'].reshape(out_shape)


def overlay_and_show(frame: np.ndarray,
                     mask: np.ndarray,
                     fps: float,
                     color: Tuple[int,int,int]) -> np.ndarray:
    overlay = frame.copy()
    overlay[mask > 0] = color
    vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    txt = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    sz = cv2.getTextSize(txt, font, 0.5, 1)[0]
    x, y = frame.shape[1] - sz[0] - 10, sz[1] + 10
    cv2.putText(vis, txt, (x, y), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
    cv2.imshow("PraNet Segmentation ", vis)
    return vis


def main():
    parser = argparse.ArgumentParser(description="real-time tensorRT video segmentation")
    parser.add_argument("--trt", type=str, default="weight/PRMWNet/PRMWNet.trt", help="tensorRT engine file path")
    parser.add_argument("--input", type=str, default="data/PolypColon.mp4", help="input video path")
    parser.add_argument("--output", type=str, default="output_trt/output.mp4", help="output video save path")
    parser.add_argument("--input_size", type=int, nargs=2, default=[352,352], help="model input: height, width")
    parser.add_argument("--threshold",  type=float, default=0.5, help="binarization threshold")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"unable to open video: {args.input}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps_list: List[float] = []

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    engine  = load_engine(args.trt)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    start = time.time()
    with tqdm(total=frame_count, ncols=80, desc="Processing", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            inp = preprocess(frame, tuple(args.input_size))
            logits = infer(context, inputs, outputs, bindings, stream, inp)
            mask = postprocess(logits, (w, h), args.threshold)
            t1 = time.time()

            curr_fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0
            fps_list.append(curr_fps)
            vis = overlay_and_show(frame, mask, curr_fps, (255,255,0))

            writer.write(vis)
            pbar.set_postfix({"Time(ms)": f"{(t1-t0)*1000:.1f}"})
            pbar.update(1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    total = time.time() - start

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"Processed {len(fps_list)} frames in {total:.2f}s, average FPS: {avg_fps:.2f}, output: {args.output}")

if __name__ == "__main__":
    main()


