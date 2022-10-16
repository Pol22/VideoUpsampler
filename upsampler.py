import cv2
import argparse
import ffmpeg
import numpy as np
import onnxruntime as ort

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Process


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def copy_audio(in_file, out_file):
    input_video = ffmpeg.input(in_file)
    output_video = ffmpeg.input(out_file)
    audio = input_video.audio
    video = output_video.video
    res = ''.join(out_file.split('.')[:-1]) + '_audio.mp4'
    stream = ffmpeg.output(audio, video, res, vcodec='copy', acodec='copy')
    ffmpeg.run(stream)


def to_onnx_model(model, inputs_shape, in_size, out_size, onnx_model_path):
    import tensorflow as tf
    import tf2onnx
    import onnx
    from tf_model import ResizeWrapper

    model = tf.keras.models.load_model(model, compile=False)
    model = ResizeWrapper(model, in_size, out_size)

    input_signature = [tf.TensorSpec(inputs_shape, tf.float32, name='input')]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)
    onnx.save(onnx_model, onnx_model_path)


def upsample(file_path, model, scale, res_height):
    # Required settings
    divisor = 8 # based on downsample layers on model
    in_channels = 9 # [previous, frame, next] * 3

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    res_width = int(np.round(width * res_height / height))

    print(f'FPS {fps}')
    print(f'Count of frames {frames}')
    print(f'Original size {width}x{height}')
    print(f'Result size {res_width}x{res_height}')

    file_path = Path(file_path)
    file_name = file_path.name
    res_name = file_name.split('.')
    res_name[-1] = '_upsampled.mp4'
    res_name = ''.join(res_name)
    res_path = str(file_path.parent / res_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    res_writer = cv2.VideoWriter(
        res_path, fourcc, fps, (res_width, res_height))

    inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    inst.setUseSpatialPropagation(False)
    _, prev = cap.read()
    _, frame = cap.read()

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nxt = None
    flow = None

    # Write first frame
    prev_scaled = cv2.resize(
        prev, (res_width, res_height), interpolation=cv2.INTER_CUBIC)
    res_writer.write(prev_scaled)

    inputs_shape = (1,
                    height + (divisor - height % divisor),
                    width + (divisor - width % divisor),
                    in_channels)

    # Convert to onnx model
    tmp_onnx = f'tmp_in_{height}_{width}_out_{res_height}_{res_width}.onnx'
    in_size = (height * scale, width * scale)
    out_size = (res_height, res_width)
    p = Process(
        target=to_onnx_model,
        args=(model, inputs_shape, in_size, out_size, tmp_onnx))
    p.start()
    p.join()

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        tmp_onnx,
        sess_options=so,
        providers=[
            ('TensorrtExecutionProvider',
             {'trt_fp16_enable': True,
              'trt_engine_cache_enable': True,
              'trt_engine_cache_path': './engine'}),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'])

    for _ in tqdm(range(2, frames)):
        _, nxt = cap.read()

        nxt_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)

        flow = inst.calc(prev_gray, frame_gray, None)
        warpped_prev = warp_flow(prev, flow)
        flow = inst.calc(nxt_gray, frame_gray, None)
        warpped_nxt = warp_flow(nxt, flow)

        # Preprocess model input
        inputs = np.concatenate([warpped_prev, frame, warpped_nxt], axis=2)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = inputs.astype(np.float32) / 255.0
        # Add zeros to required shape
        zeros = np.zeros(inputs_shape, dtype=np.float32)
        zeros[:, :height, :width, :] = inputs
        inputs = zeros

        pred = session.run(None, {'input': inputs})[0]
        res_writer.write(pred)

        prev = frame
        prev_gray = frame_gray
        frame = nxt
        frame_gray = nxt_gray

    # Write last frame
    frame_scaled = cv2.resize(
        frame, (res_width, res_height), interpolation=cv2.INTER_CUBIC)
    res_writer.write(frame_scaled)

    # When everything done, release the video capture object
    cap.release()
    res_writer.release()

    copy_audio(file_path, res_path)


def main():
    parser = argparse.ArgumentParser(description='MP4 Upsampler')
    parser.add_argument('--file', help='MP4 Video file', required=True)
    parser.add_argument('--model', help='H5 TF upsample model', required=True)
    parser.add_argument('--scale', help='Model scale', type=int, default=2)
    parser.add_argument('--format', help='Result video format', type=int,
                        default=1080)
    args = parser.parse_args()

    upsample(args.file, args.model, args.scale, args.format)


if __name__ == '__main__':
    main()
