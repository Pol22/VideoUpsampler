import cv2
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def main():
    parser = argparse.ArgumentParser(description='MP4 Upsampler')
    parser.add_argument('--file', help='MP4 Video file', required=True)
    parser.add_argument('--model', help='H5 TF upsample model', required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f'FPS {fps}')
    print(f'Count of frames {frames}')
    print(f'Original size {height}x{width}')

    # Required settings
    scale = 2
    divisor = 4
    in_channels = 9

    file_path = Path(args.file)
    file_name = file_path.name
    res_name = file_name.split('.')
    res_name[-1] = '_upsampled.mp4'
    res_name = ''.join(res_name)
    res_path = file_path.parent / res_name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    res_writer = cv2.VideoWriter(
        str(res_path), fourcc, fps, (width * scale, height * scale))

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
        prev, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    res_writer.write(prev_scaled)

    model = tf.keras.models.load_model(args.model, compile=False)

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
        inputs = inputs.astype(np.float32)
        # Add zeros to required shape
        inputs_shape = (1,
                        height + (divisor - height % divisor),
                        width + (divisor - width % divisor),
                        in_channels)
        zeros = np.zeros(inputs_shape, dtype=np.float32)
        zeros[:, :height, :width, :] = inputs
        inputs = zeros
        inputs = inputs / 255.0
        
        pred = model.predict(inputs)
        # Select required shape
        pred = np.uint8(pred[0, :height * scale, :width * scale, :] * 255.0)

        res_writer.write(pred)

        prev = frame
        prev_gray = frame_gray
        frame = nxt
        frame_gray = nxt_gray

    # Write last frame
    frame_scaled = cv2.resize(
        frame, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    res_writer.write(frame_scaled)

    # When everything done, release the video capture object
    cap.release()
    res_writer.release()

if __name__ == '__main__':
    main()