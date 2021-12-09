import cv2
import numpy as np
import tensorflow as tf


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

file_name = 'out.mp4'
cap = cv2.VideoCapture(file_name)
fps = cap.get(cv2.CAP_PROP_FPS)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f'FPS {fps}')
print(f'frames {frames}')
print(f'height {height}')
print(f'width {width}')

scale = 2
div = 8
in_c = 9

res_name = file_name.split('.')[0] + '_upsample.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
res_writer = cv2.VideoWriter(
    res_name, fourcc, fps, (width * scale, height * scale))

use_spatial_propagation = False
inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
inst.setUseSpatialPropagation(use_spatial_propagation)
_, prev = cap.read()
_, frame = cap.read()

prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
nxt = None
flow = None

model = tf.keras.models.load_model(
    'results/resunet_040-35.03.h5', compile=False)

for i in range(2, 100):
    # Capture frame-by-frame
    _, nxt = cap.read()

    nxt_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)

    flow = inst.calc(prev_gray, frame_gray, None)
    warpped_prev = warp_flow(prev, flow)
    flow = inst.calc(nxt_gray, frame_gray, None)
    warpped_nxt = warp_flow(nxt, flow)

    inputs = np.concatenate([warpped_prev, frame, warpped_nxt], axis=2)
    inputs = np.expand_dims(inputs, axis=0)
    inputs = inputs.astype(np.float32)
    zeros = np.zeros(
        (1, height + (div - height % div), width + (div - width % div), in_c),
        dtype=np.float32
    )
    zeros[:, :height, :width, :] = inputs
    inputs = zeros
    inputs = inputs / 255.0
    
    pred = model.predict(inputs)
    pred = np.uint8(pred[0, :height * scale, :width * scale, :] * 255.0)

    res_writer.write(pred)

    prev = frame
    prev_gray = frame_gray
    frame = nxt
    frame_gray = nxt_gray


# When everything done, release the video capture object
cap.release()
res_writer.release()
