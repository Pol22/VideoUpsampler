import cv2
import numpy as np


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


cap = cv2.VideoCapture('out.mp4')
res_cap = cv2.VideoCapture('Top Gear - 22x01 - 1080p.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f'FPS {fps}')
print(f'frames {frames}')

use_spatial_propagation = False
inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
inst.setUseSpatialPropagation(use_spatial_propagation)
_, prev = cap.read()
_, frame = cap.read()
_, _ = res_cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
nxt = None
flow = None
for i in range(2, 1000):
    # Capture frame-by-frame
    _, nxt = cap.read()
    _, res_frame = res_cap.read()

    nxt_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)

    flow = inst.calc(prev_gray, frame_gray, None)
    warpped_prev = warp_flow(prev, flow)
    flow = inst.calc(nxt_gray, frame_gray, None)
    warpped_nxt = warp_flow(nxt, flow)

    cv2.imwrite(f'frames/{i-1}_frame.png', frame)
    cv2.imwrite(f'frames/{i-1}_prev.png', warpped_prev)
    cv2.imwrite(f'frames/{i-1}_nxt.png', warpped_nxt)

    cv2.imwrite(f'frames/{i-1}_res.png', res_frame)

    prev = frame
    prev_gray = frame_gray
    frame = nxt
    frame_gray = nxt_gray


# When everything done, release the video capture object
cap.release()
res_cap.release()
