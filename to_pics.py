import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def main():
    parser = argparse.ArgumentParser(description='Dataset creator')
    parser.add_argument('--video', help='MP4 Video file',
                        required=True)
    parser.add_argument('--dir', help='Output dir', default='frames')
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'FPS {fps}')
    print(f'frames {frames}')

    inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    inst.setUseSpatialPropagation(False)

    _, prev = cap.read()
    _, res_frame = cap.read()

    prev = cv2.resize(
        prev, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    frame = cv2.resize(
        res_frame, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nxt = None
    flow = None

    for i in tqdm(range(2, frames)):
        cv2.imwrite(os.path.join(args.dir, f'{i-1}_frame.png'), frame)
        cv2.imwrite(os.path.join(args.dir, f'{i-1}_res.png'), res_frame)

        _, res_frame = cap.read()
        nxt = cv2.resize(
            res_frame, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        nxt_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)

        flow = inst.calc(prev_gray, frame_gray, None)
        warpped_prev = warp_flow(prev, flow)
        flow = inst.calc(nxt_gray, frame_gray, None)
        warpped_nxt = warp_flow(nxt, flow)

        cv2.imwrite(os.path.join(args.dir, f'{i-1}_prev.png'), warpped_prev)
        cv2.imwrite(os.path.join(args.dir, f'{i-1}_nxt.png'), warpped_nxt)

        prev = frame
        prev_gray = frame_gray
        frame = nxt
        frame_gray = nxt_gray

    cap.release()


if __name__ == '__main__':
    main()
