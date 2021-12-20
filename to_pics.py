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
    parser.add_argument('--lr_video', help='Low resolution MP4 Video file',
                        required=True)
    parser.add_argument('--hr_video', help='High resolution MP4 Video file',
                        required=True)
    parser.add_argument('--dir', help='Output dir', default='frames')
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    cap = cv2.VideoCapture(args.lr_video)
    res_cap = cv2.VideoCapture(args.hr_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    res_frames = int(res_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'FPS {fps}')
    print(f'Frames LR {frames}, HR {res_frames}')

    inst = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    inst.setUseSpatialPropagation(False)
    _, prev = cap.read()
    _, frame = cap.read()
    _, _ = res_cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nxt = None
    flow = None

    for i in tqdm(range(2, frames)):
        # Capture frame-by-frame
        _, nxt = cap.read()
        _, res_frame = res_cap.read()

        nxt_gray = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)

        flow = inst.calc(prev_gray, frame_gray, None)
        warpped_prev = warp_flow(prev, flow)
        flow = inst.calc(nxt_gray, frame_gray, None)
        warpped_nxt = warp_flow(nxt, flow)

        cv2.imwrite(os.path.join(args.dir, f'{i-1}_frame.png'), frame)
        cv2.imwrite(os.path.join(args.dir, f'{i-1}_prev.png'), warpped_prev)
        cv2.imwrite(os.path.join(args.dir, f'{i-1}_nxt.png'), warpped_nxt)

        cv2.imwrite(os.path.join(args.dir, f'{i-1}_res.png'), res_frame)

        prev = frame
        prev_gray = frame_gray
        frame = nxt
        frame_gray = nxt_gray

    # When everything done, release the video capture object
    cap.release()
    res_cap.release()


if __name__ == '__main__':
    main()
