import os
import cv2
import argparse
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from math import ceil

from ABME.utils import warp
from ABME.model import SBMENet, ABMRNet, SynthesisNet


def main():
    parser = argparse.ArgumentParser(description='Frame interpolation')
    # parser.add_argument('--video', help='Input MP4 Video file', required=True)
    # parser.add_argument('--out_video', help='Output MP4 Video file', required=True)
    args = parser.parse_args()
    args.DDP = False

    cap = cv2.VideoCapture('Top Gear - 1x01 -.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f'FPS {fps}')
    print(f'Count of frames {frames}')
    print(f'Frame size {H}x{W}')

    # 4K video requires GPU memory of more than 24GB. We recommend crop it 
    # into 4 regions with some margin.
    if H < 512:
        divisor = 64.
        D_factor = 1.
    else:
        divisor = 128.
        D_factor = 0.5

    H_ = int(ceil(H / divisor) * divisor * D_factor)
    W_ = int(ceil(W / divisor) * divisor * D_factor)   

    SBMNet = SBMENet()
    ABMNet = ABMRNet()
    SynNet = SynthesisNet(args)

    SBMNet.load_state_dict(torch.load('ABME/Best/SBME_ckpt.pth', map_location='cpu'))
    ABMNet.load_state_dict(torch.load('ABME/Best/ABMR_ckpt.pth', map_location='cpu'))
    SynNet.load_state_dict(torch.load('ABME/Best/SynNet_ckpt.pth', map_location='cpu'))

    for param in SBMNet.parameters():
        param.requires_grad = False 
    for param in ABMNet.parameters():
        param.requires_grad = False
    for param in SynNet.parameters():
        param.requires_grad = False
    
    SBMNet.cuda()
    ABMNet.cuda()
    SynNet.cuda()

    _, frame1 = cap.read()

    for i in range(frames-1):
        _, frame3 = cap.read()

        if i < 1000: # TODO
            continue

        with torch.no_grad():
            frame1_ = F.interpolate(frame1, (H_, W_), mode='bicubic')
            frame3_ = F.interpolate(frame3, (H_, W_), mode='bicubic')

            SBM = SBMNet(torch.cat((frame1_, frame3_), dim=1))[0]
            SBM_= F.interpolate(SBM, scale_factor=4, mode='bilinear') * 20.0

            frame2_1, Mask2_1 = warp(frame1_, SBM_ * (-1),  return_mask=True)
            frame2_3, Mask2_3 = warp(frame3_, SBM_       ,  return_mask=True)

            frame2_Anchor_ = (frame2_1 + frame2_3) / 2
            frame2_Anchor = frame2_Anchor_ + 0.5 * (frame2_3 * (1-Mask2_1) + frame2_1 * (1-Mask2_3))

            Z  = F.l1_loss(frame2_3, frame2_1, reduction='none').mean(1, True)
            Z_ = F.interpolate(Z, scale_factor=0.25, mode='bilinear') * (-20.0)
            
            ABM_bw, _ = ABMNet(torch.cat((frame2_Anchor, frame1_), dim=1), SBM*(-1), Z_.exp())
            ABM_fw, _ = ABMNet(torch.cat((frame2_Anchor, frame3_), dim=1), SBM, Z_.exp())

            SBM_     = F.interpolate(SBM, (H, W), mode='bilinear')   * 20.0
            ABM_fw   = F.interpolate(ABM_fw, (H, W), mode='bilinear') * 20.0
            ABM_bw   = F.interpolate(ABM_bw, (H, W), mode='bilinear') * 20.0

            SBM_[:, 0, :, :] *= W / float(W_)
            SBM_[:, 1, :, :] *= H / float(H_)
            ABM_fw[:, 0, :, :] *= W / float(W_)
            ABM_fw[:, 1, :, :] *= H / float(H_)
            ABM_bw[:, 0, :, :] *= W / float(W_)
            ABM_bw[:, 1, :, :] *= H / float(H_)

            divisor = 8.
            H_ = int(ceil(H / divisor) * divisor)
            W_ = int(ceil(W / divisor) * divisor)
            
            Syn_inputs = torch.cat((frame1, frame3, SBM_, ABM_fw, ABM_bw), dim=1)
            
            Syn_inputs = F.interpolate(Syn_inputs, (H_,W_), mode='bilinear')
            Syn_inputs[:, 6, :, :] *= float(W_) / W
            Syn_inputs[:, 7, :, :] *= float(H_) / H
            Syn_inputs[:, 8, :, :] *= float(W_) / W
            Syn_inputs[:, 9, :, :] *= float(H_) / H
            Syn_inputs[:, 10, :, :] *= float(W_) / W
            Syn_inputs[:, 11, :, :] *= float(H_) / H 

            result = SynNet(Syn_inputs)
            
            result = F.interpolate(result, (H,W), mode='bicubic')

            cv2.imwrite(f'frame_{i}r.png', result)

        cv2.imwrite(f'frame_{i}.png', frame1) # TODO
        frame1 = frame3

        if i == 1001: # TODO
            break


if __name__ == '__main__':
    main()
