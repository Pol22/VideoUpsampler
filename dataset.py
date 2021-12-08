import numpy as np
import random
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class TripletDataset(Dataset):
    def __init__(self, data_folder, crop_size=256, scale=2):
        self.crop_size = crop_size
        self.scale = scale

        image_list = os.listdir(data_folder)
        image_list = list(map(
            lambda x: os.path.join(data_folder, x), image_list))

        self.frame_list = sorted(filter(lambda x: '_frame' in x, image_list))
        self.nxt_list = sorted(filter(lambda x: '_nxt' in x, image_list))
        self.prev_list = sorted(filter(lambda x: '_prev' in x, image_list))
        self.res_list = sorted(filter(lambda x: '_res' in x, image_list))

        assert(len(self.frame_list) == len(self.nxt_list) == \
               len(self.prev_list) == len(self.res_list))

    def __getitem__(self, i):
        frame = Image.open(self.frame_list[i], mode='r')
        nxt = Image.open(self.nxt_list[i], mode='r')
        prev = Image.open(self.prev_list[i], mode='r')
        res = Image.open(self.res_list[i], mode='r')

        lr_crop_size = self.crop_size // self.scale
        left = random.randint(1, frame.width - lr_crop_size)
        top = random.randint(1, frame.height - lr_crop_size)
        right = left + lr_crop_size
        bottom = top + lr_crop_size

        frame = frame.crop((left, top, right, bottom))
        nxt = nxt.crop((left, top, right, bottom))
        prev = prev.crop((left, top, right, bottom))

        left = left * self.scale
        top = top * self.scale
        right = left + self.crop_size
        bottom = top + self.crop_size

        res = res.crop((left, top, right, bottom))

        frame = np.array(frame, dtype=np.float32)
        nxt = np.array(nxt, dtype=np.float32)
        prev = np.array(prev, dtype=np.float32)
        res = np.array(res, dtype=np.float32)

        lr_img = np.concatenate([prev, frame, nxt], axis=2) / 255.0
        hr_img = res / 255.0

        return to_tensor(lr_img), to_tensor(hr_img)

    def __len__(self):
        return len(self.frame_list)
