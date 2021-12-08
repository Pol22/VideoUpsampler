import os
from shutil import move
from random import sample


def main():
    data_folder = 'frames'
    save_folder = 'test_data'
    split = 0.1

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    image_list = os.listdir(data_folder)

    frame_list = sorted(filter(lambda x: '_frame' in x, image_list))
    nxt_list = sorted(filter(lambda x: '_nxt' in x, image_list))
    prev_list = sorted(filter(lambda x: '_prev' in x, image_list))
    res_list = sorted(filter(lambda x: '_res' in x, image_list))

    selected = sample(range(len(frame_list)), k=int(split * len(frame_list)))
    for ind in selected:
        move(os.path.join(data_folder, frame_list[ind]),
             os.path.join(save_folder, frame_list[ind]))
        move(os.path.join(data_folder, nxt_list[ind]),
             os.path.join(save_folder, nxt_list[ind]))
        move(os.path.join(data_folder, prev_list[ind]),
             os.path.join(save_folder, prev_list[ind]))
        move(os.path.join(data_folder, res_list[ind]),
             os.path.join(save_folder, res_list[ind]))


if __name__ == '__main__':
    main()
