import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# img = Image.open('tmp_img.png')
# img = Image.open('frames/95.png')
img = Image.open('a2345-_DSC0114_0.png')
# img = Image.open('a2972-_DSC6416_0.png')
img = np.asarray(img, dtype=np.float32)


def rad_crop(tensor, r1: float, r2: float):
    assert(0.0 <= r1 and r1 < r2 and r2 <= 1.0)
    img_shape = tf.shape(tensor)
    h = img_shape[0]
    w = img_shape[1]
    c_x = tf.cast(w, tf.float32) / 2
    c_y = tf.cast(h, tf.float32) / 2
    x = tf.range(w, dtype=tf.float32)
    x = tf.reshape(x, (1, w)) + tf.zeros(shape=(h, w))
    y = tf.range(h, dtype=tf.float32)
    y = tf.reshape(y, (h, 1)) + tf.zeros(shape=(h, w))
    x -= c_x
    y -= c_y
    r = tf.math.sqrt(x ** 2 + y ** 2)
    r_max = tf.sqrt(c_x ** 2 + c_y ** 2)
    r /= r_max
    cond = tf.math.logical_and(r <= r2, r > r1)
    cond = tf.expand_dims(cond, axis=2)
    x = tf.where(cond, tensor, [0.0, 0.0, 0.0])
    return x


def select_ring(tensor, r0: float, r1: float):
    assert(0.0 <= r0 and r0 < r1 and r1 <= 1.0)
    shape = tf.shape(tensor)
    print(f'shape {shape}')
    h = shape[0]
    w = shape[1]
    c = shape[2]
    c_y = tf.cast(h, tf.float32) / 2
    c_x = tf.cast(w, tf.float32) / 2
    print(f'cx={c_x}, cy={c_y}')
    r_max = tf.math.sqrt(c_x ** 2 + c_y ** 2)
    print(f'r {r_max}')
    r0 = tf.round(r0 * r_max)
    r1 = tf.round(r1 * r_max)
    print(f'r0={r0}  r1={r1}')
    x = tf.range(r0, r1, delta=1.0, dtype=tf.float32)
    y = tf.zeros_like(x)
    coord = tf.stack([y, x])

    # pad with zeros
    if r1 > c_y or r1 > c_x:
        pad_h = r_max - c_y + 1
        pad_w = r_max - c_x + 1
        c_y += pad_h
        c_x += pad_w
        pad_h = tf.cast(tf.round(pad_h), dtype=tf.int32)
        pad_w = tf.cast(tf.round(pad_w), dtype=tf.int32)
        tensor = tf.pad(tensor, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))

    # angle according to midpoint circle algorithm
    theta = tf.math.atan2(1, tf.math.sqrt(r1 ** 2 - 1))
    print(f'theta={theta}')
    theta = tf.range(0, 2 * np.pi, theta, dtype=tf.float32)
    cos_theta = tf.cos(theta)
    sin_theta = tf.sin(theta)
    rot_mat = tf.stack(
        [tf.stack([cos_theta, sin_theta], axis=1),
         tf.stack([-sin_theta, cos_theta], axis=1)],
        axis=1
    )

    updated_coord = tf.matmul(rot_mat, coord)
    updated_coord = tf.transpose(updated_coord, (0, 2, 1))
    updated_coord = tf.reshape(updated_coord, (-1, 2))
    updated_coord += [c_y, c_x]
    updated_coord = tf.round(updated_coord)
    updated_coord = tf.cast(updated_coord, tf.int32)

    res = tf.gather_nd(tensor, updated_coord)
    res = tf.reshape(res, (tf.shape(theta)[0], -1, c))
    res = tf.transpose(res, (1, 0, 2))
    print(res.shape)

    return res


def inv_ring_warp(tensor, r0: float, r1: float):
    assert(0.0 <= r0 and r0 < r1 and r1 <= 1.0)
    shape = tf.shape(tensor)
    print(f'shape {shape}')
    r_len = shape[0]
    theta_len = shape[1]
    c = shape[2]

    r_max = tf.cast(r_len, tf.float32) / (r1 - r0)
    r0 = tf.round(r0 * r_max)
    r1 = r0 + tf.cast(r_len, tf.float32)
    print(f'r {r_max}')
    print(f'r0={r0}  r1={r1}')

    pad_0 = tf.cast(r0, tf.int32)
    diag = np.sqrt(2) * r1
    pad_1 = tf.cast(diag - r1 + 1, tf.int32)
    tensor = tf.pad(tensor, ((pad_0, pad_1), (0, 0), (0, 0)))
    print(f'tshape {tensor.shape}')

    x = tf.range(r1-1, -r1, -1.0, dtype=tf.float32)
    y = tf.range(r1-1, -r1, -1.0, dtype=tf.float32)
    h = w = tf.shape(x)[0]
    x = tf.reshape(x, (1, w))
    y = tf.reshape(y, (h, 1))
    r = tf.math.sqrt(x ** 2 + y ** 2)
    print(f'r_shape {r.shape}')
    theta = tf.math.atan2(y, x) + np.pi
    theta_one = 2 * np.pi / tf.cast(theta_len, tf.float32)
    theta /= theta_one
    theta -= 1.0 # ????
    r = tf.reshape(r, (h * w))
    print('r max', tf.reduce_max(r), 'r min', tf.reduce_min(r))
    theta = tf.reshape(theta, (h * w))
    print('theta max', tf.reduce_max(theta), 'theta min', tf.reduce_min(theta))

    coord = tf.stack([r, theta], axis=1)
    print(f'coord {coord.shape}')
    coord = tf.round(coord)
    coord = tf.cast(coord, tf.int32)

    res = tf.gather_nd(tensor, coord)
    res = tf.reshape(res, (h, w, c))

    return res


def warp_ring(tensor, r0: float, r1: float):
    assert(0.0 <= r0 and r0 < r1 and r1 <= 1.0)
    shape = tf.shape(tensor)
    print(f'shape {shape}')
    h = shape[0]
    w = shape[1]
    c = shape[2]
    c_y = tf.cast(h, tf.float32) / 2
    c_x = tf.cast(w, tf.float32) / 2
    print(f'cx={c_x}, cy={c_y}')
    r_max = tf.math.sqrt(c_x ** 2 + c_y ** 2)
    print(f'r {r_max}')
    r0 = tf.round(r0 * r_max)
    r1 = tf.round(r1 * r_max)

    # angle according to midpoint circle algorithm
    angle = tf.math.atan2(1, tf.math.sqrt(r_max ** 2 - 1))
    angle_len = tf.cast(tf.round(2 * np.pi / angle), tf.int32)
    r_len = tf.cast(tf.round(r_max), tf.int32)

    flags = cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
    dsize = tf.stack([r_len, angle_len])
    center = tf.stack([c_x, c_y])

    # print([tensor, dsize, center, r_max, flags])
    # print(dsize.numpy())
    def warper(tensor, dsize, center, r_max, flags):
        dsize = tuple(dsize)
        center = tuple(center)
        res = cv2.warpPolar(tensor, dsize, center, r_max, flags)
        # TODO remove
        # res = cv2.rotate(res, cv2.ROTATE_90_CLOCKWISE)
        # res = cv2.flip(res, 1)
        return res
    
    # res = cv2.warpPolar(tensor.numpy(), np.array(dsize), center, r_max, flags)
    res = tf.numpy_function(
        warper,
        [tensor, dsize, center, r_max, flags],
        tf.float32
    )
    print(tf.shape(res))

    return res



r0 = 0.0
r1 = 1.0
x = select_ring(img, r0, r1)
Image.fromarray(np.uint8(x)).save('tmp_ring.png')


img = tf.convert_to_tensor(img, dtype=tf.float32)
x = warp_ring(img, r0, r1)

Image.fromarray(np.uint8(x)).save('tmp_ring1.png')

# y = inv_ring_warp(x, r0, r1)
# Image.fromarray(np.uint8(y)).save('tmp_ring_inv.png')
