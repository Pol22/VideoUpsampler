import tensorflow as tf
# import tensorflow_addons as tfa
from PIL import Image
import numpy as np


img = Image.open('frames/95.png')
img = np.asarray(img, dtype=np.float32)

shape = tf.shape(img)
print(f'shape {shape}')
h = shape[0]
w = shape[1]
c = shape[2]
r0 = 0.1
r1 = 0.45
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
coord = tf.stack([x, y])

# angle according to midpoint circle algorithm
theta = tf.math.atan2(1, tf.math.sqrt(r1 ** 2 - 1))
# print(theta)
# print(np.pi/4)
theta_len = 2 * np.pi / theta
q_len = theta_len / 8
theta_len = tf.cast(tf.round(theta_len), tf.int32)
r_len = tf.shape(x)[0]
res = np.zeros((r_len, theta_len, c), dtype=np.float32)
print(res.shape)

img = tf.transpose(img, (1, 0, 2))

for i in range(theta_len):
    alpha = i * theta
    print(f'angle={alpha}')
    rot_mat = [[tf.cos(alpha), -tf.sin(alpha)],
               [tf.sin(alpha), tf.cos(alpha)]]
    updated_coord = tf.matmul(rot_mat, coord)
    updated_coord = tf.round(updated_coord)
    updated_coord = tf.cast(updated_coord, tf.int32)
    updated_coord = tf.transpose(updated_coord, (1, 0))
    # print(updated_coord)
    updated_coord += [c_x, c_y]
    res[:, i, :] = tf.gather_nd(img, updated_coord)
    print(i)

# print(updated_coord[:, :10])

Image.fromarray(np.uint8(res)).save('tmp.png')