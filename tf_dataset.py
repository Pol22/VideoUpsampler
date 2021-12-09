import os
import tensorflow as tf


def get_dataset(data_folder, batch_size=16, crop_size=128,
                scale=2, repeat_count=1):
    image_list = os.listdir(data_folder)
    image_list = list(map(
        lambda x: os.path.join(data_folder, x), image_list))

    frame_list = sorted(filter(lambda x: '_frame' in x, image_list))
    nxt_list = sorted(filter(lambda x: '_nxt' in x, image_list))
    prev_list = sorted(filter(lambda x: '_prev' in x, image_list))
    res_list = sorted(filter(lambda x: '_res' in x, image_list))

    assert(len(frame_list) == len(nxt_list) == \
            len(prev_list) == len(res_list))

    frame_ds = images_dataset(frame_list)
    nxt_ds = images_dataset(nxt_list)
    prev_ds = images_dataset(prev_list)
    res_ds = images_dataset(res_list)

    ds = tf.data.Dataset.zip((prev_ds, frame_ds, nxt_ds, res_ds))
    ds = ds.map(get_random_cropper(crop_size, scale),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.repeat(repeat_count)
    ds = ds.batch(batch_size)

    return ds


def images_dataset(image_files):
    ds = tf.data.Dataset.from_tensor_slices(image_files)
    ds = ds.map(tf.io.read_file, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: tf.image.decode_png(x, channels=3),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def get_random_cropper(crop_size, scale):
    def cropper(prev, frame, nxt, res):
        img_shape = tf.shape(frame)[:2]

        w = tf.random.uniform(
            shape=(), maxval=img_shape[1] - crop_size + 1, dtype=tf.int32)
        h = tf.random.uniform(
            shape=(), maxval=img_shape[0] - crop_size + 1, dtype=tf.int32)
        scaled_w = w * scale
        scaled_h = h * scale
        scaled_crop = crop_size * scale

        frame_cropped = frame[h:h + crop_size, w:w + crop_size, :]
        prev_cropped = prev[h:h + crop_size, w:w + crop_size, :]
        nxt_cropped = nxt[h:h + crop_size, w:w + crop_size, :]
        res_cropped = res[scaled_h:scaled_h + scaled_crop,
                            scaled_w:scaled_w + scaled_crop, :]
        input_cropped = tf.concat(
            [prev_cropped, frame_cropped, nxt_cropped], axis=2)

        return input_cropped, res_cropped

    return cropper


def normalize(inputs, res):
    inputs = tf.cast(inputs, tf.float32)
    res = tf.cast(res, tf.float32)
    inputs = inputs / 255.0
    res = res / 255.0
    return inputs, res
