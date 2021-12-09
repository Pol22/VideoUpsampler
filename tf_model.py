import tensorflow as tf


def Conv(inputs, filters, kernel_size=3):
    x = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=1, padding='same')(inputs)
    x = tf.keras.layers.LeakyReLU(alpha=0.02)(x)
    return x


def ResBlock(inputs, filters, kernel_size=3):
    x = Conv(inputs, filters, kernel_size)
    x = Conv(x, filters, kernel_size)
    return x + inputs


def ResUNet(in_channels, out_channels=3, kernel_size=3):
    inputs = tf.keras.Input(shape=(None, None, in_channels))

    x = Conv(inputs, 32, kernel_size)
    x1 = ResBlock(x, 32, kernel_size)
    x = tf.nn.space_to_depth(x1, 2)
    x = Conv(x, 64, kernel_size)
    x2 = ResBlock(x, 64, kernel_size)
    x = tf.nn.space_to_depth(x2, 2)
    x = Conv(x, 128, kernel_size)
    x3 = ResBlock(x, 128, kernel_size)
    x = tf.nn.space_to_depth(x3, 2)
    x = Conv(x, 256, kernel_size)
    x = ResBlock(x, 256, kernel_size)
    x = Conv(x, 128 * 2 * 2, kernel_size)
    x = tf.nn.depth_to_space(x, 2)
    x = x + x3
    x = ResBlock(x, 128, kernel_size)
    x = Conv(x, 64 * 2 * 2, kernel_size)
    x = tf.nn.depth_to_space(x, 2)
    x = x + x2
    x = ResBlock(x, 64, kernel_size)
    x = Conv(x, 32 * 2 * 2, kernel_size)
    x = tf.nn.depth_to_space(x, 2)
    x = x + x1
    x = ResBlock(x, 32, kernel_size)
    # upsample from input size
    x = Conv(x, 16 * 2 * 2, kernel_size)
    x = tf.nn.depth_to_space(x, 2)
    x = ResBlock(x, 16, kernel_size)
    out = tf.keras.layers.Conv2D(
        out_channels, 1, strides=1, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=out)