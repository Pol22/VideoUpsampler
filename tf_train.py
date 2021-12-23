import os
import argparse
import tensorflow as tf

from tf_model import ResUNet
from tf_dataset import get_dataset
from metrics import PSNR, SSIM


def train():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--model', help='H5 TF upsample model', default=None)
    parser.add_argument('--scale', help='Model scale', default=2, type=int)
    args = parser.parse_args()

    scale = args.scale
    train_folder = './frames'
    valid_folder = './test_data'
    crop_size = 96
    repeat_count = 1
    batch_size = 16
    in_channels = 9
    out_channels = 3

    epochs = 100
    lr = 1e-4
    out_dir = 'results'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if args.model is not None:
        model = tf.keras.models.load_model(args.model, compile=False)
        model.summary()
        print('Model loaded')
    else:
        model = ResUNet(scale, in_channels, out_channels)
        print('New model created')

    train_dataset = get_dataset(
        train_folder,
        batch_size,
        crop_size,
        scale,
        repeat_count
    )

    valid_dataset = get_dataset(
        valid_folder,
        batch_size,
        crop_size,
        scale,
        repeat_count
    )

    optimizer = tf.keras.optimizers.Adam(lr, beta_1=.5)
    model.compile(
        loss=tf.losses.MeanAbsoluteError(),
        optimizer=optimizer,
        metrics=[PSNR(), SSIM()])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_PSNR', patience=10, mode='max')
    csv_log = tf.keras.callbacks.CSVLogger(
        os.path.join(out_dir, f'train_resunet.log'))

    saver = tf.keras.callbacks.ModelCheckpoint(
        filepath=out_dir + '/resunet_{epoch:03d}-{val_PSNR:.2f}.h5',
        monitor='val_PSNR')

    callbacks = [csv_log, saver, early_stop]

    model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=valid_dataset,
        verbose=1
    )


if __name__ == '__main__':
    train()
