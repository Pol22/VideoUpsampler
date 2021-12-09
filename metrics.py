import tensorflow as tf


class PSNR(tf.keras.metrics.Metric):
    def __init__(self, name='PSNR', **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.psnrs = self.add_weight(name='psnr', initializer='zeros')
        self.nums = self.add_weight(name='num', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        diff = tf.math.squared_difference(y_true, y_pred)
        mse = tf.reduce_mean(diff, axis=-1)
        mse = tf.reduce_mean(mse, axis=-1)
        mse = tf.reduce_mean(mse, axis=-1)
        psnr = -10.0 * tf.math.log(mse) / tf.math.log(10.0)
        self.psnrs.assign_add(tf.reduce_mean(psnr))
        self.nums.assign_add(1)

    def result(self):
        return self.psnrs / self.nums
    
    def reset_state(self):
        self.psnrs.assign(0)
        self.nums.assign(0)


class SSIM(tf.keras.metrics.Metric):
    def __init__(self, name='SSIM', **kwargs):
        super(SSIM, self).__init__(name=name, **kwargs)
        self.ssims = self.add_weight(name='ssim', initializer='zeros')
        self.nums = self.add_weight(name='num', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
        ssim = tf.reduce_mean(ssim)
        self.ssims.assign_add(ssim)
        self.nums.assign_add(1)

    def result(self):
        return self.ssims / self.nums
    
    def reset_state(self):
        self.ssims.assign(0)
        self.nums.assign(0)
