import tensorflow as tf
from tensorflow import keras


# 卷积层 = conv2d + bn + leaky_relu
def convolutional(inputs, filters, kernel_size, downsample=False, activate=True, bn=True):
    # 下采样
    if downsample:
        inputs = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        padding = 'valid'
        strides = 2
    else:
        padding = 'same'
        strides = 1
    conv = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                               padding=padding, use_bias=not bn, kernel_regularizer=keras.regularizers.l2(0.0005),
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                               bias_initializer=tf.constant_initializer(0.))(inputs)
    if bn:
        conv = keras.layers.BatchNormalization()(conv)
    if activate:
        conv = tf.nn.leaky_relu(conv, alpha=0.1)
    return conv


# 残差块
def residual_block(inputs, filters_num1, filters_num2):
    short_cut = inputs
    conv = convolutional(inputs, filters_num1, (1, 1))
    conv = convolutional(conv, filters_num2, (3, 3))
    residual_output = short_cut + conv
    return residual_output


# 上采样
def upsample(inputs):
    return tf.image.resize(inputs, (inputs.shape[1] * 2, inputs.shape[2] * 2), method='nearest')


