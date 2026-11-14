import tensorflow as tf


OUTPUT_CHANNELS = 1


def _downsample(filters, apply_batchnorm=True):
    layers = [tf.keras.layers.Conv3D(
        filters, (3, 3, 3), strides=1, padding="valid",
        kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
        use_bias=False,
    )]
    if apply_batchnorm:
        layers.append(tf.keras.layers.BatchNormalization())
    layers.append(tf.keras.layers.LeakyReLU())
    return tf.keras.Sequential(layers)


def _upsample(filters, apply_dropout=False):
    layers = [
        tf.keras.layers.Conv3DTranspose(
            filters, (3, 3, 3), strides=1, padding="valid",
            kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
            use_bias=False,
        ),
        tf.keras.layers.BatchNormalization(),
    ]
    if apply_dropout:
        layers.append(tf.keras.layers.Dropout(0.5))
    layers.append(tf.keras.layers.ReLU())
    return tf.keras.Sequential(layers)


def Generator():
    inputs = tf.keras.layers.Input(shape=(32, 32, 32, 4))
    down_blocks = [_downsample(filters) for filters in (32, 32, 64, 128)]
    up_blocks = [_upsample(filters) for filters in (128, 64, 32, 32)]
    features = inputs
    skips = []
    for block in down_blocks:
        features = block(features)
        skips.append(features)
    for block, skip in zip(up_blocks, reversed(skips[:-1])):
        features = tf.keras.layers.Concatenate()([block(features), skip])
    outputs = tf.keras.layers.Conv3DTranspose(
        OUTPUT_CHANNELS, (3, 3, 3), strides=1, padding="valid",
        kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
        activation="tanh",
    )(features)
    return tf.keras.Model(inputs, outputs, name="pix2pix3D_G")


def Discriminator():
    inputs = tf.keras.layers.Input(shape=(32, 32, 32, 4), name="input_image")
    targets = tf.keras.layers.Input(shape=(32, 32, 32, 1), name="target_image")
    features = tf.keras.layers.Concatenate()([inputs, targets])
    features = _downsample(32, apply_batchnorm=False)(features)
    features = _downsample(32)(features)
    features = _downsample(64)(features)
    features = tf.keras.layers.ZeroPadding3D()(features)
    features = tf.keras.layers.Conv3D(
        128, (3, 3, 3), strides=1,
        kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
        use_bias=False,
    )(features)
    features = tf.keras.layers.BatchNormalization()(features)
    features = tf.keras.layers.LeakyReLU()(features)
    features = tf.keras.layers.ZeroPadding3D()(features)
    outputs = tf.keras.layers.Conv3D(
        1, (3, 3, 3), strides=1,
        kernel_initializer=tf.random_normal_initializer(0.0, 0.02),
    )(features)
    return tf.keras.Model([inputs, targets], outputs, name="pix2pix3D_D")
