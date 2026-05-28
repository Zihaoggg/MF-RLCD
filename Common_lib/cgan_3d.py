import tensorflow as tf


OUTPUT_CHANNELS = 1


def _kernel_init():
    return tf.random_normal_initializer(0.0, 0.02)


def _downsample(filters: int, apply_batchnorm: bool = True) -> tf.keras.Sequential:
    block = tf.keras.Sequential()
    block.add(
        tf.keras.layers.Conv3D(
            filters,
            kernel_size=(3, 3, 3),
            strides=1,
            padding="valid",
            kernel_initializer=_kernel_init(),
            use_bias=False,
        )
    )
    if apply_batchnorm:
        block.add(tf.keras.layers.BatchNormalization())
    block.add(tf.keras.layers.LeakyReLU())
    return block


def _upsample(filters: int, apply_dropout: bool = False) -> tf.keras.Sequential:
    block = tf.keras.Sequential()
    block.add(
        tf.keras.layers.Conv3DTranspose(
            filters,
            kernel_size=(3, 3, 3),
            strides=1,
            padding="valid",
            kernel_initializer=_kernel_init(),
            use_bias=False,
        )
    )
    block.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        block.add(tf.keras.layers.Dropout(0.5))
    block.add(tf.keras.layers.ReLU())
    return block


def Generator() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(32, 32, 32, 4))
    down_stack = [
        _downsample(32),
        _downsample(32),
        _downsample(64),
        _downsample(128),
    ]
    up_stack = [
        _upsample(128),
        _upsample(64),
        _upsample(32),
        _upsample(32),
    ]

    x = inputs
    skips = []
    for block in down_stack:
        x = block(x)
        skips.append(x)

    for block, skip in zip(up_stack, reversed(skips[:-1])):
        x = block(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    outputs = tf.keras.layers.Conv3DTranspose(
        OUTPUT_CHANNELS,
        kernel_size=(3, 3, 3),
        strides=1,
        padding="valid",
        kernel_initializer=_kernel_init(),
        activation="tanh",
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="pix2pix3D_G")


def Discriminator() -> tf.keras.Model:
    input_image = tf.keras.layers.Input(shape=(32, 32, 32, 4), name="input_image")
    target_image = tf.keras.layers.Input(shape=(32, 32, 32, 1), name="target_image")
    x = tf.keras.layers.Concatenate()([input_image, target_image])

    x = _downsample(32, apply_batchnorm=False)(x)
    x = _downsample(32)(x)
    x = _downsample(64)(x)
    x = tf.keras.layers.ZeroPadding3D()(x)
    x = tf.keras.layers.Conv3D(
        128,
        kernel_size=(3, 3, 3),
        strides=1,
        kernel_initializer=_kernel_init(),
        use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.ZeroPadding3D()(x)
    outputs = tf.keras.layers.Conv3D(
        1,
        kernel_size=(3, 3, 3),
        strides=1,
        kernel_initializer=_kernel_init(),
    )(x)
    return tf.keras.Model(inputs=[input_image, target_image], outputs=outputs, name="pix2pix3D_D")
