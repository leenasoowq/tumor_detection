from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model

def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(filters, 1, padding='same')(shortcut)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def build_resunet(input_shape):
    inputs = Input(input_shape)
    c1 = residual_block(inputs, 32)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = residual_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = residual_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = residual_block(p3, 256)

    u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = residual_block(u5, 128)
    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = residual_block(u6, 64)
    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = residual_block(u7, 32)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    return Model(inputs, outputs)
