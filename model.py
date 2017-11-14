from keras.models import Model
from keras import layers
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation, BatchNormalization, AveragePooling2D


def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    f1, f2, f3 = filters

    x = Conv2D(f1, (1, 1), strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(f2, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(f3, (1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    shortcut = Conv2D(f3, (1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = Activation("relu")(x)

    return x


def identity_block(input_tensor, kernel_size, filters):
    f1, f2, f3 = filters

    x = Conv2D(f1, (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(f2, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(f3, (1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = layers.add([x, input_tensor])
    x = Activation("relu")(x)

    return x

def get_model(input_shape=(32, 32, 3), n_classes=42):
    input = Input(shape=input_shape)

    x = Conv2D(16, (5, 5), strides=(1, 1), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = conv_block(x, 3, [16, 16, 64], strides=(1, 1))
    x = identity_block(x, 3, [16, 16, 64])
    x = identity_block(x, 3, [16, 16, 64])

    x = conv_block(x, 3, [32, 32, 128], strides=(1, 1))
    x = identity_block(x, 3, [32, 32, 128])
    x = identity_block(x, 3, [32, 32, 128])

    x = AveragePooling2D((5, 5), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(512, activation="sigmoid")(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation="softmax")(x)

    return Model(inputs=input, outputs=x)