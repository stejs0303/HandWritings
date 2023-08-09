from keras.initializers.initializers_v2 import RandomNormal, HeNormal, GlorotUniform
from keras.models import Sequential, Model
from keras.regularizers import L2
import tensorflow as tf

import os

from layer_functions import euclidian_distance, contrastive_loss
from config import WIDTH, HEIGHT, SGD, ADAM, ADAM_SLOW, SGD_FASTER


def create_siamese_VGG16_EUCL():
    img_a_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_a_input")
    img_b_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_b_input")

    net = Sequential([
        tf.keras.layers.Reshape((HEIGHT, WIDTH, 1)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=128,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=128,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten()
    ])

    features_A = net(img_a_input)
    features_B = net(img_b_input)

    distance = tf.keras.layers.Lambda(
        euclidian_distance)([features_A, features_B])

    output = tf.keras.layers.Dense(units=1, activation='sigmoid')(distance)

    model = Model(
        inputs=[img_a_input, img_b_input], outputs=output, name="VGG16_EU-CL"
    )

    model.compile(loss=contrastive_loss,
                  optimizer=ADAM,
                  metrics=[tf.keras.metrics.TruePositives(name="TPos"),
                           tf.keras.metrics.FalseNegatives(name="FNeg"),
                           tf.keras.metrics.TrueNegatives(name="TNeg"),
                           tf.keras.metrics.FalsePositives(name="FPos")])

    return model


def create_siamese_VGG16_FCCL():
    img_a_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_a_input")
    img_b_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_b_input")

    net = Sequential([
        tf.keras.layers.Reshape((HEIGHT, WIDTH, 1)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=128,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=128,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.Conv2D(filters=512,
                               kernel_size=(5, 5),
                               strides=(1, 1),
                               padding='same',
                               activation="relu",
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten()
    ])

    features_A = net(img_a_input)
    features_B = net(img_b_input)

    concat = tf.keras.layers.Concatenate()([features_A, features_B])

    dense = tf.keras.layers.Dense(units=64,
                                  activation='relu',
                                  kernel_initializer=HeNormal(seed=1),
                                  kernel_regularizer=L2(0.0001))(concat)

    output = tf.keras.layers.Dense(units=1, activation='sigmoid')(dense)

    model = Model(
        inputs=[img_a_input, img_b_input], outputs=output, name="VGG16_FC-CL"
    )

    model.compile(loss=contrastive_loss,
                  optimizer=SGD,
                  metrics=[tf.keras.metrics.TruePositives(name="TPos"),
                           tf.keras.metrics.FalseNegatives(name="FNeg"),
                           tf.keras.metrics.TrueNegatives(name="TNeg"),
                           tf.keras.metrics.FalsePositives(name="FPos")])

    return model

if __name__ == "__main__":
    os.chdir("hand_writings/")

    model = create_siamese_VGG16_EUCL()
    model.summary()
    model.save("Models/" + f"{model.name}.h5")

    model = create_siamese_VGG16_FCCL()
    model.summary()
    model.save("Models/" + f"{model.name}.h5")

