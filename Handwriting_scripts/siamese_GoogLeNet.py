from keras.initializers.initializers_v2 import RandomUniform, RandomNormal, GlorotUniform, GlorotNormal, HeNormal
from keras.models import Sequential, Model
from keras.regularizers import L2
import tensorflow as tf

import os

from layer_functions import euclidian_distance, contrastive_loss
from config import WIDTH, HEIGHT, ADAM, SGD, SGD_FASTER


class InsceptionBlock(tf.keras.Model):
    def __init__(self, base_channels=32):
        super().__init__()
        self.base_channels = base_channels

        self.a = tf.keras.layers.Conv2D(
            base_channels*2,
            kernel_size=1,
            strides=1,
            activation="relu",
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        )

        self.b_1 = tf.keras.layers.Conv2D(
            base_channels*4,
            kernel_size=1,
            strides=1,
            activation="relu",
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        )
        self.b_2 = tf.keras.layers.Conv2D(
            base_channels*4,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        )

        self.c_1 = tf.keras.layers.Conv2D(
            base_channels,
            kernel_size=1,
            strides=1,
            activation="relu",
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        )
        self.c_2 = tf.keras.layers.Conv2D(
            base_channels,
            kernel_size=5,
            strides=1,
            padding="same",
            activation="relu",
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        )

        self.d_1 = tf.keras.layers.MaxPooling2D(
            pool_size=3,
            strides=1,
            padding="same"
        )
        self.d_2 = tf.keras.layers.Conv2D(
            base_channels,
            kernel_size=1,
            strides=1,
            activation="relu",
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        )

    def call(self, X):
        a = self.a(X)
        b = self.b_2(self.b_1(X))
        c = self.c_2(self.c_1(X))
        d = self.d_2(self.d_1(X))

        return tf.keras.layers.Concatenate(axis=-1)([a, b, c, d])

    def get_config(self):
        return {
            "base_channels": self.base_channels
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_siamese_inception_fully_conn():
    img_a_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_a_input")
    img_b_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_b_input")

    net = Sequential([
        tf.keras.layers.Reshape((HEIGHT, WIDTH, 1)),
        tf.keras.layers.Conv2D(filters=32,
                               #kernel_size=(7, 7),
                               kernel_size=(5, 5),
                               #strides=(4, 4),
                               strides=(3, 3),
                               # padding='same',
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)
                               ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     #strides=(3, 3),
                                     strides=(3, 3),
                                     # padding='same'
                                     ),
        tf.keras.layers.Conv2D(filters=64,
                               #kernel_size=(7, 7),
                               kernel_size=(5, 5),
                               #strides=(4, 4),
                               strides=(3, 3),
                               # padding='same',
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)
                               ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     # padding='same'
                                     ),
        InsceptionBlock(128),
        InsceptionBlock(256),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same'
                                     ),
        InsceptionBlock(512),
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
        inputs=[img_a_input, img_b_input], outputs=output, name="insception_FC-CL-64"
    )

    model.compile(loss=contrastive_loss,
                  optimizer=SGD,
                  metrics=[tf.keras.metrics.TruePositives(name="TPos"),
                           tf.keras.metrics.FalseNegatives(name="FNeg"),
                           tf.keras.metrics.TrueNegatives(name="TNeg"),
                           tf.keras.metrics.FalsePositives(name="FPos")]
                  )

    return model


def create_siamese_inception_euclidian_dist():
    img_a_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_a_input")
    img_b_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_b_input")

    net = Sequential([
        tf.keras.layers.Reshape((HEIGHT, WIDTH, 1)),
        tf.keras.layers.Conv2D(filters=32,
                               #kernel_size=(7, 7),
                               kernel_size=(5, 5),
                               #strides=(4, 4),
                               strides=(3, 3),
                               # padding='same',
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)
                               ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     #strides=(3, 3),
                                     strides=(3, 3),
                                     # padding='same'
                                     ),
        tf.keras.layers.Conv2D(filters=64,
                               #kernel_size=(7, 7),
                               kernel_size=(5, 5),
                               #strides=(4, 4),
                               strides=(3, 3),
                               # padding='same',
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)
                               ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     # padding='same'
                                     ),
        InsceptionBlock(128),
        InsceptionBlock(256),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same'
                                     ),
        InsceptionBlock(512),
        tf.keras.layers.Flatten()
    ])

    features_A = net(img_a_input)
    features_B = net(img_b_input)

    distance = tf.keras.layers.Lambda(
        euclidian_distance)([features_A, features_B])

    output = tf.keras.layers.Dense(units=1, activation='sigmoid')(distance)

    model = Model(
        inputs=[img_a_input, img_b_input], outputs=output, name="insception_ED-CL"
    )

    model.compile(loss=contrastive_loss,
                  optimizer=ADAM,
                  metrics=[tf.keras.metrics.TruePositives(name="TPos"),
                           tf.keras.metrics.FalseNegatives(name="FNeg"),
                           tf.keras.metrics.TrueNegatives(name="TNeg"),
                           tf.keras.metrics.FalsePositives(name="FPos")]
                  )

    return model


if __name__ == "__main__":
    os.chdir("hand_writings/")

    model = create_siamese_inception_euclidian_dist()
    model.summary()
    model.save("Models/" + f"{model.name}.h5")

    model = create_siamese_inception_fully_conn()
    model.summary()
    model.save("Models/" + f"{model.name}.h5")
