from keras.layers import Conv2D, Input, Reshape, BatchNormalization, ReLU, MaxPooling2D, Flatten, Concatenate, Dense, Lambda
from keras.initializers.initializers_v2 import RandomNormal, HeNormal, GlorotUniform
from keras.models import Sequential, Model
from keras.regularizers import L2
from keras import metrics
import tensorflow as tf

import os

from layer_functions import euclidian_distance, contrastive_loss
from config import WIDTH, HEIGHT, SGD, ADAM, ADAM_SLOW, SGD_FASTER


def create_siamese_fully_connected():
    img_a_input = Input((HEIGHT, WIDTH), name="img_a_input")
    img_b_input = Input((HEIGHT, WIDTH), name="img_b_input")

    cnn = Sequential([Reshape((HEIGHT, WIDTH, 1)),
                      # Conv layer 1
                      Conv2D(64, kernel_size=7, strides=1,
                             kernel_initializer=RandomNormal(seed=1),
                             kernel_regularizer=L2(0.0001)),
                      BatchNormalization(),
                      ReLU(),
                      MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

                      # Conv layer 2
                      Conv2D(128, kernel_size=5, strides=1,
                             kernel_initializer=RandomNormal(seed=1),
                             kernel_regularizer=L2(0.0001)),
                      BatchNormalization(),
                      ReLU(),
                      MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                      # Conv layer 3
                      Conv2D(256, kernel_size=3, strides=1,
                             kernel_initializer=RandomNormal(seed=1),
                             kernel_regularizer=L2(0.0001)),
                      BatchNormalization(),
                      ReLU(),
                      MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                      # Conv layer 4
                      Conv2D(512, kernel_size=3, strides=2,
                             kernel_initializer=RandomNormal(seed=1),
                             kernel_regularizer=L2(0.0001)),
                      BatchNormalization(),
                      ReLU(),
                      MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                      Flatten(),
                      ])

    feature_vector_A = cnn(img_a_input)
    feature_vector_B = cnn(img_b_input)

    concat = Concatenate()([feature_vector_A, feature_vector_B])

    dense = tf.keras.layers.Dense(units=64,
                                  activation='relu',
                                  kernel_initializer=HeNormal(seed=1),
                                  kernel_regularizer=L2(0.0001))(concat)

    output = Dense(1, activation="sigmoid")(dense)

    model = Model(
        inputs=[img_a_input, img_b_input], outputs=output, name="default_FC-CL-64"
    )

    model.compile(loss=contrastive_loss,
                  optimizer=SGD,
                  metrics=[metrics.TruePositives(name="TPos"),
                           metrics.FalseNegatives(name="FNeg"),
                           metrics.TrueNegatives(name="TNeg"),
                           metrics.FalsePositives(name="FPos")]
                  )

    return model


def create_siamese_euclid_dist():
    img_a_input = Input((HEIGHT, WIDTH), name="img_a_input")
    img_b_input = Input((HEIGHT, WIDTH), name="img_b_input")

    cnn = Sequential([Reshape((HEIGHT, WIDTH, 1)),
                      # Conv layer 1
                      Conv2D(64, kernel_size=7, strides=1,
                             kernel_initializer=RandomNormal(seed=1),
                             kernel_regularizer=L2(0.001)),
                      BatchNormalization(),
                      ReLU(),
                      MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

                      # Conv layer 2
                      Conv2D(128, kernel_size=5, strides=1,
                             kernel_initializer=RandomNormal(seed=1),
                             kernel_regularizer=L2(0.001)),
                      BatchNormalization(),
                      ReLU(),
                      MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                      # Conv layer 3
                      Conv2D(256, kernel_size=3, strides=1,
                             kernel_initializer=RandomNormal(seed=1),
                             kernel_regularizer=L2(0.001)),
                      BatchNormalization(),
                      ReLU(),
                      MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                      # Conv layer 4
                      Conv2D(512, kernel_size=3, strides=2,
                             kernel_initializer=RandomNormal(seed=1),
                             kernel_regularizer=L2(0.001)),
                      BatchNormalization(),
                      ReLU(),
                      MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

                      Flatten(),
                      ])

    feature_vector_A = cnn(img_a_input)
    feature_vector_B = cnn(img_b_input)

    distance = Lambda(euclidian_distance)([feature_vector_A, feature_vector_B])

    output = Dense(1, activation="sigmoid")(distance)

    model = Model(
        inputs=[img_a_input, img_b_input], outputs=output, name="default_ED-CL"
    )

    model.compile(loss=contrastive_loss,
                  optimizer=ADAM,
                  metrics=[metrics.TruePositives(name="TPos"),
                           metrics.FalseNegatives(name="FNeg"),
                           metrics.TrueNegatives(name="TNeg"),
                           metrics.FalsePositives(name="FPos")]
                  )

    return model


if __name__ == "__main__":
    os.chdir("hand_writings/")

    model = create_siamese_fully_connected()
    model.summary()
    model.save("Models/" + f"{model.name}.h5")

    model = create_siamese_euclid_dist()
    model.summary()
    model.save("Models/" + f"{model.name}.h5")
