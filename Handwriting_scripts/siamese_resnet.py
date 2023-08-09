from keras.initializers.initializers_v2 import RandomUniform, RandomNormal, GlorotUniform, GlorotNormal, HeNormal
from keras.models import Sequential, Model
from keras.regularizers import L2
import tensorflow as tf

import os

from layer_functions import euclidian_distance, contrastive_loss
from config import WIDTH, HEIGHT, SGD, ADAM, ADAM_SLOW, SGD_FASTER


class Residual18(tf.keras.Model):
    """The Residual block of ResNet18 models."""

    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.num_channels = num_channels
        self.use_1x1conv = use_1x1conv
        self.strides = strides
        self.conv1 = tf.keras.layers.Conv2D(
            num_channels,
            padding='same',
            kernel_size=3,
            strides=strides,
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        )

        self.conv2 = tf.keras.layers.Conv2D(
            num_channels,
            kernel_size=3,
            padding='same',
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        )

        self.conv3 = tf.keras.layers.Conv2D(
            num_channels,
            kernel_size=1,
            strides=strides,
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        ) if use_1x1conv else None

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)

    def get_config(self):
        return {
            "num_channels": self.num_channels,
            "use_1x1conv": self.use_1x1conv,
            "strides": self.strides}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Residual50(tf.keras.Model):
    """The Residual block of ResNet50 models."""

    def __init__(self, num_channels, use_1x1conv=False, strides=1, shrink=1):
        super().__init__()
        self.num_channels = num_channels
        self.use_1x1conv = use_1x1conv
        self.strides = strides
        self.shrink = shrink

        self.conv1 = tf.keras.layers.Conv2D(
            num_channels * shrink,
            padding='same',
            kernel_size=1,
            strides=strides,
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        )

        self.conv2 = tf.keras.layers.Conv2D(
            num_channels,
            kernel_size=3,
            padding='same',
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        )

        self.conv3 = tf.keras.layers.Conv2D(
            num_channels * 4,
            kernel_size=1,
            padding='same',
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        )

        self.conv4 = tf.keras.layers.Conv2D(
            num_channels * 4,
            kernel_size=1,
            strides=strides,
            kernel_initializer=GlorotUniform(seed=1),
            kernel_regularizer=L2(0.0005)
        ) if use_1x1conv else None

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = tf.keras.activations.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4 is not None:
            X = self.conv4(X)
        Y += X
        return tf.keras.activations.relu(Y)

    def get_config(self):
        return {
            "num_channels": self.num_channels,
            "use_1x1conv": self.use_1x1conv,
            "strides": self.strides,
            "shrink": self.shrink}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def add_resnet18_block(num_residuals, num_channels, first_block=False):
    block = Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.add(Residual18(num_channels, use_1x1conv=True, strides=2))
        else:
            block.add(Residual18(num_channels))
    return block


def add_resnet50_block(num_residuals, num_channels, first_block=False):
    block = Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.add(Residual50(num_channels,
                      use_1x1conv=True, strides=2, shrink=2))
        else:
            block.add(Residual50(num_channels))

    return block


def create_resnet18_euclid_dist():
    img_a_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_a_input")
    img_b_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_b_input")

    net = Sequential([
        tf.keras.layers.Reshape((HEIGHT, WIDTH, 1)),
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)
                               ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same'
                                     ),
        add_resnet18_block(2, 64),
        add_resnet18_block(2, 128),
        add_resnet18_block(2, 256),
        add_resnet18_block(2, 512),
        tf.keras.layers.Flatten()
    ])

    features_A = net(img_a_input)
    features_B = net(img_b_input)

    distance = tf.keras.layers.Lambda(
        euclidian_distance)([features_A, features_B])

    output = tf.keras.layers.Dense(units=1, activation='sigmoid')(distance)

    model = Model(
        inputs=[img_a_input, img_b_input], outputs=output, name="resnet18_ED-CL"
    )

    model.compile(loss=contrastive_loss,
                  optimizer=ADAM,
                  metrics=[tf.keras.metrics.TruePositives(name="TPos"),
                           tf.keras.metrics.FalseNegatives(name="FNeg"),
                           tf.keras.metrics.TrueNegatives(name="TNeg"),
                           tf.keras.metrics.FalsePositives(name="FPos")]
                  )

    return model


def create_resnet18_fully_connected():

    img_a_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_a_input")
    img_b_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_b_input")

    net = Sequential([
        tf.keras.layers.Reshape((HEIGHT, WIDTH, 1)),
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)
                               ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same'
                                     ),
        add_resnet18_block(2, 64),
        add_resnet18_block(2, 128),
        add_resnet18_block(2, 256),
        add_resnet18_block(2, 512),
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
        inputs=[img_a_input, img_b_input], outputs=output, name="resnet18_FC-CL-64"
    )

    model.compile(loss=contrastive_loss,
                  optimizer=SGD,
                  metrics=[tf.keras.metrics.TruePositives(name="TPos"),
                           tf.keras.metrics.FalseNegatives(name="FNeg"),
                           tf.keras.metrics.TrueNegatives(name="TNeg"),
                           tf.keras.metrics.FalsePositives(name="FPos")]
                  )

    return model


def create_resnet34_euclid_dist():
    img_a_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_a_input")
    img_b_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_b_input")

    net = Sequential([
        tf.keras.layers.Reshape((HEIGHT, WIDTH, 1)),
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)
                               ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same'
                                     ),
        add_resnet18_block(3, 64),
        add_resnet18_block(4, 128),
        add_resnet18_block(6, 256),
        add_resnet18_block(3, 512),
        tf.keras.layers.Flatten()
    ])

    features_A = net(img_a_input)
    features_B = net(img_b_input)

    distance = tf.keras.layers.Lambda(
        euclidian_distance)([features_A, features_B])

    output = tf.keras.layers.Dense(units=1, activation='sigmoid')(distance)

    model = Model(
        inputs=[img_a_input, img_b_input], outputs=output, name="resnet34_ED-CL"
    )

    model.compile(loss=contrastive_loss,
                  optimizer=ADAM,
                  metrics=[tf.keras.metrics.TruePositives(name="TPos"),
                           tf.keras.metrics.FalseNegatives(name="FNeg"),
                           tf.keras.metrics.TrueNegatives(name="TNeg"),
                           tf.keras.metrics.FalsePositives(name="FPos")]
                  )

    return model


def create_resnet34_fully_connected():
    img_a_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_a_input")
    img_b_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_b_input")

    net = Sequential([
        tf.keras.layers.Reshape((HEIGHT, WIDTH, 1)),
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)
                               ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same'
                                     ),
        add_resnet18_block(3, 64),
        add_resnet18_block(4, 128),
        add_resnet18_block(6, 256),
        add_resnet18_block(3, 512),
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
        inputs=[img_a_input, img_b_input], outputs=output, name="resnet34_FC-CL-64"
    )

    model.compile(loss=contrastive_loss,
                  optimizer=SGD,
                  metrics=[tf.keras.metrics.TruePositives(name="TPos"),
                           tf.keras.metrics.FalseNegatives(name="FNeg"),
                           tf.keras.metrics.TrueNegatives(name="TNeg"),
                           tf.keras.metrics.FalsePositives(name="FPos")]
                  )

    return model


def create_resnet50_euclid_dist():
    img_a_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_a_input")
    img_b_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_b_input")

    net = Sequential([
        tf.keras.layers.Reshape((HEIGHT, WIDTH, 1)),
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)
                               ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same'
                                     ),
        add_resnet50_block(3, 64),
        add_resnet50_block(4, 128),
        add_resnet50_block(6, 256),
        add_resnet50_block(3, 512),
        tf.keras.layers.Flatten()
    ])

    features_A = net(img_a_input)
    features_B = net(img_b_input)

    distance = tf.keras.layers.Lambda(
        euclidian_distance)([features_A, features_B])

    output = tf.keras.layers.Dense(units=1, activation='sigmoid')(distance)

    model = Model(
        inputs=[img_a_input, img_b_input], outputs=output, name="resnet50_ED-CL"
    )

    model.compile(loss=contrastive_loss,
                  optimizer=ADAM,
                  metrics=[tf.keras.metrics.TruePositives(name="TPos"),
                           tf.keras.metrics.FalseNegatives(name="FNeg"),
                           tf.keras.metrics.TrueNegatives(name="TNeg"),
                           tf.keras.metrics.FalsePositives(name="FPos")]
                  )

    return model


def create_resnet50_fully_connected():
    img_a_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_a_input")
    img_b_input = tf.keras.layers.Input((HEIGHT, WIDTH), name="img_b_input")

    net = Sequential([
        tf.keras.layers.Reshape((HEIGHT, WIDTH, 1)),
        tf.keras.layers.Conv2D(filters=32,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=GlorotUniform(seed=1),
                               kernel_regularizer=L2(0.0001)
                               ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3),
                                     strides=(2, 2),
                                     padding='same'
                                     ),
        add_resnet50_block(3, 64),
        add_resnet50_block(4, 128),
        add_resnet50_block(6, 256),
        add_resnet50_block(3, 512),
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
        inputs=[img_a_input, img_b_input], outputs=output, name="resnet50_FC-CL-64"
    )

    model.compile(loss=contrastive_loss,
                  optimizer=SGD,
                  metrics=[tf.keras.metrics.TruePositives(name="TPos"),
                           tf.keras.metrics.FalseNegatives(name="FNeg"),
                           tf.keras.metrics.TrueNegatives(name="TNeg"),
                           tf.keras.metrics.FalsePositives(name="FPos")]
                  )

    return model


if __name__ == "__main__":
    os.chdir("hand_writings/")

    model = create_resnet18_euclid_dist()
    model.summary()
    model.save("Models/" + f"{model.name}.h5")

    model = create_resnet18_fully_connected()
    model.summary()
    model.save("Models/" + f"{model.name}.h5")
    
    model = create_resnet34_euclid_dist()
    model.summary()
    model.save("Models/" + f"{model.name}.h5")
    
    model = create_resnet34_fully_connected()
    model.summary()
    model.save("Models/" + f"{model.name}.h5")
    
    """ model = create_resnet50_euclid_dist()
    model.summary()
    model.save("Models/" + f"{model.name}.h5")
    
    model = create_resnet50_fully_connected()
    model.summary()
    model.save("Models/" + f"{model.name}.h5") """
