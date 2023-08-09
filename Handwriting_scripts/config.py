import tensorflow as tf

# IMAGE PROCESSING
WIDTH = int(1000*.75)
HEIGHT = int(342*.75)
INVERT = False

# ATTENTION MAPS GENERATION
FULL_RESOLUTION = True
GENERATE_FILTER_MAPS = False

# PATHS
WORKING_DIRECTORY = "/home/xstejs31/hand_writings"

# OPTIMIZERS
SGD = tf.keras.optimizers.SGD(learning_rate=0.001,
                              # beta_1=0.9,
                              # beta_2=0.999,
                              # epsilon=1e-7,
                              # amsgrad=False,
                              # decay=0.0005,
                              clipvalue=0.5,
                              )
SGD_FASTER = tf.keras.optimizers.SGD(learning_rate=0.005,
                                     # beta_1=0.9,
                                     # beta_2=0.999,
                                     # epsilon=1e-7,
                                     # amsgrad=False,
                                     # decay=0.0005,
                                     clipvalue=0.5,
                                     )
ADAM = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                beta_1=0.9,
                                beta_2=0.999,
                                epsilon=1e-7,
                                amsgrad=False,
                                # decay=0.0005,
                                clipvalue=0.5,
                                )
ADAM_SLOW = tf.keras.optimizers.Adam(learning_rate=0.00001,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=1e-7,
                                     amsgrad=False,
                                     # decay=0.0005,
                                     clipvalue=0.5,
                                     )
