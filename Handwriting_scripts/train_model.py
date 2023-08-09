import keras.api._v2.keras as keras
import tensorflow as tf
import argparse
import time
import os

from keras.callbacks import ModelCheckpoint

from data_generators import DataLoader, ParallelDataLoader
from siamese_resnet import Residual18, Residual50
from siamese_GoogLeNet import InsceptionBlock
from layer_functions import contrastive_loss

from config import WORKING_DIRECTORY


def main():
    os.chdir(WORKING_DIRECTORY)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", dest="epochs", type=int, default=1)
    parser.add_argument("--init_epoch", dest="init_epoch", type=int, default=0)
    parser.add_argument("--verbose", dest="verbose", type=int, default=1)
    parser.add_argument("--batch_size", dest="batch_size", type=int, default=16)
    parser.add_argument("--gpu", dest="gpu", type=int, default=None)
    parser.add_argument("--model", dest="model", type=str, default=None)
    args = parser.parse_args()

    curr_time = time.localtime()

    if args.gpu is not None and (gpus := tf.config.experimental.list_physical_devices("GPU")):
        try:
            tf.config.experimental.set_visible_devices(gpus[args.gpu], "GPU")
            tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
        except RuntimeError as e:
            print(e)

    try:
        model: keras.Model = keras.models.load_model(
            f"{args.model}",
            custom_objects={"Residual18": Residual18,
                            "Residual50": Residual50,
                            "InsceptionBlock": InsceptionBlock,
                            "contrastive_loss": contrastive_loss}
        )
    except RuntimeError as e:
        print("Model could not be loaded!")
        print(e)

    print(f"Loaded {model.name} model.")
    if not args.init_epoch and not os.path.exists(f"Models/{model.name}/"):
        os.mkdir(f"Models/{model.name}/")

    training_generator = ParallelDataLoader(
        "train", batch_size=args.batch_size
    )
    validation_generator = DataLoader("valid")

    name = model.name.split("/")
    path = f"Models/{name[0]}/epoch_"
    training_history: keras.callbacks.History = model.fit(
        training_generator,
        validation_data=validation_generator,
        initial_epoch=args.init_epoch,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
        shuffle=True,
        callbacks=ModelCheckpoint(filepath=path+"{epoch:02d}.h5",
                                  monitor="val_loss",
                                  verbose=1,
                                  save_best_only=False))

    with open(
        f"Results/{model.name} D={curr_time.tm_mday}.{curr_time.tm_mon} T={curr_time.tm_hour}.{curr_time.tm_min} Params={model.count_params()}.txt", 'a'
    ) as file:
        for key, values in training_history.history.items():
            file.write(f"{key}: ")
            for value in values:
                file.write(f"{value}, ")
            file.write("\n")


if __name__ == "__main__":
    main()
