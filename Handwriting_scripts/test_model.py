import keras.api._v2.keras as keras
import tensorflow as tf
import numpy as np
import argparse
import os

from siamese_resnet import Residual18, Residual50
from siamese_GoogLeNet import InsceptionBlock
from layer_functions import contrastive_loss
from data_generators import DataLoader

from config import WORKING_DIRECTORY

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def main():
    os.chdir(WORKING_DIRECTORY)

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", dest="verbose", type=int, default=1)
    parser.add_argument("--gpu", dest="gpu", type=int, default=None)
    parser.add_argument("--model", dest="model", type=str, default=None)
    args = parser.parse_args()

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

    testing_generator = DataLoader("test")

    testing_result: np.ndarray = model.predict(testing_generator,
                                               batch_size=1,
                                               verbose=args.verbose)

    y_test = [match for _, _, match in testing_generator.content]

    Correct_Positive, Correct_Negative, False_Positive, False_Negative = 0, 0, 0, 0
    for predicted, accurate in zip(testing_result, y_test):
        if round(predicted[0]) == accurate:
            if accurate == 1:
                Correct_Positive += 1
            else:
                Correct_Negative += 1
        else:
            if accurate == 1:
                False_Negative += 1
            else:
                False_Positive += 1
    
    # Jelikož contrastive loss přiřazuje podobným ukázkám nižší hodnoty a rozdílným vyšší, 
    # jsou jednotlivé proměnné prohozené a místo přesnosti je počítána chybovost.
    print(f"Positive error rate: {((Correct_Positive/(Correct_Positive + False_Negative))*100):0.1f} %\n"
          f"Negative error rate: {((Correct_Negative/(Correct_Negative + False_Positive))*100):0.1f} %\n"
          f"Overall error rate: {(((Correct_Positive + Correct_Negative)/len(y_test))*100):0.1f} %")


if __name__ == "__main__":
    main()
