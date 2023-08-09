import keras.api._v2.keras as keras
import tensorflow as tf
import numpy as np
import cv2 as cv
import os

from siamese_resnet import Residual18, Residual50
from siamese_GoogLeNet import InsceptionBlock
from layer_functions import contrastive_loss
from dataset_loader import load_pair

from config import WORKING_DIRECTORY, WIDTH, HEIGHT, FULL_RESOLUTION, GENERATE_FILTER_MAPS

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def get_model_structure(sequential: tf.keras.Model, subnetworks: list = ["sequential"], depth: int = 0):
    for layer in sequential.layers:
        print(f" {'-' * depth}> {layer.name} (input={layer.input_shape}, output={layer.output_shape})")
        if any(subnetwork in layer.name for subnetwork in subnetworks):
            get_model_structure(layer, subnetworks, depth+1)


def get_activations(path_a: str, path_b: str, model):
    pair, match = load_pair(path_a, path_b, match=1)
    X = np.array([pair])

    return model.predict(X[:, 0, :, :])


def makedirs(model_name: str, names: str):
    for layer_name in names:
        os.makedirs(f"Attention_maps/{model_name}/{layer_name}", exist_ok=True)


def normalize(image: np.ndarray, width: int, height: int):
    img: np.ndarray = cv.resize(
        image, (width, height), interpolation=cv.INTER_LINEAR
    )
    img -= img.min()
    img /= img.max()
    img *= 255

    return img


def generate_maps(image: np.ndarray, width: int, height: int, attention_map_path: str, heat_map_path: str):
    img = normalize(image, width, height)
    cv.imwrite(attention_map_path, img)

    heat_map = cv.applyColorMap(img.astype(np.uint8), cv.COLORMAP_JET)
    cv.imwrite(heat_map_path, heat_map)


def create_filter_maps(images: np.ndarray, model_name: str, layer_name: str, width: int, height: int):
    os.makedirs(
        f"Attention_maps/{model_name}/{layer_name}/attention_map", exist_ok=True
    )
    os.makedirs(
        f"Attention_maps/{model_name}/{layer_name}/heat_map", exist_ok=True
    )

    for idx in range(images.shape[2]):
        generate_maps(
            image=images[:, :, idx],
            attention_map_path=f"Attention_maps/{model_name}/{layer_name}/attention_map/{idx+1}.png",
            heat_map_path=f"Attention_maps/{model_name}/{layer_name}/heat_map/{idx+1}.png",
            width=width,
            height=height
        )


def create_maps(model_name: str, names: np.ndarray, activations: np.ndarray, width: int, height: int):
    makedirs(model_name, names)

    attention_map = np.zeros((height, width), np.float32)
    for layer_name, activation in zip(names, activations):
        images: np.ndarray = np.absolute(activation[0])

        # Attention maps of filters
        if GENERATE_FILTER_MAPS:
            create_filter_maps(
                images, model_name, layer_name, width, height
            )

        # Attention maps of layers
        img = np.sum(images, axis=2)
        generate_maps(
            image=img,
            attention_map_path=f"Attention_maps/{model_name}/{layer_name}/attention_map.png",
            heat_map_path=f"Attention_maps/{model_name}/{layer_name}/heat_map.png",
            width=width,
            height=height
        )

        attention_map = cv.add(
            attention_map, normalize(img, width, height), dtype=cv.CV_32F
        )

    # Combination of all layers
    generate_maps(
        image=attention_map,
        attention_map_path=f"Attention_maps/{model_name}/attention_map.png",
        heat_map_path=f"Attention_maps/{model_name}/heat_map.png",
        width=width,
        height=height
    )


def apply_heat_map(img_path: str, model_name: str, width: int, height: int):
    image = cv.resize(
        cv.imread(img_path, cv.IMREAD_COLOR),
        (width, height), interpolation=cv.INTER_AREA
    )
    heat_map = cv.imread(f"Attention_maps/{model_name}/heat_map.png", cv.IMREAD_COLOR)

    output = cv.addWeighted(heat_map, .7, image, .3, 0)

    cv.imwrite(f"Attention_maps/{model_name}/applied_heat_map.png", output)


def resnet18(model: tf.keras.Model):

    # Get one of the model's inputs
    img_a_input = model.get_layer(index=0).input

    # Extract Sequential cnn
    cnn: tf.keras.Sequential = model.get_layer(index=2)

    # Reconstruct cnn using functional aproach to access output of individual layers
    reshape = cnn.layers[0](img_a_input)
    conv2D = cnn.layers[1](reshape)
    batch_norm = cnn.layers[2](conv2D)
    re_lu = cnn.layers[3](batch_norm)
    max_pooling = cnn.layers[4](re_lu)
    residual1 = cnn.layers[5].layers[0](max_pooling)
    residual2 = cnn.layers[5].layers[1](residual1)
    residual3 = cnn.layers[6].layers[0](residual2)
    residual4 = cnn.layers[6].layers[1](residual3)
    residual5 = cnn.layers[7].layers[0](residual4)
    residual6 = cnn.layers[7].layers[1](residual5)
    residual7 = cnn.layers[8].layers[0](residual6)
    residual8 = cnn.layers[8].layers[1](residual7)
    flatten = cnn.layers[9](residual8)

    outputs = [
        re_lu, max_pooling, residual1, residual2, residual3, residual4,
        residual5, residual6, residual7, residual8, flatten
    ]

    layers_name = np.array([
        "re_lu", "max_pooling", "residual1", "residual2", "residual3", "residual4",
        "residual5", "residual6", "residual7", "residual8", "flatten"
    ])

    cnn_model = tf.keras.Model(inputs=img_a_input, outputs=outputs)

    """ "paragraphs/all/0/00465.jpg_p4.png", "paragraphs/all/0/00465.jpg_p4.png" """
    """ "paragraphs/all/0/00202.jpg_p4.png", "paragraphs/all/0/00202.jpg_p1.png" """
    """ "paragraphs/all/0/00541.jpg_p2.png", "paragraphs/all/0/00545.jpg_p3.png" """
    img_a_path, img_b_path = "paragraphs/all/0/00202.jpg_p4.png", "paragraphs/all/0/00202.jpg_p1.png"

    activations = get_activations(img_a_path, img_b_path, cnn_model)

    width = WIDTH if FULL_RESOLUTION else activations[0][0].shape[1]
    height = HEIGHT if FULL_RESOLUTION else activations[0][0].shape[0]

    model_name = "resnet18-fullyconnected"
    
    create_maps(
        model_name=model_name,
        names=layers_name[: -1],
        activations=activations[: -1],
        width=width,
        height=height
    )

    apply_heat_map(
        img_path=f"Photos/{img_a_path}",
        model_name=model_name,
        width=width,
        height=height
    )

    print("Done")

def VGG(model: tf.keras.Model):
   
    # Get one of the model's inputs
    img_a_input = model.get_layer(index=0).input

    # Extract Sequential cnn
    cnn: tf.keras.Sequential = model.get_layer(index=2)

    # Reconstruct cnn using functional aproach to access output of individual layers
    reshape = cnn.layers[0](img_a_input)
    conv1 = cnn.layers[1](reshape)
    conv2 = cnn.layers[2](conv1)
    max_pooling1 = cnn.layers[3](conv2)
    conv3 = cnn.layers[4](max_pooling1)
    conv4 = cnn.layers[5](conv3)
    max_pooling2 = cnn.layers[6](conv4)
    conv5 = cnn.layers[7](max_pooling2)
    conv6 = cnn.layers[8](conv5)
    conv7 = cnn.layers[9](conv6)
    max_pooling3 = cnn.layers[10](conv7)
    conv8 = cnn.layers[11](max_pooling3)
    conv9 = cnn.layers[12](conv8)
    conv10 = cnn.layers[13](conv9)
    max_pooling4 = cnn.layers[14](conv10)
    conv11 = cnn.layers[15](max_pooling4)
    conv12 = cnn.layers[16](conv11)
    conv13 = cnn.layers[17](conv12)
    max_pooling5 = cnn.layers[18](conv13)
    flatten = cnn.layers[19](max_pooling5)

    outputs = [
        conv1, conv2, max_pooling1, conv3, conv4, max_pooling2,
        conv5, conv6, conv7, max_pooling3, conv8, conv9, conv10, 
        max_pooling4, conv11, conv12, conv13, max_pooling5, flatten
    ]

    layers_name = np.array([
        "conv1", "conv2", "max_pooling1", "conv3", "conv4", "max_pooling2",
        "conv5", "conv6", "conv7", "max_pooling3", "conv8", "conv9", "conv10", 
        "max_pooling4", "conv11", "conv12", "conv13", "max_pooling5", "flatten"
    ])

    cnn_model = tf.keras.Model(inputs=img_a_input, outputs=outputs)

    """ "paragraphs/all/0/00465.jpg_p4.png", "paragraphs/all/0/00465.jpg_p4.png" """
    """ "paragraphs/all/0/00202.jpg_p4.png", "paragraphs/all/0/00202.jpg_p1.png" """
    """ "paragraphs/all/0/00541.jpg_p2.png", "paragraphs/all/0/00545.jpg_p3.png" """
    img_a_path, img_b_path = "paragraphs/all/0/00202.jpg_p4.png", "paragraphs/all/0/00202.jpg_p1.png"

    activations = get_activations(img_a_path, img_b_path, cnn_model)

    width = WIDTH if FULL_RESOLUTION else activations[0][0].shape[1]
    height = HEIGHT if FULL_RESOLUTION else activations[0][0].shape[0]

    model_name = "VGG16-FC"
    
    create_maps(
        model_name=model_name,
        names=layers_name[: -1],
        activations=activations[: -1],
        width=width,
        height=height
    )

    apply_heat_map(
        img_path=f"Photos/{img_a_path}",
        model_name=model_name,
        width=width,
        height=height
    )

    print("Done")



if __name__ == "__main__":
    os.chdir(WORKING_DIRECTORY)

    try:
        model: keras.Model = keras.models.load_model(
            # Working
            #filepath="Keepers/resnet18_750x256_Adam/epoch_20.h5",
            
            # Working Fully Connected
            filepath="Keepers/resnet18_750x256_SGD/epoch_09.h5",
            
            # VGG EU
            #filepath="Keepers/VGG16_750x256_Adam/epoch_18.h5",
            
            # VGG Fully Connected
            #filepath="Keepers/VGG16_750x256_SGD/epoch_19.h5",
            custom_objects={
                "Residual18": Residual18,
                "Residual50": Residual50,
                "InsceptionBlock": InsceptionBlock,
                "contrastive_loss": contrastive_loss
            },
            compile=False
        )
    except RuntimeError as e:
        print("Model could not be loaded!")
        print(e)

    resnet18(model)
    """ VGG(model) """
    """ get_model_structure(model, ["sequential"]) """
