import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

ROOT = "Keepers"
BEST750 = [
    "resnet18_750x256_Adam", 
    "resnet50_750x256_SGD",
    "VGG16_750x256_Adam", 
    "VGG16_750x256_SGD"
]
BEST1000 = [
    "resnet18_1000x342_Adam",
    "resnet18_1000x342_SGD",
    "VGG16_1000x342_SGD", 
    "insception_1000x342_SGD", 
]
BEST1250 = [
    "insception_1250x427_SGD", 
    "VGG16_1250x427_SGD",
    "resnet18_1250x427_Adam", 
    "resnet34_1250x427_Adam",
]

NAMES = {
    "resnet18": "ResNet18",
    "resnet50": "ResNet50",
    "VGG16": "VGG16",
    "insception": "GoogLeNet",
    "resnet34": "ResNet34"
}

COLORS = {
    "resnet18_750x256_Adam": "tab:blue",
    "resnet18_1000x342_Adam": "tab:blue",
    "resnet18_1250x427_Adam": "tab:blue",
    "resnet18_1000x342_SGD": "tab:pink",
    "resnet34_1250x427_Adam": "tab:green",
    "resnet50_750x256_SGD": "tab:orange",
    "VGG16_750x256_Adam": "tab:red",
    "VGG16_750x256_SGD": "tab:purple",
    "VGG16_1000x342_SGD": "tab:purple",
    "VGG16_1250x427_SGD": "tab:purple",
    "insception_1000x342_SGD": "tab:cyan",
    "insception_1250x427_SGD": "tab:cyan",
}

folders = os.listdir(os.path.join(ROOT))
files = {
    model: glob(os.path.join(f"{ROOT}\\{model}\\*.txt"))[0] 
           for model in folders
}

def get_data(list_of_models:list):
    results = {}
    for model in list_of_models:
        data = {}
        with open(files[model], 'r') as training_results:
            while ((line := training_results.readline())):
                splits = line.split(':', 1)
                data[splits[0]] = [
                    float(value) for value in splits[1].split(',')[:-1]
                ]

        accuracy = [
            np.round(((FPos + FNeg)/(FPos + FNeg + TPos + TNeg))*100, 2)
            for FPos, FNeg, TPos, TNeg
            in zip(data['val_FPos'], data['val_FNeg'], data['val_TPos'], data['val_TNeg'])
        ]

        results[model] = {
            "accuracy": accuracy,
            "loss": data["val_loss"],
            "name": f"{NAMES[model.split('_')[0]]}\n{'Euklidovská metrika' if 'Adam' in model else 'Plně propojené'}"
        }
    return results

def plot_acc_loss(models: list):
    results = get_data(models)
    # create a figure and axis object
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, layout='constrained', figsize=(12, 7))

    # plot the first graph
    ax1.set_ylabel("Přesnost odhadu [%]", fontsize=14)

    # plot the second graph
    ax2_2 = ax2.twinx()
    ax2.set_ylabel("Ztrátovost [~]", fontsize=14)
    ax2.set_xlabel("epocha [~]", fontsize=14)
    ax2_2.set_ylabel("Ztrátovost [~]", fontsize=14)

    for idx, model in enumerate(models):
        ax1.plot(
            results[model]['accuracy'][:30], '-', 
            label=f"{results[model]['name']}", 
            linewidth=2, color=COLORS[model]
        )
        if results[model]["loss"][0] < 1:
            ax2.plot(
                results[model]['loss'][:30], "-", 
                #label=f"{results[model]['name']}", 
                linewidth=2, color=COLORS[model]
            )
        else:
            ax2_2.plot(
                results[model]['loss'][:30], "-", 
                #label=f"{results[model]['name']}", 
                linewidth=2, color=COLORS[model]
            )

    # display the plot
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(1, 0.982))
    plt.show()

def plot_acc(models: list):
    results = get_data(models)
    # create a figure and axis object
    fig, ax = plt.subplots(figsize=(8, 3))

    # plot the first graph
    ax.set_ylabel("Přesnost odhadu [%]", fontsize=14)
    ax.set_xlabel("Epocha [~]", fontsize=14)

    for idx, model in enumerate(models):
        ax.plot(
            results[model]['accuracy'][:30], '-', 
            label=f"{results[model]['name']}", 
            linewidth=2, color=COLORS[model]
        )

    # display the plot
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(1, 0.982))
    plt.show()

if __name__=="__main__":
    #plot_acc_loss(BEST750)
    plot_acc(BEST1000)