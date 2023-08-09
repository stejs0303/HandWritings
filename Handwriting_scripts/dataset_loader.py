from glob import glob

import numpy as np
import cv2 as cv
import csv
import os


def load_CSVs():
    return [file for file in glob("Photos/*.csv")]


def select_paths(csvs: list, csv_type: str):
    return [file_name for file_name in csvs if csv_type in file_name]


def load_content(paths: list, csv_delimiter: str):
    """Loads content of .csv i.e. paths."""
    dataset = []
    for path in paths:
        with open(path) as csv_file:
            file = csv.reader(csv_file, delimiter=csv_delimiter)
            for row in file:
                dataset.append([row[0], row[1], int(row[2])])
    return dataset


def load_dataset(csv_content: list, early_stop: int = -1):
    """Loads photos into memory"""
    X_pairs, y_pairs = [], []
    for idx, (imgA_path, imgB_path, match) in enumerate(csv_content):
        if idx == early_stop:
            break

        imgA = np.array(
            cv.imread(f"Photos/{imgA_path}", cv.IMREAD_GRAYSCALE)
        ).astype(np.float32)

        imgB = np.array(
            cv.imread(f"Photos/{imgB_path}", cv.IMREAD_GRAYSCALE)
        ).astype(np.float32)

        X_pairs.append([imgA, imgB])
        y_pairs.append(match)

    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)

    return X_pairs, y_pairs


def load_pair(imgA_path: str, imgB_path: str, match: int):
    """Loads a single pair of photos"""

    imgA = np.array(
        cv.imread(f"Photos/{imgA_path}", cv.IMREAD_GRAYSCALE)
    ).astype(np.float32)

    imgB = np.array(
        cv.imread(f"Photos/{imgB_path}", cv.IMREAD_GRAYSCALE)
    ).astype(np.float32)

    return [imgA, imgB], match


def initialize_dataset(csv_type: str, csv_delimiter: str, early_stop: int = -1):
    """
    Initializes dataset for training/testing/validation from csv files.

    > csv_type - train / valid / test / cross (train + valid) / ...\\
    > csv_delimiter - ; / : / ...
    """
    csvs = load_CSVs()
    print(f"Found {len(csvs)} .csv in /Photos/ folder.")

    csv_paths = select_paths(csvs, csv_type)
    print(f"Found {len(csv_paths)} files to process.")

    csv_content = load_content(csv_paths, csv_delimiter)
    print(f"Loaded files contain {len(csv_content)} pairs.")

    return load_dataset(csv_content, early_stop)


if __name__ == "__main__":
    os.chdir("hand_writings/")
    X_pairs, y_pairs = initialize_dataset("train", ';', 10)

    print(X_pairs.shape)
