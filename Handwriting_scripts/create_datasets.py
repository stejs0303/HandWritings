from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import defaultdict

import csv
import os
import sys
import itertools


def add_same(dictionary1: dict, dictionary2: dict = None):
    if dictionary2 is None:
        dictionary2 = dictionary1

    same_author_pairs = []
    for key in dictionary1.keys():
        same_author_pairs.extend(
            [pair for pair in itertools.product(dictionary1[key],
                                                dictionary2[key],
                                                [1])]
        )

    return same_author_pairs


def add_different(dictionary1: dict, dictionary2: dict = None):
    if dictionary2 is None:
        dictionary2 = dictionary1

    different_author_pairs = []
    ids = list(dictionary1.keys())
    for itter in range(len(ids)):
        custom_list = []
        for idx in range(len(dictionary2[ids[itter]])):
            custom_list.append(
                dictionary2[ids[(itter + idx + 1) % len(ids)]][idx])

        different_author_pairs.extend(
            [pair for pair in itertools.product(dictionary1[ids[itter]],
                                                custom_list,
                                                [0])]
        )

    return different_author_pairs


def split(pairs: list, num_of_people: int):
    return train_test_split(
        pairs,
        test_size=((num_of_people*(4**2)) / len(pairs)),
        random_state=1,
        shuffle=False
    )


def get_train_test_valid(operate_on: list, testing_num_of_people: int, validation_num_of_people: int):
    final_train, final_test, final_valid = [], [], []
    for photos in operate_on:
        train, test = split(photos, num_of_people=testing_num_of_people)
        train, valid = split(train, num_of_people=validation_num_of_people)

        final_train.extend(train)
        final_test.extend(test)
        final_valid.extend(valid)

    return final_train, final_test, final_valid


def load_photo_names(type, folder, path):
    photos = defaultdict(list)

    for name in os.listdir(os.path.dirname(f"{path}/paragraphs/{type}/{folder}/")):
        name_split = name.strip(".png").strip("_augmented").split("_")
        id = name_split[0].strip(".jpg")

        photos[id].append(f"paragraphs/{type}/{folder}/{name}")

        sys.stdout.write(f"\rLoading file {name}.")
    sys.stdout.write("\r\n")

    return dict(sorted(photos.items()))


def create_pairs(types: list, folders: list, path: str, num_of_ppl_testing_set: int, num_of_ppl_validation_set: int):
    photos = defaultdict(list)
    augmented_photos = defaultdict(list)

    for type in types:
        print(f"{type = }")
        for folder in folders:
            print(f"{folder = }")

            if "augmented" not in folder:
                photos |= load_photo_names(type, folder, path)
            else:
                augmented_photos |= load_photo_names(
                    type, folder, path
                )

    print("Splitting loaded photos into train, test, valid groups.")
    train, test, valid = get_train_test_valid(
        operate_on=[
            add_same(photos), 
            add_different(photos)
            ],
        testing_num_of_people=num_of_ppl_testing_set,
        validation_num_of_people=num_of_ppl_validation_set
    )

    train2, _, _ = get_train_test_valid(
        operate_on=[
            add_same(photos, augmented_photos),
            add_different(photos, augmented_photos)
        ],
        testing_num_of_people=num_of_ppl_testing_set,
        validation_num_of_people=num_of_ppl_validation_set
    )
    train.extend(train2)

    print(f"{len(train) = }, {len(test) = }, {len(valid) = }")

    train = shuffle(train, random_state=1)
    test = shuffle(test, random_state=1)
    valid = shuffle(valid, random_state=1)

    with open(f"{path}/pairs_train.csv", 'w') as csv_train, \
        open(f"{path}/pairs_test.csv", 'w') as csv_test, \
        open(f"{path}/pairs_valid.csv", 'w') as csv_valid, \
        open(f"{path}/pairs_cross.csv", 'w') as csv_cross, \
        open(f"{path}/pairs_all.csv", 'w') as csv_all:
            csv.writer(csv_train, delimiter=";", lineterminator="\n").writerows(train)            
            csv.writer(csv_test, delimiter=";", lineterminator="\n").writerows(test)
            csv.writer(csv_valid, delimiter=";", lineterminator="\n").writerows(valid)
            csv.writer(csv_cross, delimiter=";", lineterminator="\n").writerows(train)
            csv.writer(csv_cross, delimiter=";", lineterminator="\n").writerows(valid)
            csv.writer(csv_all, delimiter=";", lineterminator="\n").writerows(train)       
            csv.writer(csv_all, delimiter=";", lineterminator="\n").writerows(test)
            csv.writer(csv_all, delimiter=";", lineterminator="\n").writerows(valid)


if __name__ == "__main__":
    os.chdir("hand_writings/")
    create_pairs(types=["all"],
                 folders=['0', '0_augmented'],
                 path="Photos",
                 num_of_ppl_testing_set=60,
                 num_of_ppl_validation_set=30)
