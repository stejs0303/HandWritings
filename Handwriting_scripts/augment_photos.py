import random as rn
import numpy as np
import cv2 as cv
import time
import sys
import os

from enum import Enum

from config import WIDTH, HEIGHT, INVERT


class Orientation(Enum):
    VERT_HORI = 0
    VERT = 1
    HORI = 2


def translate(image, pos: tuple):
    bordervalue = 0 if INVERT else 255
    return cv.warpAffine(src=image,
                         dsize=(image.shape[1], image.shape[0]),
                         M=np.array([[1, 0, pos[0]], [0, 1, pos[1]]],
                                    np.float32),
                         flags=cv.INTER_AREA,
                         borderMode=cv.BORDER_CONSTANT,
                         borderValue=bordervalue)


def rotate(image, angle: float):
    bordervalue = 0 if INVERT else 255
    return cv.warpAffine(src=image,
                         dsize=(image.shape[1], image.shape[0]),
                         M=cv.getRotationMatrix2D((image.shape[0]/2,
                                                   image.shape[1]/2),
                                                  angle=angle,
                                                  scale=1),
                         flags=cv.INTER_AREA,
                         borderMode=cv.BORDER_CONSTANT,
                         borderValue=bordervalue)


def thicken(image, kernel: tuple = (3, 3), iterations: int = 1):
    return cv.erode(src=image,
                    kernel=kernel,
                    anchor=(-1, -1),
                    iterations=iterations,
                    borderType=cv.BORDER_DEFAULT)


def degrade(image, kernel: tuple = (3, 3), iterations: int = 1):
    return cv.dilate(src=image,
                     kernel=kernel,
                     anchor=(-1, -1),
                     iterations=iterations,
                     borderType=cv.BORDER_DEFAULT)


def random_erase(image, lines_count: int, ori: Orientation = None):
    width_y, width_x = image.shape
    erase_values = 0 if INVERT else 255
    if ori in (Orientation.VERT, Orientation.VERT_HORI):
        xs = [(rn.randint(0, width_x//2),
               rn.randint(width_x//50, width_x//20)/2,
               rn.randint(0, 1)) for _ in range(lines_count)]

        for x, erase_width, rnd in xs:
            if rnd == 0:
                image[:, round(width_x/2-x-erase_width): round(width_x/2-x+erase_width)] = erase_values
            else:
                image[:, round(width_x/2+x-erase_width): round(width_x/2+x+erase_width)] = erase_values

    if ori in (Orientation.HORI, Orientation.VERT_HORI):
        ys = [(rn.randint(0, width_y//3),
               rn.randint(width_y//30, width_y//15)/2,
               rn.randint(0, 1)) for _ in range(lines_count)]

        for y, erase_width, rnd in ys:
            if rnd == 0:
                image[round(width_y/2-y-erase_width): round(width_y/2-y+erase_width), :] = erase_values
            else:
                image[round(width_y/2+y-erase_width): round(width_y/2+y+erase_width), :] = erase_values

    return image


def add_noise(image, mean, stddev):
    noise = cv.randn(np.array(image), mean=mean, stddev=stddev)
    if not INVERT:
        image = cv.bitwise_not(image)
    image = cv.add(noise, image)
    if not INVERT:
        image = cv.bitwise_not(image)
    return image


def get_random_values(deg_range: float, x_range: int, y_range: int):
    return (rn.randint(0, deg_range) - deg_range/2), \
           (rn.randint(0, x_range) - x_range/2), \
           (rn.randint(0, y_range) - y_range/2)


def generate_augmented_photos(types: list, folders: list, path: str, random_erase_lines: int = 2, degrade_itter: int = 1, early_stop: int = -1):
    rn.seed(time.time())

    print("Starting augmentation.")
    for folder_type in types:
        for folder in folders:
            listing = sorted(
                os.listdir(os.path.dirname(
                    f"{path}/paragraphs/{folder_type}/{folder}/"
                ))
            )

            for idx, name in enumerate(listing, 1):
                sys.stdout.write(f"\rProcessing photo {idx}/{len(listing)}.")
                if idx == early_stop:
                    break

                img = cv.imread(
                    f"{path}/paragraphs/{folder_type}/{folder}/{name}", cv.IMREAD_GRAYSCALE
                )

                rot, x, y = get_random_values(12, 12, 10)

                img = rotate(img, rot)
                img = translate(img, (x, y))

                if INVERT:
                    img = thicken(img, (5, 5), degrade_itter)
                else:
                    img = degrade(img, (5, 5), degrade_itter)

                #img = add_noise(img, 0, 45)
                img = random_erase(
                    img, random_erase_lines, Orientation.VERT_HORI
                )

                name_stripped = name.strip(".png")
                cv.imwrite(
                    f"{path}/paragraphs/{folder_type}/{folder}_augmented/{name_stripped}_augmented.png", img
                )

            print(
                f"\n{folder_type.capitalize()} photos from folder {folder} have been augmented."
            )


def generate_downscaled_photos(path_read: str, path_write: str, folders: list):
    print("Starting downscaling.")
    for folder in folders:
        
        listing = sorted(os.listdir(os.path.dirname(f"{path_read}/{folder}/")))

        for idx, name in enumerate(listing, 1):
            sys.stdout.write(f"\rProcessing photos in folder {folder}: {idx}/{len(listing)}.")

            img = cv.imread(f"{path_read}/{folder}/{name}")
            img = cv.resize(img, (WIDTH, HEIGHT), fx=0, fy=0, interpolation=cv.INTER_AREA)

            if INVERT:
                img = cv.bitwise_not(img)

            cv.imwrite(f"{path_write}/{folder}/{name}", img)
    
        print(f"\nFinished folder {folder}.")

    print(f"\nPhotos have been downscaled.")


if __name__ == "__main__":
    os.chdir("hand_writings/")
    MADE_CHANGES = True

    if MADE_CHANGES:
        generate_downscaled_photos(
            path_read="Photos/paragraphs/original/",
            path_write="Photos/paragraphs/all/",
            folders=['0']
        )

    generate_augmented_photos(
        types=["all"],
        folders=['0'],
        path="Photos",
        random_erase_lines=2,   # 750 - 2, 1000 - 3, 1250 - 4
        degrade_itter=2         # 750 - 2, 1000 - 3, 1250 - 5
    )
