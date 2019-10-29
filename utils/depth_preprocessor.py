# This file provides functionality to convert all data inside the {path}/{class}/depth folder into RGB pictures
# inside the {path}/{class}/depth-rgb folder

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os


def convert_image(_img_path, cmap_name):
    _img = cv2.imread(_img_path, cv2.IMREAD_GRAYSCALE)
    _img = _img.astype(np.float32)

    # normalize between min und 2nd-max
    img_min = np.min(_img)
    # all invalid values = -1
    _img[np.where(_img == 255)] = -1
    # do normalization steps

    img_max = np.max(_img)
    img_range = img_max - img_min
    _img = np.multiply(np.divide(_img - img_min, img_range), 0.9)

    # return invalid values back to 255
    _img[np.where(_img < 0)] = 1.0

    # jet color map
    _cmap = plt.get_cmap(cmap_name)

    # RGB BGR
    _img = _cmap(_img)[:, :, :3]  # discard last layer because it's just 1.0 anyway
    # however, we want RGB values between 0-255
    _img = np.multiply(_img, 255)
    return _img


# define path
INPUT_PATH = "/Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data"
OUTPUT_PATH = ""
if len(OUTPUT_PATH) < 1:
    OUTPUT_PATH = INPUT_PATH

# define classes -> simply all classes in path
dirs = os.listdir(INPUT_PATH)
dirs = list(filter(lambda _dir: os.path.isdir(os.path.join(INPUT_PATH, _dir)), dirs))
print("Reading files from {} in {} classes".format(INPUT_PATH, len(dirs)))

file_count = 0
valid_file_suffices = ['.jpg', '.png']

# load images
# every class
for cls in dirs:
    # every picture in dir
    cls_path = os.path.join(INPUT_PATH, cls, "depth")
    cls_output_path = os.path.join(OUTPUT_PATH, cls, "rgb-depth")
    if not os.path.exists(cls_output_path):
        os.mkdir(cls_output_path)
    imgs = os.listdir(cls_path)
    print("Reading {} files from {}".format(len(imgs), cls_path))
    # convert images
    for img_name in imgs:
        file_count += 1
        img_path = os.path.join(cls_path, img_name)
        img = convert_image(img_path, 'jet')
        save_path = os.path.join(cls_output_path, img_name)
        cv2.imwrite(save_path, img)

print("{} files converted".format(file_count))
