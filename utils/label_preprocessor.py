# This file provides functionality to convert a label.xml file to the needed input structure, where the label files
# are located in the same path as the image files, with the "images" renamed to "labels" however
from xml.dom import minidom
import os

# example: /Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/labels
LABEL_PATH = "/Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/labels"
# example: /Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/images
IMAGES_PATH = "/Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/images"
# example: /Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/household.names
NAMES_PATH = "/Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/household.names"
XML_PATH = "/Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/household_dataset.xml"
# import label file first
label_xml = minidom.parse(XML_PATH)

if not os.path.exists(LABEL_PATH):
    os.mkdir(LABEL_PATH)

# /Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/images/book/color
# ->
# /Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/labels/book/color
# &&
# /Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/images/book/rgb-depth
# ->
# /Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/labels/book/rgb-depth

# import label names & indices
with open(NAMES_PATH, "r") as f:
    label_names = f.read().splitlines()

images = label_xml.getElementsByTagName("image")
for img in images:
    img_color_path = img.attributes["name"].value  # book/color/123456.png
    img_rgb_depth_path = img_color_path.replace("color", "rgb-depth")  # book/rgb-depth/123456.png
    label_color_path = img_color_path.replace("images", "labels")
    label_rgb_depth_path = img_rgb_depth_path.replace("images", "labels")

    # Problem: the path does not exist (only the labels)
    # LABEL_PATH + img_{color, rgb_depth}_path = full path to file
    img_color_path_parts = img_color_path.split("/")[:-1]
    # class path
    if not os.path.exists(os.path.join(LABEL_PATH, img_color_path_parts[0])):
        os.mkdir(os.path.join(LABEL_PATH, img_color_path_parts[0]))

    # color/rgb-depth path
    if not os.path.exists(os.path.join(LABEL_PATH, img_color_path_parts[0], "color")):
        os.mkdir(os.path.join(LABEL_PATH, img_color_path_parts[0], "color"))
    if not os.path.exists(os.path.join(LABEL_PATH, img_color_path_parts[0], "rgb-depth")):
        os.mkdir(os.path.join(LABEL_PATH, img_color_path_parts[0], "rgb-depth"))

    boxes = img.getElementsByTagName("box")
    boxes_out = []
    for box in boxes:
        label_name = box.attributes["label"].value
        if label_name not in label_names:
            continue
        label_no = label_names.index(label_name)
        xtl = float(box.attributes["xtl"].value)
        ytl = float(box.attributes["ytl"].value)
        xbr = float(box.attributes["xbr"].value)
        ybr = float(box.attributes["ybr"].value)
        # convert labels cls, x, y, w, h
        x = (xtl + xbr) / 2
        y = (ytl + ybr) / 2
        w = xbr - xtl
        h = ybr - ytl
        # normalize
        img_width = int(img.attributes["width"].value)
        img_height = int(img.attributes["height"].value)
        xn = x / img_width
        yn = y / img_height
        wn = w / img_width
        hn = h / img_height

        # add to list of labels
        boxes_out.append(("{} "*5).format(label_no, xn, yn, wn, hn))

    # write list of lists to file
    with open(os.path.join(LABEL_PATH, label_color_path).replace(".png", ".txt"), "w") as f:
        f.writelines(boxes_out)
    with open(os.path.join(LABEL_PATH, label_rgb_depth_path).replace(".png", ".txt"), "w") as f:
        f.writelines(boxes_out)

# DO NOT FORGET TO CREATE TXT FILES FOR TRAIN AND TEST!!
