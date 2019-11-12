# This script reads the image folders and builds test/train sets from them
# and write them to files
# via parameters the goal is adjustable (color, depth, both)
import os
import random


# parameters
IMAGE_FOLDER_PATH = "/Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/images"
NAMES_PATH = "/Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/household.names"
OUTPUT_PATH = "/Users/demming/Studium/Masterarbeit/Dev/master-thesis-final/data/household_data/partitions"
goal = ["color", "rgb-depth"]
test_size = 0.2
instance_threshold = 300

print("IMAGE FOLDER: {}".format(IMAGE_FOLDER_PATH))
print("NAMES FILE: {}".format(IMAGE_FOLDER_PATH))
print("OUTPUT FOLDER: {}".format(IMAGE_FOLDER_PATH))
print("Goals: {}".format(goal))
print("Test size: {}".format(test_size))
print("### Starting file generation ###")


# get folders according to file list (maybe get classes from *.names)
with open(NAMES_PATH, "r") as f:
    labels = f.read().splitlines()

# read files lists
all_files = []
test_set = []
train_set = []
cls_counter = 0
old_test_count = 0
old_train_count = 0
for g in goal:
    for cls in labels:
        full_path = os.path.join(IMAGE_FOLDER_PATH, cls, g)
        cls_file_list = list(filter(lambda s: ".png" == s[-4:], os.listdir(full_path)))
        cls_file_list = list(sorted(cls_file_list))
        # first do the partitioning
        last_timestamp = -1
        cls_parted_list = []
        for pic in cls_file_list:
            new_timestamp = int(pic.split("/")[-1][:-4])
            path_prefix = os.path.join(IMAGE_FOLDER_PATH, cls, g)

            if abs(new_timestamp - last_timestamp) > instance_threshold:
                # new instance, create new instance list in list
                cls_parted_list.append([os.path.join(path_prefix, pic)])
            else:
                # append pic to last list in list
                cls_parted_list[-1].append(os.path.join(path_prefix, pic))
            last_timestamp = new_timestamp
        test_set.extend(cls_parted_list.pop(0))
        for l in cls_parted_list:
            random.shuffle(l)
            test_set.extend(l[:int(test_size*len(l))])
            train_set.extend(l[int(test_size*len(l)):])
        cls_counter += 1
        print("Added {} pics from class {} to test".format(len(test_set) - old_test_count, cls))
        print("Added {} pics from class {} to train".format(len(train_set) - old_train_count, cls))
        old_test_count = len(test_set)
        old_train_count = len(train_set)

print("### Finished process ###")
print("Read {} pics into test".format(len(test_set)))
print("Read {} pics into train".format(len(train_set)))
print("### Writing to files ###".format(len(train_set)))

# finish
with open(os.path.join(OUTPUT_PATH, "household_train_{}.txt".format("_".join(goal))), "w+") as f:
    for ele in train_set:
        f.write("{}\n".format(ele))

with open(os.path.join(OUTPUT_PATH, "household_test_{}.txt".format("_".join(goal))), "w+") as f:
    for ele in test_set:
        f.write("{}\n".format(ele))

print("### Files written successfully ###")
print("### Exiting ###")
