import os
import random
import shutil
import utils

def load_image_names(path):
    file_list = os.listdir(path)
    return file_list


def split_data_in_two(dataset, train_percentage, origin_path, test_set_path, train_set_path):
    utils.clear_directory(train_set_path)
    utils.clear_directory(test_set_path)
    total = len(dataset)
    train_total = int(total / 100 * train_percentage)

    random.shuffle(dataset)

    train_data = []
    test_data = []
    counter = 0

    while counter < train_total:
        train_data.append(dataset[counter])
        counter += 1

    while counter < total:
        test_data.append(dataset[counter])
        counter += 1

    print("Copying " + str(len(train_data)) + " files to trainset...")
    for file in train_data:
        filepath = origin_path + "/" + file
        targetpath = train_set_path + "/" + file
        shutil.copyfile(filepath, targetpath)

    print("Copying " + str(len(test_data)) + " files to trainset...")
    for file in test_data:
        filepath = origin_path + "/" + file
        targetpath = test_set_path + "/" + file
        shutil.copyfile(filepath, targetpath)

    return


def get_audio_name(imagename):
    strings = imagename.split(".")
    audioname = strings[0] + ".wav"
    return audioname
