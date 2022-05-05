import csv
import os
import random
import shutil
import utils
import pandas as pd

def load_file_names(path):
    file_list = os.listdir(path)
    return file_list


def split_csv(files, main_csv, train_csv, test_csv, train_percentage):
    random.shuffle(files)
    total = len(files)
    train_total = int(total / 100 * train_percentage)
    train_files, test_files = [], []
    counter = 0

    # lists with file names
    while counter < train_total:
        train_files.append(files[counter])
        counter += 1
    while counter < total:
        test_files.append(files[counter])
        counter += 1

    # open and clear csv files
    trainCSV = open(train_csv, 'w+', encoding='UTF8', newline='')
    testCSV = open(test_csv, 'w+', encoding='UTF8', newline='')
    train_writer = csv.writer(trainCSV)
    test_writer = csv.writer(testCSV)

    # write header and filenames + classID to csv files
    csv_header = ["slice_file_name", "classID"]
    train_writer.writerow(csv_header)
    test_writer.writerow(csv_header)

    df = pd.read_csv(main_csv)
    for index, row in df.iterrows():
        if row["slice_file_name"] in train_files:
            values = [row["slice_file_name"], row["classID"]]
            train_writer.writerow(values)
        elif row["slice_file_name"] in test_files:
            values = [row["slice_file_name"], row["classID"]]
            test_writer.writerow(values)

    trainCSV.close()
    testCSV.close()
    return


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
