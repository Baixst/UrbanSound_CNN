import os
import csv
import random
import pandas as pd

def load_file_names(path):
    file_list = os.listdir(path)
    return file_list


def split_csv(files, main_csv, train_csv, test_csv, train_percentage):
    total = len(files)
    counter = 0

    # files list will be compared to the main csv, which has the wav files listed
    while counter < total:
        string_list = files[counter].split(".")
        new_file_name = string_list[0] + ".wav"
        files[counter] = new_file_name
        counter += 1

    random.shuffle(files)
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

    # shuffel rows of csv files
    df = pd.read_csv(train_csv)
    shuffled_df = df.sample(frac=1)
    shuffled_df.to_csv(train_csv, index=False)

    df = pd.read_csv(test_csv)
    shuffled_df = df.sample(frac=1)
    shuffled_df.to_csv(test_csv, index=False)

    return


def create_cross_val_csv(files, main_csv, result_csv):
    random.shuffle(files)
    total = len(files)
    print(total)
    resultCSV = open(result_csv, 'w+', encoding='UTF8', newline='')
    writer = csv.writer(resultCSV)

    # write header and filenames + classID to csv files
    csv_header = ["slice_file_name", "classID"]
    writer.writerow(csv_header)

    df = pd.read_csv(main_csv)
    for index, row in df.iterrows():
        string_list = row["slice_file_name"].split(".")
        file_name = string_list[0] + ".png"
        if file_name in files:
            values = [row["slice_file_name"], row["classID"]]
            writer.writerow(values)
    resultCSV.close()

    # shuffel csv rows
    df = pd.read_csv(result_csv)
    shuffled_df = df.sample(frac=1)
    shuffled_df.to_csv(result_csv, index=False)

    return


def get_audio_name(imagename):
    strings = imagename.split(".")
    audioname = strings[0] + ".wav"
    return audioname


def create_validation_dataset(data, labels, val_percentage):

    val_total = int((len(data) / 100) * val_percentage)

    data_val = data[:val_total]
    partial_data_train = data[val_total:]
    labels_val = labels[:val_total]
    partial_labels_train = labels[val_total:]
    print("Using " + str(len(labels_val)) + " files for validation")

    return data_val, partial_data_train, labels_val, partial_labels_train
