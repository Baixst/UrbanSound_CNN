import os
import librosa
import utils
import pandas as pd
from PIL import Image
from pylab import *

def GenerateArraysDefCrossVal(index, csv_path, img_path, px_x, px_y):
    data_csv = csv_path + "/train" + str(index) + ".csv"
    X_train, y_train = GenerateArraysCrossVal(data_csv, img_path, px_x, px_y)

    data_csv = csv_path + "/test" + str(index) + ".csv"
    X_test, y_test = GenerateArraysCrossVal(data_csv, img_path, px_x, px_y)

    return X_train, y_train, X_test, y_test


def GenerateArraysCrossVal(data_csv, img_path, px_x, px_y):
    df = pd.read_csv(data_csv)
    file_list = os.listdir(img_path)

    labels = GenerateLabelArray(df)
    image_data = GenerateImageArray(df, file_list, img_path, px_x, px_y)
    return image_data, labels


def GenerateArrays_STFT(train_csv, test_csv, img_path, px_x, px_y):
    df_train = pd.read_csv(train_csv)
    print(df_train)
    df_test = pd.read_csv(test_csv)
    file_list = os.listdir(img_path)

    print("Generating Train Labels Array...")
    train_labels = GenerateLabelArray(df_train)
    print("Generating Test Labels Array...")
    test_labels = GenerateLabelArray(df_test)

    print("Generating train image data matrix...")
    train_images = GenerateImageArray(df_train, file_list, img_path, px_x, px_y)
    print("Generating test image data matrix...")
    test_images = GenerateImageArray(df_test, file_list, img_path, px_x, px_y)

    print("Train labels shape: " + str(train_labels.shape))
    print("Train images shape: " + str(train_images.shape))
    print("Test labels shape: " + str(test_labels.shape))
    print("Test images shape: " + str(test_images.shape))
    return train_images, train_labels, test_images, test_labels


def GenerateArrays_DWT(train_csv, test_csv, features_csv):
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    df_features = pd.read_csv(features_csv)

    print("Generating Train Labels Array...")
    train_labels = GenerateLabelArray(df_train)
    print("Generating Test Labels Array...")
    test_labels = GenerateLabelArray(df_test)

    print("Generating train feature data matrix...")
    train_images = GenerateFeaturesArray(df_train, df_features)
    print("Generating test feature data matrix...")
    test_images = GenerateFeaturesArray(df_test, df_features)

    print("Train labels shape: " + str(train_labels.shape))
    print("Train features shape: " + str(train_images.shape))
    print("Test labels shape: " + str(test_labels.shape))
    print("Test features shape: " + str(test_images.shape))
    return train_images, train_labels, test_images, test_labels


def GenerateArrays_Raw(train_csv, test_csv, audio_path, duration, sr):
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    file_list = os.listdir(audio_path)

    print("Generating Train Labels Array...")
    train_labels = GenerateLabelArray(df_train)
    print("Generating Test Labels Array...")
    test_labels = GenerateLabelArray(df_test)

    print("Generating train signals matrix...")
    train_data = GenerateRawSignalArray(df_train, file_list, audio_path, duration, sr)
    print("Generating test signals matrix...")
    test_data = GenerateRawSignalArray(df_test, file_list, audio_path, duration, sr)

    print("Train labels shape: " + str(train_labels.shape))
    print("Train data shape: " + str(train_data.shape))
    print("Test labels shape: " + str(test_labels.shape))
    print("Test data shape: " + str(test_data.shape))
    return train_data, train_labels, test_data, test_labels


def GenerateLabelArray(dataframe):
    arr = np.array([[]])
    for index, row in dataframe.iterrows():
        arr = np.append(arr, row["classID"])

    arr = arr.reshape(-1, 1)
    return arr


def GenerateImageArray(dataframe, file_list, img_path, px_x, px_y):
    amount_files = len(dataframe.index)
    print("collecting data from " + str(amount_files) + " images")

    files_added = 0
    arr = np.array([[[]]])
    for index, row in dataframe.iterrows():
        string_list = row["slice_file_name"].split(".")
        img_name = string_list[0] + ".png"

        if img_name in file_list:
            img_name = img_path + "/" + img_name
            image = Image.open(img_name).convert("L")
            data = asarray(image)
            arr = np.append(arr, data)

            files_added += 1
            utils.progress_bar(current=files_added, total=amount_files)

    arr = arr.reshape(-1, px_x, px_y, 1)
    print()
    return arr


def GenerateFeaturesArray(df_file_and_label, df_features):
    amount_files = len(df_file_and_label.index)
    print("collecting data from " + str(amount_files) + " lines")

    amount_features = len(df_features.axes[1])-1
    files_with_features = df_features.audio_file.to_list()

    files_added = 0
    arr = np.array([[]])
    for index, row in df_file_and_label.iterrows():
        if row["slice_file_name"] in files_with_features:
            # look for row with the correct audio file name and write that row to a list
            feat_list = df_features.loc[df_features['audio_file'] == row["slice_file_name"]].values.flatten().tolist()

            # the list contains the audio file name on position 0, which needs to be removed
            feat_list = feat_list[1:]
            arr = np.append(arr, feat_list)

            files_added += 1
            utils.progress_bar(current=files_added, total=amount_files)

    # Normalize Data
    # arr = feat.sigmoid(arr)

    arr = arr.reshape(-1, amount_features)
    return arr


def GenerateRawSignalArray(dataframe, file_list, audio_path, duration, samplerate):
    amount_files = len(dataframe.index)
    in_shape = int(duration * samplerate)
    print("collecting data from " + str(amount_files) + " audio files" )

    files_added = 0
    arr = np.array([[]])
    for index, row in dataframe.iterrows():
        audio_name = row["slice_file_name"]

        if audio_name in file_list:
            audio_name = audio_path + "/" + audio_name
            data, sr = librosa.load(audio_name, sr=samplerate)
            arr = np.append(arr, data)

            files_added += 1
            utils.progress_bar(current=files_added, total=amount_files)

    arr = arr.reshape(-1, in_shape, 1)
    return arr
