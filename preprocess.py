import os
import shutil
import soundfile as sf
import numpy as np
from numpy import asarray
import librosa
import utils
from matplotlib import pyplot as plt
import pandas as pd
import PIL
from PIL import Image


def CollectLongFiles(original_path, target_path, min_duration):
    print("Looking for files longer then " + str(min_duration) + "sec")
    file_list = os.listdir(original_path)
    utils.clear_directory(target_path)

    durations = {
        "4.0": 0
    }

    for file in file_list:
        filepath = original_path + "/" + file

        ob = sf.SoundFile(filepath)
        samplerate = format(ob.samplerate)
        duration = int(format(ob.frames)) / int(samplerate)
        if duration in durations:
            durations[duration] = durations[duration] + 1
        else:
            durations[duration] = 1

        if duration >= min_duration:
            target = target_path + "/" + file
            shutil.copyfile(filepath, target)

    print("copied all long audio files to target directory")
    utils.print_class_distribution(target_path)
    return


def PrintDurationInfo(durations):

    min_duration = 100
    max_duration = 0
    amount_4sec_clips = 0
    sum = 0
    amount_clips = 0

    for key in sorted(durations):
        if float(key) > max_duration:
            max_duration = float(key)
        if float(key) < min_duration:
            min_duration = float(key)

        sum = sum + (float(key) * durations[key])
        amount_clips += durations[key]

    mean_duration = round(sum / amount_clips, 5)

    print("------------------------------------------")
    print("Anzahl an 4sek clips: " + str(durations["4.0"]))
    print("Anzahl unterschiedlicher Zeiten: " + str(len(durations)))
    print("Längster Clip: " + str(max_duration))
    print("Kürzester Clip: " + str(min_duration))
    print("Durchschnittliche Länge: " + str(mean_duration))
    print("------------------------------------------")

    return

def AugementData(audio_array, sample_rate, target_duration):
    duration = len(audio_array) / sample_rate

    if duration < target_duration:
        while duration < target_duration:
            audio_array = np.append(audio_array, audio_array)
            duration = len(audio_array) / sample_rate

    target_length = 4 * sample_rate
    audio_array = audio_array[0:target_length]

    return audio_array

def CreateSpectrograms(audio_path, img_save_path, FrameSize, HopSize, freq_scale):
    print("generating spectrograms...")
    utils.clear_directory(img_save_path)
    file_list = os.listdir(audio_path)
    amount_files = len(file_list)
    images_finished = 0
    fig = plt.figure(figsize=(1.66, 1.67)) # will result in a 128x128 image

    for file in file_list:
        file_path = audio_path + "/" + file

        audioArray, sampleRate = librosa.load(file_path)
        audioArray = AugementData(audioArray, sampleRate, 4)

        # 1.) Extract Short-Time Fourier Transform
        short_audio = librosa.stft(audioArray, n_fft=FrameSize, hop_length=HopSize)

        # 2.) calculate spectrogram
        # 2.1 we currently have complex numbers (can't be visualized), square values to convert to floats
        Y_audio = np.abs(short_audio) ** 2  # Y value is the amplitude (because we did a fourier transform)

        # 2.2 convert amplitude values from linear to logarithmic scale
        Y_log_audio = librosa.power_to_db(Y_audio)

        utils.plot_spectrogram(audioFileName=file, Y=Y_log_audio, samplerate=sampleRate, frame_size=FrameSize,
            hop_size=HopSize, y_axis=freq_scale, save_image=True, save_path=img_save_path, show_plot=False, figure=fig)

        images_finished += 1
        print(str(images_finished) + "/" + str(amount_files))

    return


def GenerateArrays(train_path, test_path, csv_file):
    df = pd.read_csv(csv_file)

    file_list = os.listdir(train_path)
    train_labels = GenerateLabelArray(df, file_list)
    train_images = GenerateImageArray(df, file_list, train_path)
    print("Train labels shape: " + str(train_labels.shape))
    print("Train images shape: " + str(train_images.shape))

    file_list = os.listdir(test_path)
    test_labels = GenerateLabelArray(df, file_list)
    test_images = GenerateImageArray(df, file_list, test_path)
    print("Test labels shape: " + str(test_labels.shape))
    print("Test images shape: " + str(test_images.shape))
    print(test_labels.shape)

    return train_images, train_labels, test_images, test_labels


def GenerateLabelArray(dataframe, file_list):
    # go through csv, find classID for the file name and fill array with values
    print("Generating Labels Array...")
    arr = np.array([[]])
    for index, row in dataframe.iterrows():
        string_list = row["slice_file_name"].split(".")
        img_name = string_list[0] + ".png"

        if img_name in file_list:
            arr = np.append(arr, row["classID"])
    arr = arr.reshape(-1, 1)
    print("Labels Array finished")

    return arr


def GenerateImageArray(dataframe, file_list, img_path):
    print("Generating image data matrix...")
    amount_files = len(file_list)
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
            print(str(files_added) + "/" + str(amount_files))

    arr = arr.reshape(-1, 128, 128, 1)
    print(arr)
    print("finished image data matrix")
    return arr
