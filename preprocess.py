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


def AugementData(audio_array, sample_rate, target_duration):
    duration = len(audio_array) / sample_rate

    if duration < target_duration:
        while duration < target_duration:
            audio_array = np.append(audio_array, audio_array)
            duration = len(audio_array) / sample_rate

    target_length = 4 * sample_rate
    audio_array = audio_array[0:target_length]

    return audio_array

def CreateSpectrograms(audio_path, img_save_path, FrameSize, HopSize, freq_scale, px_x, px_y, monitor_dpi):
    utils.clear_directory(img_save_path)
    file_list = os.listdir(audio_path)
    amount_files = len(file_list)
    images_finished = 0
    print("Generating " + str(amount_files) + " Spectrograms")

    # set output image size
    x_offset = 0 # needed cause matplotlib is weird, play around with value until it works
    fig = plt.figure(figsize=(px_x / (monitor_dpi + x_offset), px_y / monitor_dpi))

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
        utils.progress_bar(images_finished, amount_files)

    return


def GenerateArrays(train_csv, test_csv, img_path, px_x, px_y):
    df_train = pd.read_csv(train_csv)
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


def GenerateLabelArray(dataframe):
    arr = np.array([[]])
    for index, row in dataframe.iterrows():
        arr = np.append(arr, row["classID"])

    arr = arr.reshape(-1, 1)
    return arr


def GenerateImageArray(dataframe, file_list, img_path, px_x, px_y):
    amount_files = int(dataframe.size / 2)
    print("collecting data from " + str(amount_files) + " images" )

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
    return arr


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
