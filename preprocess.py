import os
import shutil
import scipy
import wavfile
import soundfile as sf
import numpy as np
from numpy import asarray
import librosa
from scipy.interpolate import griddata
from scipy import signal
import utils
from matplotlib import pyplot as plt
import pandas as pd
import PIL
from PIL import Image
import pywt
from pylab import *
import soundfile as sf

def CreateCWTScaleogram():
    # 1. Read Audio file
    audio_file = "res/audio/34050-7-0-0.wav"
    data, samplerate = librosa.load(audio_file)
    data = data / max(data)
    ob = sf.SoundFile(audio_file)
    print("Samplerate of audiofile: " + str(format(ob.samplerate)))

    scales = np.arange(1, 65) # range of scales
    wavelet = "morl"
    times = data[:max(scales)*500]
    times = times / max(times)
    print(times.shape)

    dt = 1 / samplerate # timestep difference
    frequencies = pywt.scale2frequency(wavelet, scales) / dt  # Get frequencies corresponding to scales


    coeffs, freqs = pywt.cwt(times, scales, wavelet)

    # create scalogram
    plt.imshow(coeffs, cmap='gray', aspect='auto')
    plt.ylabel('Scale')
    plt.xlabel('Time')
    plt.show()

    return


def CreateDWTScaleogram():
    # 1. Read Audio file
    audio_file = "res/audio/34050-7-0-0.wav"
    data, samplerate = librosa.load(audio_file)
    data = data / max(data)
    ob = sf.SoundFile(audio_file)
    print("Samplerate of audiofile: " + str(format(ob.samplerate)))

    # data = [0] * 32
    # data[0] = 1
    data = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]
    wavelet = "haar"
    max_level = pywt.dwt_max_level(len(data), wavelet)

    # Multilevel decomposition
    result_wavedec = pywt.wavedec(data, wavelet, level=max_level)
    n = len(result_wavedec)
    print("wavedec results:")
    print("cA: " + str(result_wavedec[0]))
    print("cD" + str(n-1) + ": " + str(result_wavedec[1]))
    print("cD" + str(n-2) + ": " + str(result_wavedec[2]))
    print("cD" + str(n-3) + ": " + str(result_wavedec[3]))
    print("cD" + str(n-4) + ": " + str(result_wavedec[4]))

    # Single level DWT
    cA, cD = pywt.dwt(data, wavelet)
    print("----------------------")
    print("dwt results:")
    print("cA: " + str(cA.shape))
    print(cA)
    print("------")
    print("cD: " + str(cD.shape))
    print(cD)


    return


def CollectLongFiles(original_path, target_path, min_duration):
    print("Looking for files longer then " + str(min_duration) + "sec")
    file_list = os.listdir(original_path)
    utils.clear_directory(target_path)

    durations = {"4.0": 0}

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


def DuplicateDataUntilDuration(audio_array, sample_rate, target_duration):
    duration = len(audio_array) / sample_rate

    if duration < target_duration:
        while duration < target_duration:
            audio_array = np.append(audio_array, audio_array)
            duration = len(audio_array) / sample_rate

    target_length = target_duration * sample_rate
    audio_array = audio_array[0:target_length]

    return audio_array

def FillWithSilenceUntilDuration(audio_array, sample_rate, target_duration):
    duration = len(audio_array) / sample_rate
    if duration < target_duration:
        target_samples = sample_rate * target_duration
        extra_samples = target_samples - len(audio_array)
        arr_type = audio_array.dtype

        zero_arr = np.zeros(extra_samples, arr_type)
        audio_array = np.append(audio_array, zero_arr)
    else:
        target_length = target_duration * sample_rate
        audio_array = audio_array[0:target_length]

    return audio_array


def CreateSTFTSpectrograms(audio_path, img_save_path, FrameSize, HopSize, freq_scale, px_x, px_y, monitor_dpi,
                           fill_mode="duplicate"):
    utils.clear_directory(img_save_path)
    file_list = os.listdir(audio_path)
    amount_files = len(file_list)
    images_finished = 0
    print("Generating " + str(amount_files) + " Spectrograms")

    # set output image size
    x_offset = 0  # needed because matplotlib is weird, play around with value until it works
    fig = plt.figure(figsize=(px_x / (monitor_dpi + x_offset), px_y / monitor_dpi))

    for file in file_list:
        file_path = audio_path + "/" + file

        audioArray, sampleRate = librosa.load(file_path)
        if fill_mode == "duplicate":
            audioArray = DuplicateDataUntilDuration(audioArray, sampleRate, 4)
        elif fill_mode == "silence":
            audioArray = FillWithSilenceUntilDuration(audioArray, sampleRate, 4)

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
