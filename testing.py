import os
import csv
import pandas as pd
import random
import pydub
from pydub import AudioSegment
import shutil
import librosa
import librosa.display
import soundfile as sf
from matplotlib import pyplot as plt
import utils
import numpy as np
import preprocess
from sklearn.preprocessing import MinMaxScaler


def findMonoClip(audio_path, target):
    file_list = os.listdir(audio_path)
    counter = 0

    for file in file_list:
        file_path = audio_path + "/" + file
        ob = sf.SoundFile(file_path)
        channel = format(ob.channels)
        if str(channel) == "1":
            counter += 1
            print("found mono clip")
            target = target + "/" + file
            shutil.copyfile(file_path, target)
            break
    print(counter)
    return

def findClipWithSamplerate(sr, audio_path, target):
    file_list = os.listdir(audio_path)
    counter = 0

    for file in file_list:
        file_path = audio_path + "/" + file
        ob = sf.SoundFile(file_path)
        channel = format(ob.samplerate)
        if str(channel) == str(sr):
            counter += 1
            print("found clip with target samplerate")
            target = target + "/" + file
            shutil.copyfile(file_path, target)
            break
    print(counter)
    return


def convertMonoToStereo(file, new_file):
    sound = AudioSegment.from_wav(file)
    sound = sound.set_channels(2)
    sound.export(new_file, format="wav")
    return

def plotSpectrogramWithSamplerate(samplerate, file, y_axis, audio_arr=[0]):

    px_x, px_y, monitor_dpi, x_offset = 1024, 512, 77, 0
    figure = plt.figure(figsize=(px_x / (monitor_dpi + x_offset), px_y / monitor_dpi))

    if len(audio_arr) < 2:
        audioArray, sampleRate = librosa.load(file, sr=samplerate)
        print(sampleRate)
    else:
        audioArray = audio_arr
        sampleRate = 22050

    # 1.) Extract Short-Time Fourier Transform
    short_audio = librosa.stft(audioArray, n_fft=1024, hop_length=256)

    # 2.) calculate spectrogram
    # 2.1 we currently have complex numbers (can't be visualized), square values to convert to floats
    Y_audio = np.abs(short_audio) ** 2  # Y value is the amplitude (because we did a fourier transform)

    # 2.2 convert amplitude values from linear to logarithmic scale
    Y_log_audio = librosa.power_to_db(Y_audio)

    librosa.display.specshow(Y_log_audio, sr=sampleRate, hop_length=256, x_axis="time", y_axis=y_axis)

    cmap = plt.get_cmap("gray")
    plt.set_cmap(cmap)
    plt.axis('off')

    image_name = "res/test2/stft_spec_" + str(sampleRate) + ".png"
    figure.savefig(image_name, bbox_inches='tight', pad_inches=0)
    plt.clf()
    figure.clear()
    return


# findMonoClip("res/audio", "res/test")
# convertMonoToStereo("res/test/mono.wav", "res/test/stereo.wav")
# findClipWithSamplerate(8000, "res/audio", "res/test")

file_path = "res/test/sr8000.wav"
savePath = "res/test2"
# plotSpectrogramWithSamplerate(samplerate=8000, file=file_path, y_axis="log")

# preprocess.CreateDWTScaleogram()

# audioArray, sampleRate = librosa.load("res/test/short_clip.wav", sr=22050)
# audioArray2 = preprocess.FillWithSilenceUntilDuration(audioArray, sampleRate, 4)
# plotSpectrogramWithSamplerate(samplerate=sampleRate, file="res/test/short_clip.wav", y_axis="log", audio_arr=audioArray2)


# -------------------------------------------------------------------------------------
def CreateDefinedFoldCSVs_test_val():
    df_main = pd.read_csv("metadata/UrbanSound8K.csv")

    for i in range(1, 11):
        # test files
        result_csv = "metadata/def_folds/test" + str(i) + ".csv"
        csvFile = open(result_csv, 'w+', encoding='UTF8', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(["slice_file_name", "classID", "fold"])

        for index, row in df_main.iterrows():
            if row["fold"] == i:
                values = [row["slice_file_name"], row["classID"], row["fold"]]
                writer.writerow(values)
        csvFile.close()

        # shuffel csv rows
        df = pd.read_csv(result_csv)
        shuffled_df = df.sample(frac=1)
        shuffled_df.to_csv(result_csv, index=False)

        # validation files
        result_csv = "metadata/def_folds/val" + str(i) + ".csv"
        csvFile = open(result_csv, 'w+', encoding='UTF8', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(["slice_file_name", "classID", "fold"])

        j = i+1
        if j == 11:
            j = 1

        for index, row in df_main.iterrows():
            if row["fold"] == j:
                values = [row["slice_file_name"], row["classID"], row["fold"]]
                writer.writerow(values)
        csvFile.close()

        # shuffel csv rows
        df = pd.read_csv(result_csv)
        shuffled_df = df.sample(frac=1)
        shuffled_df.to_csv(result_csv, index=False)

    return


def CreateDefinedFoldCSVs_train():
    df_main = pd.read_csv("metadata/UrbanSound8K.csv")

    for i in range(1, 11):

        if i == 1:
            folds = [3,4,5,6,7,8,9,10]
        if i == 2:
            folds = [1,4,5,6,7,8,9,10]
        if i == 3:
            folds = [1,2,5,6,7,8,9,10]
        if i == 4:
            folds = [1,2,3,6,7,8,9,10]
        if i == 5:
            folds = [1,2,3,4,7,8,9,10]
        if i == 6:
            folds = [1,2,3,4,5,8,9,10]
        if i == 7:
            folds = [1,2,3,4,5,6,9,10]
        if i == 8:
            folds = [1,2,3,4,5,6,7,10]
        if i == 9:
            folds = [1,2,3,4,5,6,7,8]
        if i == 10:
            folds = [2,3,4,5,6,7,8,9]

        # test files
        result_csv = "metadata/def_folds/train" + str(i) + ".csv"
        csvFile = open(result_csv, 'w+', encoding='UTF8', newline='')
        writer = csv.writer(csvFile)
        writer.writerow(["slice_file_name", "classID", "fold"])

        for index, row in df_main.iterrows():
            if row["fold"] in folds:
                values = [row["slice_file_name"], row["classID"], row["fold"]]
                writer.writerow(values)
        csvFile.close()

        # shuffel csv rows
        df = pd.read_csv(result_csv)
        shuffled_df = df.sample(frac=1)
        shuffled_df.to_csv(result_csv, index=False)

    return


# CreateDefinedFoldCSVs_test_val()
# CreateDefinedFoldCSVs_train()

# -------------------------------------------------------------------------------------

def plot_mfccs(audiofile):
    signal, sr = librosa.load(audiofile)
    signal = preprocess.FillWithSilenceUntilDuration(signal, sr, 4)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(signal, n_mfcc=24, sr=sr)

    # Calculate delta and delta2 MFCCs
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # Concatenate deltas and mfccs
    comp_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))

    # Normalize
    delta2_mfccs = delta2_mfccs / np.amax(delta2_mfccs)

    # Visualise MFCCs
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(delta2_mfccs, x_axis="time", sr=sr)
    plt.colorbar(format="%+2f")
    plt.show()
    return


plot_mfccs("res/audio/518-4-0-0.wav")

# -------------------------------------------------------------------------------------

# preprocess.CreateSTFTSpectrograms("res/test", "res/test2", 1024, 256, "mel", 256, 256, 77,
#                           fill_mode="duplicate")

def plot_mel_spectrogram():
    audiofile = "res/test/99179-9-0-12.wav"
    y, sr = librosa.load(audiofile)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()

    return


# plot_mel_spectrogram()