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
import scipy.signal as scsi


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

    cmap = plt.get_cmap("magma")
    plt.set_cmap(cmap)
    plt.axis('off')

    image_name = "res/test/stft_spec_" + str(sampleRate) + ".png"
    figure.savefig(image_name, bbox_inches='tight', pad_inches=0)
    plt.clf()
    figure.clear()
    return


# findMonoClip("res/audio", "res/test")
# convertMonoToStereo("res/test/mono.wav", "res/test/stereo.wav")
# findClipWithSamplerate(8000, "res/audio", "res/test")

def getLogPsdSegments(segments, fs, file):
    """
    Parameters
    ----------
    segments : array [n_segments x samples_per_segment]
        segment matrix
        each line represent one segment
        each column is a sampling point out of the input signal
    fs : int
        used sampling frequency on the signal

    Returns
    -------
    freq : array
        frequency over the index of psd_segments
    psd_segments : array [n_segments x ~samples_per_segment/2]
        power over the frequencies given in freq
    """
    audioArray, sampleRate = librosa.load(file, sr=fs)
    freq, psdSegments = scsi.periodogram(segments, fs=fs)
    psdSegments = 10 * np.log10(psdSegments + 1);
    return freq, psdSegments


def plotPSD(file):
    audioArray, sampleRate = librosa.load(file, sr=44100)
    freqs, psd = scsi.welch(audioArray, sampleRate, nperseg=1024, noverlap=512)

    plt.figure(figsize=(12, 8))
    plt.semilogy(freqs, psd)
    plt.title('Power spectral density')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.show()
    return


# plotPSD("res/audio_4sec/157695-3-0-2.wav")


# file_path = "res/audio/181725-3-0-16.wav"
# savePath = "res/test"
# plotSpectrogramWithSamplerate(samplerate=22050, file=file_path, y_axis="mel")
# plotPSD(file_path)
# getLogPsdSegments(fs=22050, file=file_path)




# preprocess.CreateDWTScaleogram()

# audioArray, sampleRate = librosa.load("res/test/short_clip.wav", sr=22050)
# audioArray2 = preprocess.FillWithSilenceUntilDuration(audioArray, sampleRate, 4)
# plotSpectrogramWithSamplerate(samplerate=sampleRate, file="res/test/short_clip.wav", y_axis="log", audio_arr=audioArray2)


# -------------------------------------------------------------------------------------
def CreateDefinedFoldCSVs_test():
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

    return


def CreateDefinedFoldCSVs_train():
    df_main = pd.read_csv("metadata/UrbanSound8K.csv")

    for i in range(1, 11):

        if i == 1:
            folds = [2,3,4,5,6,7,8,9,10]
        if i == 2:
            folds = [1,3,4,5,6,7,8,9,10]
        if i == 3:
            folds = [1,2,4,5,6,7,8,9,10]
        if i == 4:
            folds = [1,2,3,5,6,7,8,9,10]
        if i == 5:
            folds = [1,2,3,4,6,7,8,9,10]
        if i == 6:
            folds = [1,2,3,4,5,7,8,9,10]
        if i == 7:
            folds = [1,2,3,4,5,6,8,9,10]
        if i == 8:
            folds = [1,2,3,4,5,6,7,9,10]
        if i == 9:
            folds = [1,2,3,4,5,6,7,8,10]
        if i == 10:
            folds = [1,2,3,4,5,6,7,8,9]

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


# CreateDefinedFoldCSVs_test()
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
    mfccs = mfccs / np.amax(mfccs)
    delta_mfccs = delta_mfccs / np.amax(delta_mfccs)
    delta2_mfccs = delta2_mfccs / np.amax(delta2_mfccs)

    # Visualise MFCCs
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(delta_mfccs, x_axis="time", sr=sr)
    plt.colorbar(format="%+2f")
    plt.show()
    return


# plot_mfccs("res/audio/518-4-0-0.wav")  # 344-3-0-0.wav for short dog bark clip

# -------------------------------------------------------------------------------------

# preprocess.CreateSTFTSpectrograms("res/test", "res/test2", 1024, 256, "mel", 256, 256, 77,
#                           fill_mode="duplicate")

def plot_mel_spectrogram(audiofile):
    y, sr = librosa.load(audiofile, sr=22050)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=1024, hop_length=256, power=2)
    fig = plt.figure(figsize=(10, 7))
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, n_fft=1024, hop_length=256, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format="%+2.f dB")
    fig.suptitle('Mel-frequency spectrogram')

    image_name = "Mel log Spectrogram.png"
    fig.savefig(image_name)
    plt.show()

    return


def plot_stft_spectrogram(audiofile):
    y, sr = librosa.load(audiofile, sr=22050)

    short_audio = librosa.stft(y, n_fft=1024, hop_length=256)
    Y_audio = np.abs(short_audio) ** 2

    # 2.2 convert amplitude values from linear to logarithmic scale
    Y_log_audio = librosa.power_to_db(Y_audio)

    fig = plt.figure(figsize=(10, 7))
    librosa.display.specshow(Y_log_audio, sr=sr, hop_length=256, n_fft=1024, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.f dB")

    fig.suptitle("STFT Spectrogram")
    cmap = plt.get_cmap("magma")
    plt.set_cmap(cmap)

    image_name = "STFT Spectrogram.png"
    fig.savefig(image_name)
    plt.show()
    return


plot_mel_spectrogram("res/audio_4sec_centered/6902-2-0-4.wav")
# plot_stft_spectrogram("res/audio_4sec_centered/6902-2-0-4.wav")


def create4secWaveFiles(orginal_files, save_path):
    # preprocess.CollectLongFiles(orginal_files, save_path, 4)

    file_list = os.listdir(orginal_files)
    for file in file_list:
        audio_array, sr = librosa.load(orginal_files + "/" + file)
        duration = len(audio_array) / sr
        if duration < 4:
            audio_array = preprocess.DuplicateDataUntilDuration(audio_array, sr, 4)
            wav_file = save_path + "/" + file
            sf.write(wav_file, audio_array, sr, 'PCM_24')
    return

def createCenteredWaveFiles(orginal_files, save_path, target_duration):
    samples_needed = target_duration * 22050
    file_list = os.listdir(orginal_files)

    for file in file_list:
        audio_array, sr = librosa.load(orginal_files + "/" + file)

        if len(audio_array) > samples_needed:
            offset = int(len(audio_array) - samples_needed)
            offset_front = int(offset / 2)
            offset_end = samples_needed + offset_front
            audio_array = audio_array[offset_front:offset_end]

        else:
            audio_array = preprocess.center_audiosignal(audio_array, sr, target_duration)

        wav_file = save_path + "/" + file
        sf.write(wav_file, audio_array, sr, 'PCM_24')
    return


# createCenteredWaveFiles("res/audio", "res/audio_3sec_centered", 3)

def GetSubtypeOf(filename):
    ob = sf.SoundFile(filename)
    print('Sample rate: {}'.format(ob.samplerate))
    print('Channels: {}'.format(ob.channels))
    print('Subtype: {}'.format(ob.subtype))
    return

# GetSubtypeOf("res/audio/100032-3-0-0.wav")

def AnalizeAudioFiles(save_path):
    subtypes = {
        "PCM_16": 0
    }
    sampleRates = {
        "44100": 0
    }
    channels = {
        "2": 0
    }
    durations = {
        "4.0": 0
    }

    filesAnalized = 0
    filesInWrongFormat = []

    file_list = os.listdir(save_path)

    for file in file_list:
        filepath = save_path + "/" + file

        try:
            ob = sf.SoundFile(filepath)
            subtype = format(ob.subtype)
            sampleRate = format(ob.samplerate)
            channel = format(ob.channels)
            duration = int(format(ob.frames)) / int(sampleRate)
            duration = str(round(duration, 2))

            if subtype == "IMA_ADPCM":
                print(format(ob.subtype_info))

            if subtype in subtypes:
                subtypes[subtype] = subtypes[subtype] + 1
            else:
                subtypes[subtype] = 1

            if sampleRate in sampleRates:
                sampleRates[sampleRate] = sampleRates[sampleRate] + 1
            else:
                sampleRates[sampleRate] = 1

            if channel in channels:
                channels[channel] = channels[channel] + 1
            else:
                channels[channel] = 1

            if duration in durations:
                durations[duration] = durations[duration] + 1
            else:
                durations[duration] = 1

            filesAnalized += 1
            # if subtype == "FLOAT":
            #    print(file)
        except:
            filesInWrongFormat.append(file)
            print("Found bad file")

    print("")
    print("Subtypes: " + str(subtypes))
    print("Samplerates: " + str(sampleRates))
    print("Channels: " + str(channels))
    utils.PrintDurationInfo(durations)
    print("Amount of files in wrong format: " + str(len(filesInWrongFormat)))
    print("Files analyzed: " + str(filesAnalized))
    print("------------------------------------------")
    print("Files in wrong format:")
    for x in filesInWrongFormat:
        print(x)
    return


# AnalizeAudioFiles("res/audio_4sec_centered")