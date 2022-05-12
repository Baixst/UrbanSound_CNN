import os
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

    image_name = "res/test2/sr8000_" + str(sampleRate) + ".png"
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

audioArray, sampleRate = librosa.load("res/test/short_clip.wav", sr=22050)
audioArray2 = preprocess.FillWithSilenceUntilDuration(audioArray, sampleRate, 4)
plotSpectrogramWithSamplerate(samplerate=sampleRate, file="res/test/short_clip.wav", y_axis="log", audio_arr=audioArray2)