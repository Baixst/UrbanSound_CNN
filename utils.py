import os
import soundfile as sf
import librosa
import librosa.display
from matplotlib import pyplot as plt
import shutil


def CheckDurations(path):
    file_list = os.listdir(path)

    durations = {
        "4.0": 0
    }

    for file in file_list:
        filepath = path + "/" + file
        ob = sf.SoundFile(filepath)
        sampleRate = format(ob.samplerate)
        duration = int(format(ob.frames)) / int(sampleRate)
        duration = str(round(duration, 2))
        if duration in durations:
            durations[duration] = durations[duration] + 1
        else:
            durations[duration] = 1

    return durations


def plot_spectrogram(audioFileName, Y, samplerate, frame_size, hop_size, figure, save_path, y_axis="linear", save_image=False, show_plot=True):
    # fig = plt.figure(figsize=(4, 2))
    librosa.display.specshow(Y, sr=samplerate, hop_length=hop_size, x_axis="time", y_axis=y_axis)

    # plt.colorbar(format="%+2.f dB")
    # title = y_axis + "-scale Frequency, Frame Size = " + str(frame_size) + ", Hop Size = " + str(hop_size) + " | " + audioFileName
    # fig.subtitle(title)

    cmap = plt.get_cmap("gray")
    plt.set_cmap(cmap)
    plt.axis('off')

    if save_image:
        image_name = GenerateImageName(audioFileName, save_path)
        figure.savefig(image_name, bbox_inches='tight', pad_inches=0)
    if show_plot:
        plt.show()

    plt.clf()
    figure.clear()
    return

def GenerateImageName(audio_file_name, save_path):
    string_list = audio_file_name.split(".")
    img_name = string_list[0] + ".png"
    return save_path + "/" + img_name


def clear_directory(dir_path):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    return


def print_class_distribution(path):
    classes = {
        "0": 0
    }

    for file in os.listdir(path):
        string_list = file.split("-")
        classID = string_list[1]
        if classID in classes:
            classes[classID] = classes[classID] + 1
        else:
            classes[classID] = 1

    print("Class Distribution:")
    print(classes)
    return

