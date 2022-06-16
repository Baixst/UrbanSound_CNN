import os
import csv
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


def plot_spectrogram(audioFileName, Y, samplerate, frame_size, hop_size, save_path, y_axis, save_image, show_plot, figure):
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


def plot_scalogram(audioFileName, Y, save_path, save_image, show_plot, figure):

    plt.imshow(Y, cmap='gray', aspect='auto')  # seismic is a nice cmap
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


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("Could not delete " + file_path + " because it does not exist")
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


def progress_bar(current, total, bar_length=30):
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}% --- {current}/{total}', end=ending)
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

