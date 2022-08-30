import shutil
import csv
import scipy
import statistics
import librosa
import utils
import feature_extraction as feat
import pywt
from pylab import *
import soundfile as sf

def CreateCWTScaleograms(audio_path, img_save_path, freq_scales, wavelet, px_x, px_y, monitor_dpi, fill_mode):

    utils.clear_directory(img_save_path)
    file_list = os.listdir(audio_path)
    amount_files = len(file_list)
    images_finished = 0
    print("Generating " + str(amount_files) + " Scalograms")

    # set output image size
    x_offset = 0.5  # needed because matplotlib is weird, play around with value until it works
    fig = plt.figure(figsize=(px_x / (monitor_dpi + x_offset), px_y / monitor_dpi))

    for file in file_list:
        # 1.) load audio data
        file_path = audio_path + "/" + file
        audioArray, sampleRate = librosa.load(file_path, sr=22050)
        scales = np.arange(1, freq_scales + 1)  # range of scales

        # 2.) extend short clips
        if fill_mode == "duplicate":
            audioArray = DuplicateDataUntilDuration(audioArray, sampleRate, 4)
        elif fill_mode == "silence":
            audioArray = FillWithSilenceUntilDuration(audioArray, sampleRate, 4)
        elif fill_mode == "centered":
            audioArray = center_audiosignal(audioArray, sampleRate, 4)

        # dt = 1 / sampleRate  # timestep difference
        # frequencies = pywt.scale2frequency(wavelet, scales) / dt  # Get frequencies corresponding to scales

        # 3.) calculate CWT coefficients
        coeffs, freqs = pywt.cwt(audioArray, scales, wavelet)
        coeffs = abs(coeffs)

        # 4.) plot coefficients to scalogram
        # plt.imshow(coeffs, cmap='seismic', aspect='auto')
        # plt.ylabel('Scale')
        # plt.xlabel('Time')
        # plt.show()
        # fig.savefig("results/cwt_scalogram/" + title + ".png")

        utils.plot_scalogram(audioFileName=file, Y=coeffs, save_image=True, save_path=img_save_path,
                               show_plot=False, figure=fig)

        images_finished += 1
        utils.progress_bar(images_finished, amount_files)

    return


def dwt_feature_extraction(audio_path, dwt_feature_csv):
    """
    Extract Detail Coeffs, build features from them and write results to csv file
    """

    file_list = os.listdir(audio_path)
    files_done = 0

    dataCSV = open(dwt_feature_csv, 'w+', encoding='UTF8', newline='')
    data_writer = csv.writer(dataCSV)
    csvHeader = ["audio_file"]
    # write first csv line
    for i in range(1, 15):  # second parameter is dwt max level+1
        csvHeader.append("entropie_" + str(i))
        csvHeader.append("mean_" + str(i))
        csvHeader.append("variance_" + str(i))
        csvHeader.append("std_" + str(i))
        csvHeader.append("iqr_" + str(i))
        csvHeader.append("skew_" + str(i))
        csvHeader.append("kurtosis_" + str(i))
        csvHeader.append("standard_error_mean" + str(i))
        csvHeader.append("median_abs_deviation_" + str(i))
    data_writer.writerow(csvHeader)

    for file in file_list:
        file_path = audio_path + "/" + file

        # 1. Read Audio file
        data, samplerate = librosa.load(file_path)
        # data = data / max(data)
        data = data[0:32768]
        wavelet = "db1"
        max_level = pywt.dwt_max_level(len(data), wavelet) - 1

        coeffs = pywt.wavedec(data, wavelet, level=max_level, mode="symmetric")
        line = [file]

        MAX_median_abs_dev = 0
        MAX_standard_error_mean = 0
        MAX_iqr = 0
        MAX_variation = 0

        # Calculate Features
        for i in range(1, len(coeffs)):
            shannon_ent = feat.shannon_entropy(coeffs[i])
            mean = statistics.mean(coeffs[i])
            variance = statistics.variance(coeffs[i], mean)
            std = np.std(coeffs[i])
            iqr = scipy.stats.iqr(coeffs[i])
            skew = scipy.stats.skew(coeffs[i])
            kurtosis = scipy.stats.kurtosis(coeffs[i])
            standard_error_mean = scipy.stats.sem(coeffs[i])
            median_abs_deviation = scipy.stats.median_abs_deviation(coeffs[i])

            line.append(shannon_ent)
            line.append(mean)
            line.append(variance)
            line.append(std)
            line.append(iqr)
            line.append(skew)
            line.append(kurtosis)
            line.append(standard_error_mean)
            line.append(median_abs_deviation)

        # zero_crossing_rate

        data_writer.writerow(line)
        files_done += 1

        utils.progress_bar(files_done, len(file_list))

    dataCSV.close()
    return


def collect_all_DWT_data(audio_path, dwt_full_data_csv):
    file_list = os.listdir(audio_path)
    files_done = 0

    dataCSV = open(dwt_full_data_csv, 'w+', encoding='UTF8', newline='')
    data_writer = csv.writer(dataCSV)

    for file in file_list:
        audio_file = audio_path + "/" + file

        # 1. Read Audio file
        data, samplerate = librosa.load(audio_file)
        # data = data / max(data)
        data = data[0:32768]

        wavelet = "db1"
        max_level = pywt.dwt_max_level(len(data), wavelet)
        print("max_level = " + str(max_level))

        arr = np.array([[file]])
        for i in range(1, max_level+1):
            coeffs = pywt.wavedec(data, wavelet, level=i, mode="symmetric")
            arr = np.append(arr, coeffs[0])
            if i == max_level:
                for j in range(0, max_level):
                    arr = append(arr, coeffs[max_level-j])
                    # arr length will be: amout of samples * 2 - 2 (only when sample amount is in x^2 line)
        data_writer.writerow(arr)
        files_done += 1

        utils.progress_bar(files_done, len(file_list))
    dataCSV.close()

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

def center_audiosignal(audio_array, sample_rate, target_duration):
    """
    Add zeroes infront and behind an audio signal (array) until it is target_duration in secons long
    :param audio_array: array that contains the audio signal, must be shortern then target duration
    :param sample_rate: sample rate of audio signal
    :param target_duration: how long the resulting clip should be in seconds
    :return: array like: [0, 0, 0, 1, 5, 6, 2, 1, 3, 0, 0, 0]
    """

    samples_needed = sample_rate * target_duration
    len(audio_array)
    if len(audio_array) == samples_needed:
        return audio_array

    padding = int((samples_needed - len(audio_array)) / 2) + 1
    centered_signal = np.zeros(padding)
    centered_signal = np.append(centered_signal, audio_array)
    centered_signal = np.append(centered_signal, np.zeros(padding))
    centered_signal = centered_signal[0:samples_needed]

    return centered_signal


def CreateSTFTSpectrograms(audio_path, img_save_path, FrameSize, HopSize, mels, freq_scale, px_x, px_y, monitor_dpi,
                           duration, spec_type, fill_mode="duplicate", samplerate=22050):
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

        audioArray, sampleRate = librosa.load(file_path, sr=samplerate)
        if fill_mode == "duplicate":
            audioArray = DuplicateDataUntilDuration(audioArray, sampleRate, duration)
        elif fill_mode == "silence":
            audioArray = FillWithSilenceUntilDuration(audioArray, sampleRate, duration)
        elif fill_mode == "centered":
            audioArray = center_audiosignal(audioArray, sampleRate, duration)


        # 1.) Extract Short-Time Fourier Transform
        if spec_type == "stft":
            short_audio = librosa.stft(audioArray, n_fft=FrameSize, hop_length=HopSize)
            # short_audio contains complex numbers (can't be visualized), square values to convert to floats
            Y_audio = np.abs(short_audio) ** 2  # Y value is the amplitude (because we did a fourier transform)
        if spec_type == "mel":
            # does the same as the stft version but also applies mel bins to the result
            Y_audio = librosa.feature.melspectrogram(y=audioArray, n_mels=mels, n_fft=FrameSize, hop_length=HopSize)

        # 2.) convert amplitude values from linear to logarithmic scale (to get dB)
        Y_log_audio = librosa.power_to_db(Y_audio)

        # 3.) Plot results
        utils.plot_spectrogram(audioFileName=file, Y=Y_log_audio, samplerate=sampleRate, frame_size=FrameSize,
            hop_size=HopSize, y_axis=freq_scale, save_image=True, save_path=img_save_path, show_plot=False, figure=fig)

        images_finished += 1
        utils.progress_bar(images_finished, amount_files)

    return

