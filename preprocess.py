import os
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
    x_offset = 0.3  # needed because matplotlib is weird, play around with value until it works
    fig = plt.figure(figsize=(px_x / (monitor_dpi + x_offset), px_y / monitor_dpi))

    for file in file_list:
        # 1.) load audio data
        file_path = audio_path + "/" + file
        audioArray, sampleRate = librosa.load(file_path, sr=44100)
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


def dwt_feature_extraction_V4(audio_path, dwt_feature_csv, samplerate, wavelet="db1"):
    """
    Extract Detail- and Approx- Coeffs, build features from them and write results to csv file
    Use 1 Segment that is 131.072 (ca. 3sec) long
    Compared to V1 power, energy and zero-crossing-rate is added
    Uses 14 (15 with approx.) DWT Levels with 10 featuers per level (resulting in 150 features)
    """

    file_list = os.listdir(audio_path)
    files_done = 0

    dataCSV = open(dwt_feature_csv, 'w+', encoding='UTF8', newline='')
    data_writer = csv.writer(dataCSV)
    csvHeader = ["audio_file"]
    # write first csv line
    for i in range(1, 16):  # second parameter is dwt max level+2
        csvHeader.append("entropie_" + str(i))
        csvHeader.append("mean_" + str(i))
        csvHeader.append("variance_" + str(i))
        csvHeader.append("std_" + str(i))
        csvHeader.append("iqr_" + str(i))
        csvHeader.append("skew_" + str(i))
        csvHeader.append("kurtosis_" + str(i))
        csvHeader.append("standard_error_mean_" + str(i))
        csvHeader.append("median_abs_deviation_" + str(i))
        # csvHeader.append("energy_" + str(i))
        # csvHeader.append("avg_power_" + str(i))
        csvHeader.append("zero_cross_rate_" + str(i))
    data_writer.writerow(csvHeader)

    for file in file_list:
        file_path = audio_path + "/" + file

        # 1. Read Audio file
        data, sr = librosa.load(file_path, sr=samplerate)
        data = data[614:131686]  # results in center 131.072 samples of 3 sec clip

        # amp_mean = statistics.mean(abs(data))         tested, performed about 5% worse in cross-val
        # data = data / amp_mean

        coeffs = pywt.wavedec(data, wavelet, level=14, mode="symmetric")
        line = [file]

        # Calculate Features
        for i in range(0, len(coeffs)):
            shannon_ent = feat.shannon_entropy(coeffs[i])
            mean = statistics.mean(coeffs[i])
            variance = statistics.variance(coeffs[i], mean)
            std = np.std(coeffs[i])
            iqr = scipy.stats.iqr(coeffs[i])
            skew = scipy.stats.skew(coeffs[i])
            kurtosis = scipy.stats.kurtosis(coeffs[i])
            standard_error_mean = scipy.stats.sem(coeffs[i])
            median_abs_deviation = scipy.stats.median_abs_deviation(coeffs[i])
            zero_cross_arr = librosa.feature.zero_crossing_rate(coeffs[i],
                                                    frame_length=len(coeffs[i]), hop_length=len(coeffs[i])+1)
            zero_cross_rate = zero_cross_arr[0][0]

            line.append(shannon_ent)
            line.append(mean)
            line.append(variance)
            line.append(std)
            line.append(iqr)
            line.append(skew)
            line.append(kurtosis)
            line.append(standard_error_mean)
            line.append(median_abs_deviation)
            line.append(zero_cross_rate)

        data_writer.writerow(line)
        files_done += 1

        utils.progress_bar(files_done, len(file_list))

    dataCSV.close()
    return


def dwt_feature_extraction_V5(audio_path, dwt_feature_csv, samplerate, wavelet="db1"):
    """
    Extract Detail- and Approx- Coeffs, build features from them and write results to csv file
    """

    file_list = os.listdir(audio_path)
    files_done = 0

    dataCSV = open(dwt_feature_csv, 'w+', encoding='UTF8', newline='')
    data_writer = csv.writer(dataCSV)
    csvHeader = ["audio_file"]

    # write first csv line
    i = 1
    while i < 65:  # max_level+1 * anzahl von segmenten
        csvHeader.append("entropie_" + str(i))
        csvHeader.append("mean_" + str(i))
        csvHeader.append("variance_" + str(i))
        csvHeader.append("std_" + str(i))
        csvHeader.append("iqr_" + str(i))
        csvHeader.append("skew_" + str(i))
        csvHeader.append("kurtosis_" + str(i))
        csvHeader.append("standard_error_mean_" + str(i))
        csvHeader.append("median_abs_deviation_" + str(i))
        csvHeader.append("zero_cross_rate_" + str(i))

        csvHeader.append("entropie_" + str(i + 1))
        csvHeader.append("mean_" + str(i + 1))
        csvHeader.append("variance_" + str(i + 1))
        csvHeader.append("std_" + str(i + 1))
        csvHeader.append("iqr_" + str(i + 1))
        csvHeader.append("skew_" + str(i + 1))
        csvHeader.append("kurtosis_" + str(i + 1))
        csvHeader.append("standard_error_mean_" + str(i + 1))
        csvHeader.append("median_abs_deviation_" + str(i + 1))
        csvHeader.append("zero_cross_rate_" + str(i + 1))

        csvHeader.append("entropie_" + str(i + 2))
        csvHeader.append("mean_" + str(i + 2))
        csvHeader.append("variance_" + str(i + 2))
        csvHeader.append("std_" + str(i + 2))
        csvHeader.append("iqr_" + str(i + 2))
        csvHeader.append("skew_" + str(i + 2))
        csvHeader.append("kurtosis_" + str(i + 2))
        csvHeader.append("standard_error_mean_" + str(i + 2))
        csvHeader.append("median_abs_deviation_" + str(i + 2))
        csvHeader.append("zero_cross_rate_" + str(i + 2))

        csvHeader.append("entropie_" + str(i + 3))
        csvHeader.append("mean_" + str(i + 3))
        csvHeader.append("variance_" + str(i + 3))
        csvHeader.append("std_" + str(i + 3))
        csvHeader.append("iqr_" + str(i + 3))
        csvHeader.append("skew_" + str(i + 3))
        csvHeader.append("kurtosis_" + str(i + 3))
        csvHeader.append("standard_error_mean_" + str(i + 3))
        csvHeader.append("median_abs_deviation_" + str(i + 3))
        csvHeader.append("zero_cross_rate_" + str(i + 3))

        csvHeader.append("entropie_" + str(i + 4))
        csvHeader.append("mean_" + str(i + 4))
        csvHeader.append("variance_" + str(i + 4))
        csvHeader.append("std_" + str(i + 4))
        csvHeader.append("iqr_" + str(i + 4))
        csvHeader.append("skew_" + str(i + 4))
        csvHeader.append("kurtosis_" + str(i + 4))
        csvHeader.append("standard_error_mean_" + str(i + 4))
        csvHeader.append("median_abs_deviation_" + str(i + 4))
        csvHeader.append("zero_cross_rate_" + str(i + 4))

        i += 5
    data_writer.writerow(csvHeader)

    for file in file_list:
        file_path = audio_path + "/" + file

        # 1. Read Audio file
        data, sr = librosa.load(file_path, sr=samplerate)
        seg1 = data[6280:39048]
        seg2 = data[39048:71816]
        seg3 = data[71816:104584]
        seg4 = data[104584:137352]
        seg5 = data[137352:170120]
        # max_level = pywt.dwt_max_level(len(data), wavelet) - 1

        coeffs1 = pywt.wavedec(seg1, wavelet, level=12, mode="symmetric")
        coeffs2 = pywt.wavedec(seg2, wavelet, level=12, mode="symmetric")
        coeffs3 = pywt.wavedec(seg3, wavelet, level=12, mode="symmetric")
        coeffs4 = pywt.wavedec(seg4, wavelet, level=12, mode="symmetric")
        coeffs5 = pywt.wavedec(seg5, wavelet, level=12, mode="symmetric")
        line = [file]

        # Calculate Features
        for i in range(0, len(coeffs1)):
            shannon_ent1 = feat.shannon_entropy(coeffs1[i])
            shannon_ent2 = feat.shannon_entropy(coeffs2[i])
            shannon_ent3 = feat.shannon_entropy(coeffs3[i])
            shannon_ent4 = feat.shannon_entropy(coeffs4[i])
            shannon_ent5 = feat.shannon_entropy(coeffs5[i])
            mean1 = statistics.mean(coeffs1[i])
            mean2 = statistics.mean(coeffs2[i])
            mean3 = statistics.mean(coeffs3[i])
            mean4 = statistics.mean(coeffs4[i])
            mean5 = statistics.mean(coeffs5[i])
            variance1 = statistics.variance(coeffs1[i], mean1)
            variance2 = statistics.variance(coeffs2[i], mean2)
            variance3 = statistics.variance(coeffs3[i], mean3)
            variance4 = statistics.variance(coeffs4[i], mean4)
            variance5 = statistics.variance(coeffs5[i], mean5)
            std1 = np.std(coeffs1[i])
            std2 = np.std(coeffs2[i])
            std3 = np.std(coeffs3[i])
            std4 = np.std(coeffs4[i])
            std5 = np.std(coeffs5[i])
            iqr1 = scipy.stats.iqr(coeffs1[i])
            iqr2 = scipy.stats.iqr(coeffs2[i])
            iqr3 = scipy.stats.iqr(coeffs3[i])
            iqr4 = scipy.stats.iqr(coeffs4[i])
            iqr5 = scipy.stats.iqr(coeffs5[i])
            skew1 = scipy.stats.skew(coeffs1[i])
            skew2 = scipy.stats.skew(coeffs2[i])
            skew3 = scipy.stats.skew(coeffs3[i])
            skew4 = scipy.stats.skew(coeffs4[i])
            skew5 = scipy.stats.skew(coeffs5[i])
            kurtosis1 = scipy.stats.kurtosis(coeffs1[i])
            kurtosis2 = scipy.stats.kurtosis(coeffs2[i])
            kurtosis3 = scipy.stats.kurtosis(coeffs3[i])
            kurtosis4 = scipy.stats.kurtosis(coeffs4[i])
            kurtosis5 = scipy.stats.kurtosis(coeffs5[i])
            standard_error_mean1 = scipy.stats.sem(coeffs1[i])
            standard_error_mean2 = scipy.stats.sem(coeffs2[i])
            standard_error_mean3 = scipy.stats.sem(coeffs3[i])
            standard_error_mean4 = scipy.stats.sem(coeffs4[i])
            standard_error_mean5 = scipy.stats.sem(coeffs5[i])
            median_abs_deviation1 = scipy.stats.median_abs_deviation(coeffs1[i])
            median_abs_deviation2 = scipy.stats.median_abs_deviation(coeffs2[i])
            median_abs_deviation3 = scipy.stats.median_abs_deviation(coeffs3[i])
            median_abs_deviation4 = scipy.stats.median_abs_deviation(coeffs4[i])
            median_abs_deviation5 = scipy.stats.median_abs_deviation(coeffs5[i])

            zero_cross_arr = librosa.feature.zero_crossing_rate(coeffs1[i], frame_length=len(coeffs1[i]),
                                                                hop_length=len(coeffs1[i]) + 1)
            zero_cross_rate1 = zero_cross_arr[0][0]
            zero_cross_arr = librosa.feature.zero_crossing_rate(coeffs2[i], frame_length=len(coeffs2[i]),
                                                                hop_length=len(coeffs2[i]) + 1)
            zero_cross_rate2 = zero_cross_arr[0][0]
            zero_cross_arr = librosa.feature.zero_crossing_rate(coeffs3[i], frame_length=len(coeffs3[i]),
                                                                hop_length=len(coeffs3[i]) + 1)
            zero_cross_rate3 = zero_cross_arr[0][0]
            zero_cross_arr = librosa.feature.zero_crossing_rate(coeffs4[i], frame_length=len(coeffs4[i]),
                                                                hop_length=len(coeffs4[i]) + 1)
            zero_cross_rate4 = zero_cross_arr[0][0]
            zero_cross_arr = librosa.feature.zero_crossing_rate(coeffs5[i], frame_length=len(coeffs5[i]),
                                                                hop_length=len(coeffs5[i]) + 1)
            zero_cross_rate5 = zero_cross_arr[0][0]

            line.append(shannon_ent1)
            line.append(mean1)
            line.append(variance1)
            line.append(std1)
            line.append(iqr1)
            line.append(skew1)
            line.append(kurtosis1)
            line.append(standard_error_mean1)
            line.append(median_abs_deviation1)
            line.append(zero_cross_rate1)

            line.append(shannon_ent2)
            line.append(mean2)
            line.append(variance2)
            line.append(std2)
            line.append(iqr2)
            line.append(skew2)
            line.append(kurtosis2)
            line.append(standard_error_mean2)
            line.append(median_abs_deviation2)
            line.append(zero_cross_rate2)

            line.append(shannon_ent3)
            line.append(mean3)
            line.append(variance3)
            line.append(std3)
            line.append(iqr3)
            line.append(skew3)
            line.append(kurtosis3)
            line.append(standard_error_mean3)
            line.append(median_abs_deviation3)
            line.append(zero_cross_rate3)

            line.append(shannon_ent4)
            line.append(mean4)
            line.append(variance4)
            line.append(std4)
            line.append(iqr4)
            line.append(skew4)
            line.append(kurtosis4)
            line.append(standard_error_mean4)
            line.append(median_abs_deviation4)
            line.append(zero_cross_rate4)

            line.append(shannon_ent5)
            line.append(mean5)
            line.append(variance5)
            line.append(std5)
            line.append(iqr5)
            line.append(skew5)
            line.append(kurtosis5)
            line.append(standard_error_mean5)
            line.append(median_abs_deviation5)
            line.append(zero_cross_rate5)

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
        data = data[0:131072]  # close to 3 seconds for 44khz samplerate

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
    x_offset = 0.2  # needed because matplotlib is weird, play around with value until it works
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

