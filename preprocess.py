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


def dwt_feature_extraction(audio_path, dwt_feature_csv, samplerate, wavelet="db1"):
    """
    Extract Detail- and Approx- Coeffs, build features from them and write results to csv file
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
    data_writer.writerow(csvHeader)

    for file in file_list:
        file_path = audio_path + "/" + file

        # 1. Read Audio file
        data, sr = librosa.load(file_path, sr=samplerate)
        # data = data / max(data)
        # data = data[0:131072]  # results in center 131.072 samples of 3 sec clip
        # max_level = pywt.dwt_max_level(len(data), wavelet) - 1

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


def dwt_feature_extractionV2(audio_path, dwt_feature_csv, samplerate, wavelet="db1"):
    """
    Slice audio file into 1sec clips and extract detail and approx Coeffs, build features from them and write results to csv file
    """

    file_list = os.listdir(audio_path)
    files_done = 0

    dataCSV = open(dwt_feature_csv, 'w+', encoding='UTF8', newline='')
    data_writer = csv.writer(dataCSV)
    csvHeader = ["audio_file"]

    # write first csv line
    i = 1
    while i < 52:
        csvHeader.append("entropie_" + str(i))
        csvHeader.append("mean_" + str(i))
        csvHeader.append("variance_" + str(i))
        csvHeader.append("std_" + str(i))
        csvHeader.append("iqr_" + str(i))
        csvHeader.append("skew_" + str(i))
        csvHeader.append("kurtosis_" + str(i))
        csvHeader.append("standard_error_mean_" + str(i))
        csvHeader.append("median_abs_deviation_" + str(i))

        csvHeader.append("entropie_" + str(i + 1))
        csvHeader.append("mean_" + str(i + 1))
        csvHeader.append("variance_" + str(i + 1))
        csvHeader.append("std_" + str(i + 1))
        csvHeader.append("iqr_" + str(i + 1))
        csvHeader.append("skew_" + str(i + 1))
        csvHeader.append("kurtosis_" + str(i + 1))
        csvHeader.append("standard_error_mean_" + str(i + 1))
        csvHeader.append("median_abs_deviation_" + str(i + 1))

        csvHeader.append("entropie_" + str(i + 2))
        csvHeader.append("mean_" + str(i + 2))
        csvHeader.append("variance_" + str(i + 2))
        csvHeader.append("std_" + str(i + 2))
        csvHeader.append("iqr_" + str(i + 2))
        csvHeader.append("skew_" + str(i + 2))
        csvHeader.append("kurtosis_" + str(i + 2))
        csvHeader.append("standard_error_mean_" + str(i + 2))
        csvHeader.append("median_abs_deviation_" + str(i + 2))

        csvHeader.append("entropie_" + str(i + 3))
        csvHeader.append("mean_" + str(i + 3))
        csvHeader.append("variance_" + str(i + 3))
        csvHeader.append("std_" + str(i + 3))
        csvHeader.append("iqr_" + str(i + 3))
        csvHeader.append("skew_" + str(i + 3))
        csvHeader.append("kurtosis_" + str(i + 3))
        csvHeader.append("standard_error_mean_" + str(i + 3))
        csvHeader.append("median_abs_deviation_" + str(i + 3))

        i += 4
    data_writer.writerow(csvHeader)

    for file in file_list:
        file_path = audio_path + "/" + file

        # 1. Read Audio file
        data, sr = librosa.load(file_path, sr=samplerate)

        seg1 = data[0:44000]
        seg2 = data[44000:88000]
        seg3 = data[88000:132000]
        seg4 = data[132000:176000]

        # max_level = pywt.dwt_max_level(len(data), wavelet) - 1

        coeffs1 = pywt.wavedec(seg1, wavelet, level=12, mode="symmetric")
        coeffs2 = pywt.wavedec(seg2, wavelet, level=12, mode="symmetric")
        coeffs3 = pywt.wavedec(seg3, wavelet, level=12, mode="symmetric")
        coeffs4 = pywt.wavedec(seg4, wavelet, level=12, mode="symmetric")
        line = [file]

        # Calculate Features
        for i in range(0, len(coeffs1)):
            shannon_ent1 = feat.shannon_entropy(coeffs1[i])
            shannon_ent2 = feat.shannon_entropy(coeffs2[i])
            shannon_ent3 = feat.shannon_entropy(coeffs3[i])
            shannon_ent4 = feat.shannon_entropy(coeffs4[i])
            mean1 = statistics.mean(coeffs1[i])
            mean2 = statistics.mean(coeffs2[i])
            mean3 = statistics.mean(coeffs3[i])
            mean4 = statistics.mean(coeffs4[i])
            variance1 = statistics.variance(coeffs1[i], mean1)
            variance2 = statistics.variance(coeffs2[i], mean2)
            variance3 = statistics.variance(coeffs3[i], mean3)
            variance4 = statistics.variance(coeffs4[i], mean4)
            std1 = np.std(coeffs1[i])
            std2 = np.std(coeffs2[i])
            std3 = np.std(coeffs3[i])
            std4 = np.std(coeffs4[i])
            iqr1 = scipy.stats.iqr(coeffs1[i])
            iqr2 = scipy.stats.iqr(coeffs2[i])
            iqr3 = scipy.stats.iqr(coeffs3[i])
            iqr4 = scipy.stats.iqr(coeffs4[i])
            skew1 = scipy.stats.skew(coeffs1[i])
            skew2 = scipy.stats.skew(coeffs2[i])
            skew3 = scipy.stats.skew(coeffs3[i])
            skew4 = scipy.stats.skew(coeffs4[i])
            kurtosis1 = scipy.stats.kurtosis(coeffs1[i])
            kurtosis2 = scipy.stats.kurtosis(coeffs2[i])
            kurtosis3 = scipy.stats.kurtosis(coeffs3[i])
            kurtosis4 = scipy.stats.kurtosis(coeffs4[i])
            standard_error_mean1 = scipy.stats.sem(coeffs1[i])
            standard_error_mean2 = scipy.stats.sem(coeffs2[i])
            standard_error_mean3 = scipy.stats.sem(coeffs3[i])
            standard_error_mean4 = scipy.stats.sem(coeffs4[i])
            median_abs_deviation1 = scipy.stats.median_abs_deviation(coeffs1[i])
            median_abs_deviation2 = scipy.stats.median_abs_deviation(coeffs2[i])
            median_abs_deviation3 = scipy.stats.median_abs_deviation(coeffs3[i])
            median_abs_deviation4 = scipy.stats.median_abs_deviation(coeffs4[i])

            line.append(shannon_ent1)
            line.append(mean1)
            line.append(variance1)
            line.append(std1)
            line.append(iqr1)
            line.append(skew1)
            line.append(kurtosis1)
            line.append(standard_error_mean1)
            line.append(median_abs_deviation1)

            line.append(shannon_ent2)
            line.append(mean2)
            line.append(variance2)
            line.append(std2)
            line.append(iqr2)
            line.append(skew2)
            line.append(kurtosis2)
            line.append(standard_error_mean2)
            line.append(median_abs_deviation2)

            line.append(shannon_ent3)
            line.append(mean3)
            line.append(variance3)
            line.append(std3)
            line.append(iqr3)
            line.append(skew3)
            line.append(kurtosis3)
            line.append(standard_error_mean3)
            line.append(median_abs_deviation3)

            line.append(shannon_ent4)
            line.append(mean4)
            line.append(variance4)
            line.append(std4)
            line.append(iqr4)
            line.append(skew4)
            line.append(kurtosis4)
            line.append(standard_error_mean4)
            line.append(median_abs_deviation4)

        data_writer.writerow(line)
        files_done += 1

        utils.progress_bar(files_done, len(file_list))

    dataCSV.close()
    return


def dwt_feature_extractionV3(audio_path, dwt_feature_csv, samplerate, wavelet="db1"):
    file_list = os.listdir(audio_path)
    files_done = 0

    dataCSV = open(dwt_feature_csv, 'w+', encoding='UTF8', newline='')
    data_writer = csv.writer(dataCSV)
    csvHeader = ["audio_file"]

    # write first csv line
    i = 1
    while i < 104:  # max_level+1 * anzahl von segmenten
        csvHeader.append("entropie_" + str(i))
        csvHeader.append("mean_" + str(i))
        csvHeader.append("variance_" + str(i))
        csvHeader.append("std_" + str(i))
        csvHeader.append("iqr_" + str(i))
        csvHeader.append("skew_" + str(i))
        csvHeader.append("kurtosis_" + str(i))
        csvHeader.append("standard_error_mean_" + str(i))
        csvHeader.append("median_abs_deviation_" + str(i))

        csvHeader.append("entropie_" + str(i + 1))
        csvHeader.append("mean_" + str(i + 1))
        csvHeader.append("variance_" + str(i + 1))
        csvHeader.append("std_" + str(i + 1))
        csvHeader.append("iqr_" + str(i + 1))
        csvHeader.append("skew_" + str(i + 1))
        csvHeader.append("kurtosis_" + str(i + 1))
        csvHeader.append("standard_error_mean_" + str(i + 1))
        csvHeader.append("median_abs_deviation_" + str(i + 1))

        csvHeader.append("entropie_" + str(i + 2))
        csvHeader.append("mean_" + str(i + 2))
        csvHeader.append("variance_" + str(i + 2))
        csvHeader.append("std_" + str(i + 2))
        csvHeader.append("iqr_" + str(i + 2))
        csvHeader.append("skew_" + str(i + 2))
        csvHeader.append("kurtosis_" + str(i + 2))
        csvHeader.append("standard_error_mean_" + str(i + 2))
        csvHeader.append("median_abs_deviation_" + str(i + 2))

        csvHeader.append("entropie_" + str(i + 3))
        csvHeader.append("mean_" + str(i + 3))
        csvHeader.append("variance_" + str(i + 3))
        csvHeader.append("std_" + str(i + 3))
        csvHeader.append("iqr_" + str(i + 3))
        csvHeader.append("skew_" + str(i + 3))
        csvHeader.append("kurtosis_" + str(i + 3))
        csvHeader.append("standard_error_mean_" + str(i + 3))
        csvHeader.append("median_abs_deviation_" + str(i + 3))

        i += 4
    data_writer.writerow(csvHeader)

    for file in file_list:
        file_path = audio_path + "/" + file

        # 1. Read Audio file
        data, sr = librosa.load(file_path, sr=samplerate)
        # data = data = data[614:131686]  # results in center 131.072 samples of 3 sec clip

        # 2. Calculate DWT
        coeffs = pywt.wavedec(data, wavelet, level=12, mode="symmetric")  # 14 for 4 segments
        line = [file]

        for i in range(0, len(coeffs)):

            # 3. Split coeffs in 4 segments
            half1, half2 = utils.split_list(coeffs[i])
            quader1, quader2 = utils.split_list(half1)
            quader3, quader4 = utils.split_list(half2)
            part1, part2 = utils.split_list(quader1)
            part3, part4 = utils.split_list(quader2)
            part5, part6 = utils.split_list(quader3)
            part7, part8 = utils.split_list(quader4)


            # 4. Calculate features
            shannon_ent1 = feat.shannon_entropy(part1)
            shannon_ent2 = feat.shannon_entropy(part2)
            shannon_ent3 = feat.shannon_entropy(part3)
            shannon_ent4 = feat.shannon_entropy(part4)
            shannon_ent5 = feat.shannon_entropy(part5)
            shannon_ent6 = feat.shannon_entropy(part6)
            shannon_ent7 = feat.shannon_entropy(part7)
            shannon_ent8 = feat.shannon_entropy(part8)
            mean1 = statistics.mean(part1)
            mean2 = statistics.mean(part2)
            mean3 = statistics.mean(part3)
            mean4 = statistics.mean(part4)
            mean5 = statistics.mean(part5)
            mean6 = statistics.mean(part6)
            mean7 = statistics.mean(part7)
            mean8 = statistics.mean(part8)
            variance1 = statistics.variance(part1, mean1)
            variance2 = statistics.variance(part2, mean2)
            variance3 = statistics.variance(part3, mean3)
            variance4 = statistics.variance(part4, mean4)
            variance5 = statistics.variance(part5, mean5)
            variance6 = statistics.variance(part6, mean6)
            variance7 = statistics.variance(part7, mean7)
            variance8 = statistics.variance(part8, mean8)
            std1 = np.std(part1)
            std2 = np.std(part2)
            std3 = np.std(part3)
            std4 = np.std(part4)
            std5 = np.std(part5)
            std6 = np.std(part6)
            std7 = np.std(part7)
            std8 = np.std(part8)
            iqr1 = scipy.stats.iqr(part1)
            iqr2 = scipy.stats.iqr(part2)
            iqr3 = scipy.stats.iqr(part3)
            iqr4 = scipy.stats.iqr(part4)
            iqr5 = scipy.stats.iqr(part5)
            iqr6 = scipy.stats.iqr(part6)
            iqr7 = scipy.stats.iqr(part7)
            iqr8 = scipy.stats.iqr(part8)
            skew1 = scipy.stats.skew(part1)
            skew2 = scipy.stats.skew(part2)
            skew3 = scipy.stats.skew(part3)
            skew4 = scipy.stats.skew(part4)
            skew5 = scipy.stats.skew(part5)
            skew6 = scipy.stats.skew(part6)
            skew7 = scipy.stats.skew(part7)
            skew8 = scipy.stats.skew(part8)
            kurtosis1 = scipy.stats.kurtosis(part1)
            kurtosis2 = scipy.stats.kurtosis(part2)
            kurtosis3 = scipy.stats.kurtosis(part3)
            kurtosis4 = scipy.stats.kurtosis(part4)
            kurtosis5 = scipy.stats.kurtosis(part5)
            kurtosis6 = scipy.stats.kurtosis(part6)
            kurtosis7 = scipy.stats.kurtosis(part7)
            kurtosis8 = scipy.stats.kurtosis(part8)
            standard_error_mean1 = scipy.stats.sem(part1)
            standard_error_mean2 = scipy.stats.sem(part2)
            standard_error_mean3 = scipy.stats.sem(part3)
            standard_error_mean4 = scipy.stats.sem(part4)
            standard_error_mean5 = scipy.stats.sem(part5)
            standard_error_mean6 = scipy.stats.sem(part6)
            standard_error_mean7 = scipy.stats.sem(part7)
            standard_error_mean8 = scipy.stats.sem(part8)
            median_abs_deviation1 = scipy.stats.median_abs_deviation(part1)
            median_abs_deviation2 = scipy.stats.median_abs_deviation(part2)
            median_abs_deviation3 = scipy.stats.median_abs_deviation(part3)
            median_abs_deviation4 = scipy.stats.median_abs_deviation(part4)
            median_abs_deviation5 = scipy.stats.median_abs_deviation(part5)
            median_abs_deviation6 = scipy.stats.median_abs_deviation(part6)
            median_abs_deviation7 = scipy.stats.median_abs_deviation(part7)
            median_abs_deviation8 = scipy.stats.median_abs_deviation(part8)

            line.append(shannon_ent1)
            line.append(mean1)
            line.append(variance1)
            line.append(std1)
            line.append(iqr1)
            line.append(skew1)
            line.append(kurtosis1)
            line.append(standard_error_mean1)
            line.append(median_abs_deviation1)

            line.append(shannon_ent2)
            line.append(mean2)
            line.append(variance2)
            line.append(std2)
            line.append(iqr2)
            line.append(skew2)
            line.append(kurtosis2)
            line.append(standard_error_mean2)
            line.append(median_abs_deviation2)

            line.append(shannon_ent3)
            line.append(mean3)
            line.append(variance3)
            line.append(std3)
            line.append(iqr3)
            line.append(skew3)
            line.append(kurtosis3)
            line.append(standard_error_mean3)
            line.append(median_abs_deviation3)

            line.append(shannon_ent4)
            line.append(mean4)
            line.append(variance4)
            line.append(std4)
            line.append(iqr4)
            line.append(skew4)
            line.append(kurtosis4)
            line.append(standard_error_mean4)
            line.append(median_abs_deviation4)

            line.append(shannon_ent5)
            line.append(mean5)
            line.append(variance5)
            line.append(std5)
            line.append(iqr5)
            line.append(skew5)
            line.append(kurtosis5)
            line.append(standard_error_mean5)
            line.append(median_abs_deviation5)

            line.append(shannon_ent6)
            line.append(mean6)
            line.append(variance6)
            line.append(std6)
            line.append(iqr6)
            line.append(skew6)
            line.append(kurtosis6)
            line.append(standard_error_mean6)
            line.append(median_abs_deviation6)

            line.append(shannon_ent7)
            line.append(mean7)
            line.append(variance7)
            line.append(std7)
            line.append(iqr7)
            line.append(skew7)
            line.append(kurtosis7)
            line.append(standard_error_mean7)
            line.append(median_abs_deviation7)

            line.append(shannon_ent8)
            line.append(mean8)
            line.append(variance8)
            line.append(std8)
            line.append(iqr8)
            line.append(skew8)
            line.append(kurtosis8)
            line.append(standard_error_mean8)
            line.append(median_abs_deviation8)

        data_writer.writerow(line)
        files_done += 1

        utils.progress_bar(files_done, len(file_list))

    dataCSV.close()

    return


def dwt_feature_extraction_V4(audio_path, dwt_feature_csv, samplerate, wavelet="db1"):
    """
    Extract Detail- and Approx- Coeffs, build features from them and write results to csv file
    """

    file_list = os.listdir(audio_path)
    files_done = 0

    dataCSV = open(dwt_feature_csv, 'w+', encoding='UTF8', newline='')
    data_writer = csv.writer(dataCSV)
    csvHeader = ["audio_file"]
    # write first csv line
    for i in range(1, 15):  # second parameter is dwt max level+2
        csvHeader.append("entropie_" + str(i))
        csvHeader.append("mean_" + str(i))
        csvHeader.append("variance_" + str(i))
        csvHeader.append("std_" + str(i))
        csvHeader.append("iqr_" + str(i))
        csvHeader.append("skew_" + str(i))
        csvHeader.append("kurtosis_" + str(i))
        csvHeader.append("standard_error_mean_" + str(i))
        csvHeader.append("median_abs_deviation_" + str(i))
        csvHeader.append("energy_" + str(i))
        csvHeader.append("avg_power_" + str(i))
        csvHeader.append("zero_cross_rate_" + str(i))
    data_writer.writerow(csvHeader)

    for file in file_list:
        file_path = audio_path + "/" + file

        # 1. Read Audio file
        data, sr = librosa.load(file_path, sr=samplerate)
        data = data[614:131686]  # results in center 131.072 samples of 3 sec clip
        # max_level = pywt.dwt_max_level(len(data), wavelet) - 1

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
            energy = sum(abs(coeffs[i]**2))
            avg_power = (1 / (2*len(coeffs[i]) + 1)) * energy
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
            line.append(energy)
            line.append(avg_power)
            line.append(zero_cross_rate)

        # zero_crossing_rate

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
        csvHeader.append("energy_" + str(i))
        csvHeader.append("avg_power_" + str(i))
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
        csvHeader.append("energy_" + str(i + 1))
        csvHeader.append("avg_power_" + str(i + 1))
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
        csvHeader.append("energy_" + str(i + 2))
        csvHeader.append("avg_power_" + str(i + 2))
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
        csvHeader.append("energy_" + str(i + 3))
        csvHeader.append("avg_power_" + str(i + 3))
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
        csvHeader.append("energy_" + str(i + 4))
        csvHeader.append("avg_power_" + str(i + 4))
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
            energy1 = sum(abs(coeffs1[i] ** 2))
            energy2 = sum(abs(coeffs2[i] ** 2))
            energy3 = sum(abs(coeffs3[i] ** 2))
            energy4 = sum(abs(coeffs4[i] ** 2))
            energy5 = sum(abs(coeffs5[i] ** 2))
            avg_power1 = (1 / (2 * len(coeffs1[i]) + 1)) * energy1
            avg_power2 = (1 / (2 * len(coeffs2[i]) + 1)) * energy2
            avg_power3 = (1 / (2 * len(coeffs3[i]) + 1)) * energy3
            avg_power4 = (1 / (2 * len(coeffs4[i]) + 1)) * energy4
            avg_power5 = (1 / (2 * len(coeffs5[i]) + 1)) * energy5

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
            line.append(energy1)
            line.append(avg_power1)
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
            line.append(energy2)
            line.append(avg_power2)
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
            line.append(energy3)
            line.append(avg_power3)
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
            line.append(energy4)
            line.append(avg_power4)
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
            line.append(energy5)
            line.append(avg_power5)
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

