import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
import os
import preprocess as pp
import evaluation as eva
import splitdata as split
import training as train
import loading as loader

# trying to solve out of memory error
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
#  tf.config.experimental.set_visible_devices([], 'GPU')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# import wandb
# from wandb.keras import WandbCallback

# run = wandb.init(project="my-test-project", entity="baixst", config={ "image_type": "stft", "learning_rate": 0.01,
#                                                                       "epochs": 10, "batch_size": 128})
# config = wandb.config

# Path Parameters
AUDIO_PATH = "res/audio"
IMAGE_PATH = "res/img_3sec_cen_256x128_22khz"
METADATA_CSV = "metadata/UrbanSound8K.csv"                              # main metadata csv from UrbandSound8K
TRAIN_CSV, TEST_CSV = "metadata/Trainfiles.csv", "metadata/Testfiles.csv"  # csv's for normal single training
CROSS_VAL_RANDOM_CSV = "metadata/RandomCrossVal.csv"                    # path of csv used for random cross validation
DEF_FOLDS_PATH = "metadata/def_folds"                                   # path of csv's contain predefined fold infos

# Script Tasks
create_spectrograms = False
collect_dwt_data = False
create_cwt_scalograms = False
split_data = False
create_cross_val_csv = False
build_and_train_STFT = True
build_and_train_DWT = False
build_and_train_Raw_MaxPool = False

# Preprocess Parameters
SPECTROGRAM_TYPE = "mel"
SPEC_FREQ_SCALE = "mel"
FRAME_SIZE = 1024
HOP_SIZE = 256
MEL_BINS = 128
CWT_FREQ_SCALES = 64
CWT_WAVELET = "morl"

# Image Parameters
IMG_SIZE_X, IMG_SIZE_Y = 256, 128
MY_DPI = 77  # weirdly not working with the actual dpi of the monitor, just play around with this value until it works

# Training Parameters
TRAIN_EPOCHS = 15  # config.get("epochs")
# VAL_SET_PERCENTAGE = 10
# BATCH_SIZE = 0
# TRAINING_RATE = 0
# DROPOUT_RATE = 0

# Evalutation Parameters
USE_DEF_CROSS_VAL = False
USE_RAND_CROSS_VAL = False
CROSS_VAL_FOLDS = 10

CLASS_NAMES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling',
               'gun_shot', 'jackhammer', 'siren', 'street_music']

# CREATE SPECTROGRAMS
# use Short Time Fourier Transform
if create_spectrograms:
    pp.CreateSTFTSpectrograms(AUDIO_PATH, IMAGE_PATH, FrameSize=FRAME_SIZE, HopSize=HOP_SIZE, mels=MEL_BINS, duration=4,
                              freq_scale=SPEC_FREQ_SCALE, px_x=IMG_SIZE_X, px_y=IMG_SIZE_Y, monitor_dpi=MY_DPI,
                              spec_type=SPECTROGRAM_TYPE, samplerate=44100,
                              fill_mode="none")  # "centered", "none", "silence" or "duplicate"

# use Wavelet Transform
if collect_dwt_data:
    pp.dwt_feature_extraction(AUDIO_PATH, "res/dwt_features.csv")
if create_cwt_scalograms:
    pp.CreateCWTScaleograms(AUDIO_PATH, IMAGE_PATH, freq_scales=CWT_FREQ_SCALES, wavelet=CWT_WAVELET,
                            px_x=IMG_SIZE_X, px_y=IMG_SIZE_Y, monitor_dpi=MY_DPI, fill_mode="centered")

# SPLIT DATA
if split_data:
    files = split.load_file_names(IMAGE_PATH)
    split.split_csv(files, METADATA_CSV, TRAIN_CSV, TEST_CSV, 80)
if create_cross_val_csv:
    files = split.load_file_names(AUDIO_PATH)
    split.create_cross_val_csv(files, METADATA_CSV, CROSS_VAL_RANDOM_CSV)

# TRAIN AND TEST MODEL
if build_and_train_STFT:
    # If Cross-Validation is used:
    if USE_DEF_CROSS_VAL:
        for i in range(1, 11):
            print("<--- TRAINING " + str(i) + "/" + str(10) + " --->")

            # Split data and get train and test dataset (image data and labels)
            X_train, y_train, X_test, y_test = split.get_def_cross_val_arrays(index=i, csv_path=DEF_FOLDS_PATH,
                                                                              img_path=IMAGE_PATH, px_x=IMG_SIZE_X, px_y=IMG_SIZE_Y)

            # NORMALIZE pixel values to be between 0 and 1
            X_train, X_test = X_train / 255.0, X_test / 255.0

            # Train Model
            model, history = train.Build_Train_CNN2D(X_train, y_train, X_test, y_test, epochs=TRAIN_EPOCHS,
                                    img_size_x=IMG_SIZE_X, img_size_y=IMG_SIZE_Y)

            # Collect evaluation data
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
            if i == 1:
                models, histories, acc, loss = [model], [history], [test_acc], [test_loss]
            else:
                models.append(model)
                histories.append(history)
                acc.append(test_acc)
                loss.append(test_loss)

            # Preperation for Confusion Matrix:
            if i == 0:
                all_pred = models[i].predict(X_test)
                all_pred = tf.argmax(all_pred, axis=-1)
                all_test_labels = y_test
            else:
                tmp_pred = models[i].predict(X_test)
                tmp_pred = tf.argmax(tmp_pred, axis=-1)
                tmp_test_labels = y_test

                all_pred = tf.concat([all_pred, tmp_pred], 0)
                all_test_labels = np.concatenate((all_test_labels, tmp_test_labels))

        # EVALUATE MODEL
        eva.EvaluteCrossValidation(histories, acc, loss, all_pred, all_test_labels, CLASS_NAMES)

    if USE_RAND_CROSS_VAL:
        X, y = loader.GenerateArraysCrossVal(CROSS_VAL_RANDOM_CSV, IMAGE_PATH, IMG_SIZE_X, IMG_SIZE_Y)
        # NORMALIZE pixel values to be between 0 and 1
        X = X / 255.0

        counter = 0
        kf = KFold(n_splits=CROSS_VAL_FOLDS, shuffle=False)
        for train_index, test_index in kf.split(X):
            print("<--- TRAINING " + str(counter+1) + "/" + str(CROSS_VAL_FOLDS) + " --->")

            # Get Folds
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Split train data further in train and validation
            # X_val, part_X_train, y_val, part_y_train = split.create_validation_dataset(X_train, y_train, VAL_SET_PERCENTAGE)

            # Train Model
            model, history = train.Build_Train_CNN2D(X_train, y_train, X_test, y_test, epochs=TRAIN_EPOCHS,
                                    img_size_x=IMG_SIZE_X, img_size_y=IMG_SIZE_Y)

            # Collect evaluation data
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
            if counter == 0:
                models, histories, acc, loss = [model], [history], [test_acc], [test_loss]
            else:
                models.append(model)
                histories.append(history)
                acc.append(test_acc)
                loss.append(test_loss)

            # Preperation for Confusion Matrix:
            if counter == 0:
                all_pred = models[counter].predict(X_test)
                all_pred = tf.argmax(all_pred, axis=-1)
                all_test_labels = y_test
            else:
                tmp_pred = models[counter].predict(X_test)
                tmp_pred = tf.argmax(tmp_pred, axis=-1)
                tmp_test_labels = y_test

                all_pred = tf.concat([all_pred, tmp_pred], 0)
                all_test_labels = np.concatenate((all_test_labels, tmp_test_labels))

            counter += 1

        # EVALUATE MODEL
        eva.EvaluteCrossValidation(histories, acc, loss, all_pred, all_test_labels, CLASS_NAMES)

    # If single training is used:
    else:
        # LOAD DATASET
        trainImages, trainLabels, testImages, testLabels = loader.GenerateArrays_STFT(TRAIN_CSV, TEST_CSV, IMAGE_PATH,
                                                                                  IMG_SIZE_X, IMG_SIZE_Y)
        print("Finished Loading Data")

        # NORMALIZE pixel values to be between 0 and 1
        trainImages, testImages = trainImages / 255.0, testImages / 255.0
        print("Normalized Datapoints")

        # Create Validation Dataset
        # data_val, part_data_train, labels_val, part_labels_train = split.create_validation_dataset(trainImages,
        #                                                                          trainLabels, VAL_SET_PERCENTAGE)

        print("<--- TRAINING 1/1 ---")
        model, history = train.Build_Train_CNN2D(trainImages, trainLabels, testImages, testLabels, epochs=TRAIN_EPOCHS,
                                img_size_x=IMG_SIZE_X, img_size_y=IMG_SIZE_Y)

        histories = [history]

        # EVALUATE MODEL
        print("------------------- \nHistory:")
        eva.evaluate_epochs(histories)
        print("---------------------------- \nEvaluation of Model:")
        test_loss, test_acc = model.evaluate(testImages, testLabels, verbose=2)
        eva.Show_Confusion_Matrix(CLASS_NAMES, model, test_acc, testImages, testLabels)

    # run.finish()

if build_and_train_DWT:
    # LOAD DATASET
    trainFeat, trainLabels, testFeat, testLabels = loader.GenerateArrays_DWT(TRAIN_CSV, TEST_CSV,
                                                                                  "res/dwt_features.csv")
    print("Finished Loading Data")

    print("<--- TRAINING 1/1 ---")
    model, history = train.Build_Train_Dense(trainFeat, trainLabels, testFeat, testLabels, epochs=TRAIN_EPOCHS,
                                             amount_features=126)

    histories = [history]

    # EVALUATE MODEL
    print("------------------- \nHistory:")
    eva.evaluate_epochs(histories)
    print("---------------------------- \nEvaluation of Model:")
    test_loss, test_acc = model.evaluate(testFeat, testLabels, verbose=2)
    eva.Show_Confusion_Matrix(CLASS_NAMES, model, test_acc, testFeat, testLabels)

if build_and_train_Raw_MaxPool:
    # LOAD DATASET
    trainData, trainLabels, testData, testLabels = loader.GenerateArrays_Raw(TRAIN_CSV, TEST_CSV, AUDIO_PATH,
                                                                             3, 22050)
    print("Finished Loading Data")

    print("<--- TRAINING 1/1 ---")
    model, history = train.Build_Train_MaxPool1D(trainData, trainLabels, testData, testLabels, epochs=TRAIN_EPOCHS,
                                             audio_duration=3, samplerate=22050)

    histories = [history]

    # EVALUATE MODEL
    print("------------------- \nHistory:")
    eva.evaluate_epochs(histories)
    print("---------------------------- \nEvaluation of Model:")
    test_loss, test_acc = model.evaluate(testData, testLabels, verbose=2)
    eva.Show_Confusion_Matrix(CLASS_NAMES, model, test_acc, testData, testLabels)