import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
import preprocess as pp
import evaluation as eva
import splitdata as split


# Path Parameters
SPECTROGRAMM_PATH = "res/img_test"
AUDIO_PATH = "res/audio"
LONG_AUDIO_PATH = "res/longaudio2"
METADATA_CSV = "metadata/UrbanSound8K.csv"                              # main metadata csv from UrbandSound8K
TRAIN_CSV, TEST_CSV = "metadata/Trainfiles.csv", "metadata/Testfiles.csv"  # csv's for normal single training
CROSS_VAL_RANDOM_CSV = "metadata/RandomCrossVal.csv"                    # path of csv used for random cross validation
DEF_FOLDS_PATH = "metadata/def_folds"                                   # path of csv's contain predefined fold infos

# Script Tasks
collect_long_files = False
create_stft_spectrograms = False
create_dwt_spectrograms = False
create_cwt_spectrograms = False
split_data = False
create_cross_val_csv = False
build_and_train = True

# Preprocess Parameters
DFT_FREQ_SCALE = "mel"
FRAME_SIZE = 1024
HOP_SIZE = 256
IMG_SIZE_X, IMG_SIZE_Y = 64, 64
MY_DPI = 77  # weirdly not working with the actual dpi of the monitor, just play around with this value until it works

# Training Parameters
VAL_SET_PERCENTAGE = 10
TRAIN_EPOCHS = 3
# BATCH_SIZE = 0
# TRAINING_RATE = 0
# DROPOUT_RATE = 0

# Evalutation Parameters
USE_DEF_CROSS_VAL = False
USE_RAND_CROSS_VAL = False
CROSS_VAL_FOLDS = 2

CLASS_NAMES = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling',
               'gun_shot', 'jackhammer', 'siren', 'street_music']

# COLLECT FILES LONGER THEN 1 SEC
if collect_long_files:
    pp.CollectLongFiles(original_path=AUDIO_PATH, target_path=LONG_AUDIO_PATH, min_duration=1)

# CREATE SPECTROGRAMS
# use Short Time Fourier Transform
if create_stft_spectrograms:
    pp.CreateSTFTSpectrograms(LONG_AUDIO_PATH, SPECTROGRAMM_PATH, FrameSize=FRAME_SIZE, HopSize=HOP_SIZE,
                              freq_scale=DFT_FREQ_SCALE, px_x=IMG_SIZE_X, px_y=IMG_SIZE_Y, monitor_dpi=MY_DPI,
                              fill_mode="silence")  # "duplicate" is possible aswell

# use Wavelet Transform
if create_dwt_spectrograms:
    pp.CreateDWTScaleogram()
if create_cwt_spectrograms:
    pp.CreateCWTScaleogram()

# SPLIT DATA
if split_data:
    files = split.load_file_names(LONG_AUDIO_PATH)
    split.split_csv(files, METADATA_CSV, TRAIN_CSV, TEST_CSV, 80)
if create_cross_val_csv:
    files = split.load_file_names(LONG_AUDIO_PATH)
    split.create_cross_val_csv(files, METADATA_CSV, CROSS_VAL_RANDOM_CSV)


def Build_Train_Test_Model(train_data, train_labels, val_data, val_labels):
    # CREATE MODEL CNN ARCHITECTURE
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(IMG_SIZE_X, IMG_SIZE_Y, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # TRAIN MODEL
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # austauschen z.b hinge loss
                  metrics=['accuracy'])

    history = model.fit(train_data, train_labels, epochs=TRAIN_EPOCHS,
                        validation_data=(val_data, val_labels))

    return model, history


if build_and_train:
    # If Cross-Validation is used:
    if USE_DEF_CROSS_VAL:
        for i in range(1, 11):
            print("<--- TRAINING " + str(i) + "/" + str(10) + " --->")

            # Split data and get train, val und test dataset (image data and labels)
            X_train, y_train, X_val, y_val, X_test, y_test = split.get_def_cross_val_arrays(index=i,
                                                                                            csv_path=DEF_FOLDS_PATH, img_path=SPECTROGRAMM_PATH,
                                                                                            px_x=IMG_SIZE_X, px_y=IMG_SIZE_Y)

            # NORMALIZE pixel values to be between 0 and 1
            X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0

            # Train Model
            model, history = Build_Train_Test_Model(X_train, y_train, X_val, y_val)

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
        eva.EvaluteCrossValidation(models, histories, acc, loss)
        eva.EvaluteCrossValidation(models, histories, acc, loss, all_pred, all_test_labels, CLASS_NAMES)

    if USE_RAND_CROSS_VAL:
        X, y = pp.GenerateArraysCrossVal(CROSS_VAL_RANDOM_CSV, SPECTROGRAMM_PATH, IMG_SIZE_X, IMG_SIZE_Y)
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
            X_val, part_X_train, y_val, part_y_train = split.create_validation_dataset(X_train, y_train, VAL_SET_PERCENTAGE)

            # Train Model
            model, history = Build_Train_Test_Model(part_X_train, part_y_train, X_val, y_val)

            # Collect evaluation data
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
            if counter == 0:
                models, histories, acc, loss = [model], [history], [test_acc], [test_loss]
            else:
                models.append(model)
                print(models)
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
        eva.EvaluteCrossValidation(models, histories, acc, loss, all_pred, all_test_labels, CLASS_NAMES)

    # If single training is used:
    else:
        # LOAD DATASET
        trainImages, trainLabels, testImages, testLabels = pp.GenerateArrays(TRAIN_CSV, TEST_CSV, SPECTROGRAMM_PATH,
                                                                             IMG_SIZE_X, IMG_SIZE_Y)
        print("Finished Loading Data")

        # NORMALIZE pixel values to be between 0 and 1
        trainImages, testImages = trainImages / 255.0, testImages / 255.0
        print("Normalized Datapoints")

        # Create Validation Dataset
        data_val, part_data_train, labels_val, part_labels_train = split.create_validation_dataset(trainImages,
                                                                                                   trainLabels, VAL_SET_PERCENTAGE)

        print("<--- TRAINING 1/1 ---")
        model, history = Build_Train_Test_Model(part_data_train, part_labels_train, data_val, labels_val)
        histories = [history]

        # EVALUATE MODEL
        print("------------------- \nHistory:")
        eva.evaluate_epochs(histories)
        print("---------------------------- \nEvaluation of Model:")
        test_loss, test_acc = model.evaluate(testImages, testLabels, verbose=2)
        eva.Show_Confusion_Matrix(CLASS_NAMES, model, test_acc, testImages, testLabels)

