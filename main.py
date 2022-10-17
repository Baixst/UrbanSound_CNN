import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
import preprocess as pp
import evaluation as eva
import splitdata as split
import training as train
import loading as loader

# Use CPU
# tf.config.experimental.set_visible_devices([], 'GPU')

# trying to solve out of memory error
'''
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

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
'''

# Path Parameters
AUDIO_PATH = "res/audio_3sec_duplicated_44khz"                             # not used for training, only for preprocessing tasks
IMAGE_PATH = "res/img_4sec_cen_224x224_44khz"
METADATA_CSV = "metadata/UrbanSound8K.csv"                                 # main metadata csv from UrbandSound8K
DWT_FEATURES_CSV = "res/dwt_features_V5_db1_less.csv"               # dwt features for training dense net
CURRENT_FOLD = 1                                                # used for cross-val when each fold is run on it's own
TRAIN_CSV = "metadata/Trainfiles.csv"
TEST_CSV = "metadata/Testfiles.csv"                               # csv's for normal single training
CROSS_VAL_RANDOM_CSV = "metadata/RandomCrossVal.csv"                    # path of csv used for random cross validation
DEF_FOLDS_PATH = "metadata/def_folds"                                   # path of csv's contain predefined fold infos
CROSS_VAL_RESULTS = "results/crossVal_results.csv"          # contains acc + loss results for manual cross val
CROSS_VAL_PREDICTIONS = "results/crossVal_predictions.csv"  # contains predictions + lables for manual cross val

# Script Tasks
create_spectrograms = False
collect_dwt_data = False
create_cwt_scalograms = False
split_data = False
create_cross_val_csv = False
build_and_train_STFT = False
stft_model_to_use = "default"         # "default", "ResNet", "own_ResNet" is possible
build_and_train_DWT = False
manual_evaluation = False
predict_from_checkpoint = True
model_to_eval = "default"         # "default", "ResNet", "own_ResNet", "dense_dwt" is possible

# Preprocess Parameters
SPECTROGRAM_TYPE = "mel"
SPEC_FREQ_SCALE = "mel"
FRAME_SIZE = 2048
HOP_SIZE = 512
MEL_BINS = 128
CWT_FREQ_SCALES = 64
CWT_WAVELET = "morl"
DWT_WAVELET = "db1"

# Image Parameters
IMG_SIZE_X, IMG_SIZE_Y = 224, 224
MY_DPI = 77  # weirdly not working with the actual dpi of the monitor, just play around with this value until it works

# Training Parameters
TRAIN_EPOCHS = 50

# Evalutation Parameters
USE_DEF_CROSS_VAL = False
USE_RAND_CROSS_VAL = False
CROSS_VAL_FOLDS = 4


CLASS_NAMES = ['Klimaanlage', 'Hupe', 'Kinder', 'Bellen', 'Bohren', 'Motor',
               'Schüsse', 'Presslufthammer', 'Sirenen', 'Straßenmusik']

# CREATE SPECTROGRAMS
# use Short Time Fourier Transform
if create_spectrograms:
    pp.CreateSTFTSpectrograms(AUDIO_PATH, IMAGE_PATH, FrameSize=FRAME_SIZE, HopSize=HOP_SIZE, mels=MEL_BINS, duration=4,
                              freq_scale=SPEC_FREQ_SCALE, px_x=IMG_SIZE_X, px_y=IMG_SIZE_Y, monitor_dpi=MY_DPI,
                              spec_type=SPECTROGRAM_TYPE, samplerate=44100,
                              fill_mode="none")  # "centered", "none", "silence" or "duplicate"

# use Wavelet Transform
if collect_dwt_data:
    pp.dwt_feature_extraction_V5(AUDIO_PATH, DWT_FEATURES_CSV, 44100, wavelet=DWT_WAVELET)
if create_cwt_scalograms:
    pp.CreateCWTScaleograms(AUDIO_PATH, IMAGE_PATH, freq_scales=CWT_FREQ_SCALES, wavelet=CWT_WAVELET,
                            px_x=IMG_SIZE_X, px_y=IMG_SIZE_Y, monitor_dpi=MY_DPI, fill_mode="centered")

# SPLIT DATA
if split_data:
    files = split.load_file_names(IMAGE_PATH)
    split.split_csv(files, METADATA_CSV, TRAIN_CSV, TEST_CSV, 80)
if create_cross_val_csv:
    files = split.load_file_names(IMAGE_PATH)
    split.create_cross_val_csv(files, METADATA_CSV, CROSS_VAL_RANDOM_CSV)

# TRAIN AND TEST MODEL
if build_and_train_STFT:
    # If Cross-Validation is used:
    if USE_DEF_CROSS_VAL:
        for i in range(1, 11):
            print("<--- TRAINING " + str(i) + "/" + str(10) + " --->")

            # Split data and get train and test dataset (image data and labels)
            X_train, y_train, X_test, y_test = loader.GenerateArraysDefCrossVal(index=i, csv_path=DEF_FOLDS_PATH,
                                                                img_path=IMAGE_PATH, px_x=IMG_SIZE_X, px_y=IMG_SIZE_Y)

            # NORMALIZE pixel values to be between 0 and 1
            X_train, X_test = X_train / 255.0, X_test / 255.0

            # Train Model
            if stft_model_to_use == "ResNet":
                # von (-1, px_x, px_y, 1) auf (-1, px_x, px_y, 3) Tensor erhöhen
                X_train = tf.repeat(X_train, 3, axis=3)
                X_test = tf.repeat(X_test, 3, axis=3)
                model, history = train.Build_Train_ResNet50(X_train, y_train, X_test, y_test, epochs=TRAIN_EPOCHS)

            if stft_model_to_use == "default":
                model, history = train.Build_Train_CNN2D(X_train, y_train, X_test, y_test, epochs=TRAIN_EPOCHS,
                                    img_size_x=IMG_SIZE_X, img_size_y=IMG_SIZE_Y)

            if stft_model_to_use == "own_ResNet":
                model, history = train.Build_Train_OwnResNet(X_train, y_train, X_test, y_test, epochs=TRAIN_EPOCHS,
                                                         img_size_x=IMG_SIZE_X, img_size_y=IMG_SIZE_Y)

            # Collect evaluation data
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
            if i == 1:
                histories, acc, loss = [history], [test_acc], [test_loss]
            else:
                histories.append(history)
                acc.append(test_acc)
                loss.append(test_loss)

            # Preperation for Confusion Matrix:
            if i == 1:
                all_pred = model.predict(X_test)
                all_pred = tf.argmax(all_pred, axis=-1)
                all_test_labels = y_test
            else:
                tmp_pred = model.predict(X_test)
                tmp_pred = tf.argmax(tmp_pred, axis=-1)
                tmp_test_labels = y_test

                all_pred = tf.concat([all_pred, tmp_pred], 0)
                all_test_labels = np.concatenate((all_test_labels, tmp_test_labels))

        # EVALUATE MODEL
        eva.EvaluteCrossValidation(histories, acc, loss, all_pred, all_test_labels, CLASS_NAMES)

    elif USE_RAND_CROSS_VAL:
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
            if stft_model_to_use == "default":
                model, history = train.Build_Train_CNN2D(X_train, y_train, X_test, y_test, epochs=TRAIN_EPOCHS,
                                                         img_size_x=IMG_SIZE_X, img_size_y=IMG_SIZE_Y)

            if stft_model_to_use == "ResNet":
                # Auf von (-1, px_x, px_y, 1) auf (-1, px_x, px_y, 3) Tensor erhöhen
                X_train = tf.repeat(X_train, 3, axis=3)
                X_test = tf.repeat(X_test, 3, axis=3)
                model, history = train.Build_Train_ResNet50(X_train, y_train, X_test, y_test, epochs=TRAIN_EPOCHS)

            if stft_model_to_use == "own_ResNet":
                model, history = train.Build_Train_OwnResNet(X_train, y_train, X_test, y_test, epochs=TRAIN_EPOCHS,
                                                         img_size_x=IMG_SIZE_X, img_size_y=IMG_SIZE_Y)

            # Collect evaluation data
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
            if counter == 0:
                histories, acc, loss = [history], [test_acc], [test_loss]
            else:
                histories.append(history)
                acc.append(test_acc)
                loss.append(test_loss)

            # Preperation for Confusion Matrix:
            if counter == 0:
                all_pred = model.predict(X_test)
                all_pred = tf.argmax(all_pred, axis=-1)
                all_test_labels = y_test
            else:
                tmp_pred = model.predict(X_test)
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

        print("<--- TRAINING 1/1 ---")
        # Train Model
        if stft_model_to_use == "default":
            checkpoint_path = "models/default_cnn/fold" + str(CURRENT_FOLD) + "/"
            model, history = train.Build_Train_CNN2D(trainImages, trainLabels, testImages, testLabels,
                                                     epochs=TRAIN_EPOCHS, img_size_x=IMG_SIZE_X, img_size_y=IMG_SIZE_Y,
                                                     checkpoint_to_load=checkpoint_path)

        if stft_model_to_use == "ResNet":
            checkpoint_path = "models/ResNet/fold" + str(CURRENT_FOLD) + "/"

            # Auf von (-1, px_x, px_y, 1) auf (-1, px_x, px_y, 3) Tensor erhöhen
            trainImages = tf.repeat(trainImages, 3, axis=3)
            testImages = tf.repeat(testImages, 3, axis=3)
            model, history = train.Build_Train_ResNet50(trainImages, trainLabels, testImages, testLabels,
                                                        epochs=TRAIN_EPOCHS, checkpoint_to_load=checkpoint_path)

        if stft_model_to_use == "own_ResNet":
            checkpoint_path = "models/own_ResNet/fold" + str(CURRENT_FOLD) + "/"
            model, history = train.Build_Train_OwnResNet(trainImages, trainLabels, testImages, testLabels,
                                                    epochs=TRAIN_EPOCHS, img_size_x=IMG_SIZE_X, img_size_y=IMG_SIZE_Y,
                                                         checkpoint_to_load=checkpoint_path)

        histories = [history]

        # EVALUATE MODEL
        print("------------------- \nHistory:")
        eva.evaluate_epochs(histories, fold=CURRENT_FOLD)
        print("---------------------------- \nEvaluation of Model:")
        test_loss, test_acc = model.evaluate(testImages, testLabels, verbose=2)
        eva.Show_Confusion_Matrix(CLASS_NAMES, model, test_acc, testImages, testLabels, CURRENT_FOLD)

    # run.finish()

if build_and_train_DWT:
    if USE_DEF_CROSS_VAL:
        for i in range(1, 11):
            print("<--- TRAINING " + str(i) + "/" + str(10) + " --->")

            # Split data and get train and test dataset (features and labels)
            trainFeat, trainLabels, testFeat, testLabels = loader.GenerateArraysDefCross_DWT(index=i,
                                                                csv_path=DEF_FOLDS_PATH, feature_csv=DWT_FEATURES_CSV)

            # Build and train model
            model, history = train.Build_Train_Dense(trainFeat, trainLabels, testFeat, testLabels, epochs=TRAIN_EPOCHS,
                                                     amount_features=150)

            # Collect evaluation data
            test_loss, test_acc = model.evaluate(testFeat, testLabels, verbose=2)
            if i == 1:
                histories, acc, loss = [history], [test_acc], [test_loss]
            else:
                histories.append(history)
                acc.append(test_acc)
                loss.append(test_loss)

            # Preperation for Confusion Matrix:
            if i == 1:
                all_pred = model.predict(testFeat)
                all_pred = tf.argmax(all_pred, axis=-1)
                all_test_labels = testLabels
            else:
                tmp_pred = model.predict(testFeat)
                tmp_pred = tf.argmax(tmp_pred, axis=-1)
                tmp_test_labels = testLabels

                all_pred = tf.concat([all_pred, tmp_pred], 0)
                all_test_labels = np.concatenate((all_test_labels, tmp_test_labels))

        # EVALUATE MODEL
        eva.EvaluteCrossValidation(histories, acc, loss, all_pred, all_test_labels, CLASS_NAMES)

    else:
        # LOAD DATASET
        trainFeat, trainLabels, testFeat, testLabels = loader.GenerateArrays_DWT(TRAIN_CSV, TEST_CSV, DWT_FEATURES_CSV)
        print("Finished Loading Data")

        print("<--- TRAINING 1/1 ---")
        checkpoint_path = "models/dense_dwt/fold" + str(CURRENT_FOLD) + "/"
        model, history = train.Build_Train_Dense(trainFeat, trainLabels, testFeat, testLabels, epochs=TRAIN_EPOCHS,
                                             amount_features=150)

        histories = [history]

        # EVALUATE MODEL
        print("------------------- \nHistory:")
        eva.evaluate_epochs(histories, fold=CURRENT_FOLD)
        print("---------------------------- \nEvaluation of Model:")
        test_loss, test_acc = model.evaluate(testFeat, testLabels, verbose=2)
        eva.Show_Confusion_Matrix(CLASS_NAMES, model, test_acc, testFeat, testLabels, CURRENT_FOLD)

if manual_evaluation:
    eva.ManualCrossVal_Eval(CLASS_NAMES, CROSS_VAL_RESULTS, CROSS_VAL_PREDICTIONS, 10)

if predict_from_checkpoint:
    if model_to_eval == "dense_dwt":
        checkpoint_file = "models/dense_dwt/fold" + str(CURRENT_FOLD) + "/cp-005"

        trainFeat, trainLabels, testFeat, testLabels = loader.GenerateArrays_DWT(TRAIN_CSV, TEST_CSV, DWT_FEATURES_CSV)
        print("Finished Loading Data")
        model, history = train.Build_Train_Dense(trainFeat, trainLabels, testFeat, testLabels, epochs=TRAIN_EPOCHS,
                                                 amount_features=150, load_weights=True,
                                                 checkpoint_to_load=checkpoint_file)
        predictions = model.predict(testFeat)

    if stft_model_to_use == "default":
        checkpoint_file = "models/default_cnn/fold" + str(CURRENT_FOLD) + "/cp-014"

        trainImages, trainLabels, testImages, testLabels = loader.GenerateArrays_STFT(TRAIN_CSV, TEST_CSV, IMAGE_PATH,
                                                                                      IMG_SIZE_X, IMG_SIZE_Y)
        print("Finished Loading Data")
        model, history = train.Build_Train_CNN2D(trainImages, trainLabels, testImages, testLabels,
                                                 epochs=TRAIN_EPOCHS, img_size_x=IMG_SIZE_X, img_size_y=IMG_SIZE_Y,
                                                 load_weights=True, checkpoint_to_load=checkpoint_file)
        predictions = model.predict(testImages)

    if stft_model_to_use == "ResNet":
        checkpoint_file = "models/ResNet/fold" + str(CURRENT_FOLD) + "/cp-005"

        trainImages, trainLabels, testImages, testLabels = loader.GenerateArrays_STFT(TRAIN_CSV, TEST_CSV, IMAGE_PATH,
                                                                                      IMG_SIZE_X, IMG_SIZE_Y)
        # Auf von (-1, px_x, px_y, 1) auf (-1, px_x, px_y, 3) Tensor erhöhen
        trainImages = tf.repeat(trainImages, 3, axis=3)
        testImages = tf.repeat(testImages, 3, axis=3)
        print("Finished Loading Data")
        model, history = train.Build_Train_ResNet50(trainImages, trainLabels, testImages, testLabels,
                                                    epochs=TRAIN_EPOCHS, load_weights=True,
                                                    checkpoint_to_load=checkpoint_file)
        predictions = model.predict(testImages)

    if stft_model_to_use == "own_ResNet":
        checkpoint_file = "models/own_ResNet/fold" + str(CURRENT_FOLD) + "/cp-005"

        trainImages, trainLabels, testImages, testLabels = loader.GenerateArrays_STFT(TRAIN_CSV, TEST_CSV, IMAGE_PATH,
                                                                                      IMG_SIZE_X, IMG_SIZE_Y)
        print("Finished Loading Data")

        model, history = train.Build_Train_OwnResNet(trainImages, trainLabels, testImages, testLabels,
                                                     epochs=TRAIN_EPOCHS, img_size_x=IMG_SIZE_X, img_size_y=IMG_SIZE_Y,
                                                     load_weights=True, checkpoint_to_load=checkpoint_file)
        predictions = model.predict(testImages)

    predictions = tf.argmax(predictions, axis=-1)
    eva.write_predictions_to_csv(predictions, testLabels, fold=CURRENT_FOLD)

