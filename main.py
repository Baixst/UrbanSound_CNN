import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import preprocess
import preprocess as pp
import utils
import splitdata
import tensorflow as tf
from tensorflow import keras
import pandas as pd

# COLLECT FILES LONGER THEN 1 SEC
# pp.CollectLongFiles(original_path="E:/Users/Basti/Documents/FirstCNN_Res/audio",
#                    target_path="E:/Users/Basti/Documents/FirstCNN_Res/longaudio")

# CREATE SPECTROGRAMS
# pp.CreateSpectrograms("E:/Users/Basti/Documents/FirstCNN_Res/longaudio", "E:/Users/Basti/Documents/FirstCNN_Res/img",
#                       FrameSize=1024, HopSize=256)

# SPLIT DATA
# files = splitdata.load_image_names("E:/Users/Basti/Documents/FirstCNN_Res/img")
# splitdata.split_data_in_two(files, 80, "E:/Users/Basti/Documents/FirstCNN_Res/img",
#                             "E:/Users/Basti/Documents/FirstCNN_Res/test", "E:/Users/Basti/Documents/FirstCNN_Res/train")

# LOAD DATASET
train_images, train_labels, test_images, test_labels = pp.GenerateArrays(train_path="E:/Users/Basti/Documents/FirstCNN_Res/train",
                                                                         test_path="E:/Users/Basti/Documents/FirstCNN_Res/test",
                                                                         csv_file="UrbanSound8K.csv")
print("Finished Loading Data")

# NORMALIZE pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
print("Normalized Datapoints")

class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling',
               'gun_shot', 'jackhammer', 'siren', 'street_music']

# Let's look at a one image

# IMG_INDEX = 1  # change this to look at other images
# plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
# plt.xlabel(class_names[int(train_labels[IMG_INDEX][1])])
# plt.show()

def Build_Train_Test_Model():
    # CREATE MODEL CNN ARCHITECTURE
    print("Setting CNN Architecture")
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10))

    # TRAIN MODEL
    print("starting to train model...")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=5,
                        validation_data=(test_images, test_labels))

    # EVALUATE MODEL
    print("---------------------------- \nEvaluation of Model:")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc)

    return


Build_Train_Test_Model()
