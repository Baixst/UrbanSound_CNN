import matplotlib.pyplot as plt
import preprocess as pp
import splitdata
import tensorflow as tf
from tensorflow import keras
import seaborn as sns

# Path Parameters
spectrogram_path = "res/img"
audio_path = "res/audio"
longaudio_path = "res/longaudio"
metadata_csv = "metadata/UrbanSound8K.csv"
train_csv, test_csv = "metadata/Trainfiles.csv", "metadata/Testfiles.csv"

# Script Tasks
collect_long_files = False
create_spectrograms = True
split_data = True
load_dataset = True
build_and_train = True

# Preprocess Parameters
dft_freq_scale = "log"
frame_size = 1024
hop_size = 256
img_size_x, img_size_y = 128, 128
my_dpi = 77  # weirdly not working with the actual dpi of the monitor, just play around with this value until it works

# Training Parameters
train_epochs = 5

class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling',
               'gun_shot', 'jackhammer', 'siren', 'street_music']

# COLLECT FILES LONGER THEN 1 SEC
if collect_long_files:
    pp.CollectLongFiles(original_path=audio_path, target_path=longaudio_path, min_duration=1)

# CREATE SPECTROGRAMS
if create_spectrograms:
    pp.CreateSpectrograms(longaudio_path, spectrogram_path, FrameSize=frame_size, HopSize=hop_size,
                          freq_scale=dft_freq_scale, px_x=img_size_x, px_y=img_size_y, monitor_dpi=my_dpi)

# SPLIT DATA
if split_data:
    files = splitdata.load_file_names(longaudio_path)
    splitdata.split_csv(files, metadata_csv, train_csv, test_csv, 80)

# LOAD DATASET
if load_dataset:
    train_images, train_labels, test_images, test_labels = pp.GenerateArrays(train_csv, test_csv, spectrogram_path, img_size_x, img_size_y)
    print("Finished Loading Data")

    # NORMALIZE pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    print("Normalized Datapoints")

# Let's look at a one image
# IMG_INDEX = 1  # change this to look at other images
# plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
# plt.xlabel(class_names[int(train_labels[IMG_INDEX][1])])
# plt.show()

def Build_Train_Test_Model():
    # CREATE MODEL CNN ARCHITECTURE
    print("Setting CNN Architecture")
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(img_size_x, img_size_y, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # TRAIN MODEL
    print("starting to train model...")
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #austauschen z.b hinge loss
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=train_epochs,
                        validation_data=(test_images, test_labels))

    # EVALUATE MODEL
    print("---------------------------- \nEvaluation of Model:")
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    Show_Confusion_Matrix(model, test_acc)

    return


def Show_Confusion_Matrix(model, test_acc):

    predictions = model.predict(test_images)
    predictions = tf.argmax(predictions, axis=-1)
    cm = tf.math.confusion_matrix(test_labels, predictions)
    cm = cm / cm.numpy().sum(axis=1)[:, tf.newaxis]

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True,
        xticklabels=class_names,
        yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fig.suptitle(("Overall Accuracy = " + str(round(test_acc, 3))))
    plt.subplots_adjust(left=0.185, bottom=0.225, right=1, top=0.89, wspace=0.2, hspace=0.2)
    plt.show()

    return


if build_and_train:
    Build_Train_Test_Model()
