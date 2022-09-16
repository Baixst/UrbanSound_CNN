import tensorflow as tf
from tensorflow import keras
import datetime

def Build_Train_CNN2D(train_data, train_labels, test_data, test_labels, epochs, img_size_x, img_size_y):
    print("TRAINING OWN DEFAULT CNN")

    # CREATE MODEL CNN ARCHITECTURE
    """
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5, 5), input_shape=(img_size_x, img_size_y, 1),  padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.Flatten())
    model.add(keras.layers.GlobalAveragePooling2D())
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    """

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5, 5), input_shape=(img_size_x, img_size_y, 1), padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.summary()

    # TRAIN MODEL
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # austauschen z.b hinge loss
                  metrics=['accuracy'])

    startTime = datetime.datetime.now()

    history = model.fit(train_data, train_labels, epochs=epochs,
                        validation_data=(test_data, test_labels))  # ,callbacks=[WandbCallback()]

    trainingTime = datetime.datetime.now() - startTime
    print("Time until training finished: " + str(trainingTime))

    return model, history


def Build_Train_ResNet50(train_data, train_labels, test_data, test_labels, epochs):
    print("TRAINING RESNET50")

    # Load ResNet and freeze all layers
    base_model = keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        if 'BatchNormalization' not in layer.__class__.__name__:
            layer.trainable = False

    # add own dense layers for classification
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    # x = keras.layers.Dense(1000, activation='relu')(x)
    predictions = keras.layers.Dense(10, activation='softmax')(x)

    head_model = keras.Model(inputs=base_model.input, outputs=predictions)
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    head_model.compile(optimizer=optimizer,
                       loss=keras.losses.sparse_categorical_crossentropy,
                       metrics=['accuracy'])

    head_model.summary()

    startTime = datetime.datetime.now()

    history = head_model.fit(train_data, train_labels, epochs=epochs,
                        validation_data=(test_data, test_labels))  # ,callbacks=[WandbCallback()]

    trainingTime = datetime.datetime.now() - startTime
    print("Time until training finished: " + str(trainingTime))

    return head_model, history


def Build_Train_MaxPool1D(train_data, train_labels, test_data, test_labels, epochs, audio_duration, samplerate):
    inputs = int(audio_duration * samplerate)

    # CREATE MODEL ARCHITECTURE
    model = keras.Sequential()
    model.add(keras.layers.MaxPooling1D(pool_size=4, strides=4, input_shape=(inputs, 1)))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.Conv1D(64, 4, strides=2, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Conv1D(128, 2, activation='relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # TRAIN MODEL
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # austauschen z.b hinge loss
                  metrics=['accuracy'])

    history = model.fit(train_data, train_labels, epochs=epochs,
                        validation_data=(test_data, test_labels))  # ,callbacks=[WandbCallback()]

    return model, history


def Build_Train_Dense(train_data, train_labels, test_data, test_labels, epochs, amount_features):
    # CREATE MODEL CNN ARCHITECTURE
    model = keras.Sequential()
    model.add(keras.layers.Dense(amount_features, activation='sigmoid', input_shape=(amount_features,)))
    model.add(keras.layers.Dense(110, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(60, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(30, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    # TRAIN MODEL
    optimizer = keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # austauschen z.b hinge loss
                  metrics=['accuracy'])

    history = model.fit(train_data, train_labels, epochs=epochs,
                        validation_data=(test_data, test_labels))  # ,callbacks=[WandbCallback()]

    return model, history