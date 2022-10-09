import tensorflow as tf
from tensorflow import keras
import datetime

def Build_Train_CNN2D(train_data, train_labels, test_data, test_labels, epochs, img_size_x, img_size_y):
    print("TRAINING OWN DEFAULT CNN")

    # CREATE MODEL CNN ARCHITECTURE
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
    # model.add(keras.layers.Flatten())
    model.add(keras.layers.GlobalAveragePooling2D())
    # model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    """

    # model.summary()

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

    # head_model.summary()

    startTime = datetime.datetime.now()

    history = head_model.fit(train_data, train_labels, epochs=epochs,
                        validation_data=(test_data, test_labels))  # ,callbacks=[WandbCallback()]

    trainingTime = datetime.datetime.now() - startTime
    print("Time until training finished: " + str(trainingTime))

    return head_model, history


def Build_Train_OwnResNet(train_data, train_labels, test_data, test_labels, epochs, img_size_x, img_size_y):
    print("TRAINING OWN RESNET")

    X_input = keras.layers.Input((img_size_x, img_size_y, 1))
    # X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

    # Step 2 (Initial Conv layer along with maxPool)
    X = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(X_input)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(X)

    # Define size of sub-blocks and initial filter size
    block_layers = [2, 2, 2, 2]  # how often a resblock is repeated before decreasing the dimension
    filter_size = 64

    # Step 3 Add the Resnet Blocks
    for i in range(len(block_layers)):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                X = res_ident_block(X, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size * 2
            X = res_conv_block(X, filter_size)
            for j in range(block_layers[i] - 1):
                X = res_ident_block(X, filter_size)

    # Step 4 End Dense Network
    # X = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(X)
    # X = tf.keras.layers.Flatten()(X)
    # X = tf.keras.layers.Dense(512, activation='relu')(X)
    X = keras.layers.GlobalAveragePooling2D()(X)
    X = tf.keras.layers.Dense(10, activation='softmax')(X)
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name="OwnResNet")

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


def Build_Train_Dense(train_data, train_labels, test_data, test_labels, epochs, amount_features):
    # CREATE MODEL CNN ARCHITECTURE

    # Model for coeffs without segmentation
    model = keras.Sequential()
    model.add(keras.layers.Dense(amount_features, activation='sigmoid', input_shape=(amount_features,)))
    model.add(keras.layers.Dense(110, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(60, activation='relu'))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(30, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    '''
    model = keras.Sequential()
    model.add(keras.layers.Dense(amount_features, activation='sigmoid', input_shape=(amount_features,)))
    model.add(keras.layers.Dense(800, activation='relu'))  # 150    0,511
    model.add(keras.layers.Dense(400, activation='relu'))  # nicht vorhanden
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(200, activation='relu'))   # 80
    model.add(keras.layers.Dense(100, activation='relu'))  # nicht vorhanden
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(30, activation='relu'))   # 30
    model.add(keras.layers.Dense(10, activation='softmax'))
    '''

    # TRAIN MODEL
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # austauschen z.b hinge loss
                  metrics=['accuracy'])

    startTime = datetime.datetime.now()

    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=64,
                        validation_data=(test_data, test_labels))  # ,callbacks=[WandbCallback()]

    trainingTime = datetime.datetime.now() - startTime
    print("Time until training finished: " + str(trainingTime))

    return model, history


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


def res_ident_block(X, filter_amount):
    # save input tensor for later
    X_skip = X

    # Layer 1
    X = tf.keras.layers.Conv2D(filter_amount, (3, 3), padding='same')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Layer 2
    X = tf.keras.layers.Conv2D(filter_amount, (3, 3), padding='same')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)

    # Add Residue and use relu afterwards
    X = tf.keras.layers.Add()([X, X_skip])
    X = tf.keras.layers.Activation('relu')(X)
    return X


def res_conv_block(X, filter_amount):
    # save input tensor for later
    X_skip = X

    # Layer 1
    X = tf.keras.layers.Conv2D(filter_amount, (3, 3), padding='same', strides=(2, 2))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Layer 2
    X = tf.keras.layers.Conv2D(filter_amount, (3, 3), padding='same')(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)

    # Processing Residue with conv(1,1)
    X_skip = tf.keras.layers.Conv2D(filter_amount, (1, 1), strides=(2, 2))(X_skip)

    # Add Residue and use relu afterwards
    X = tf.keras.layers.Add()([X, X_skip])
    X = tf.keras.layers.Activation('relu')(X)
    return X


def res_ident_block_bottleneck(X, k, filters):
    F1, F2, F3 = filters  # beispiel = [64, 64, 256]

    X_skip = X

    X = keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Conv2D(filters=F2, kernel_size=(k, k), strides=(1, 1), padding='same')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)

    X = keras.layers.Add()([X, X_skip])  # SKIP Connection
    X = keras.layers.Activation('relu')(X)

    return X


def res_conv_block_bottleneck(X, k, filters, s=2):
    F1, F2, F3 = filters

    X_shortcut = X

    X = keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Conv2D(filters=F2, kernel_size=(k, k), strides=(1, 1), padding='same')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
    X = keras.layers.BatchNormalization(axis=3)(X)

    X_shortcut = keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid')(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(axis=3)(X_shortcut)

    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X
