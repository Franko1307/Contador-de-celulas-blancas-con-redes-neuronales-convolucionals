import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras import optimizers
import cv2
import scipy
import os

def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []

    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            if wbc_type in ['NEUTROFILO']:
                label = 'NEUTROFILO'

            elif wbc_type in ['EOSINOFILO']:
                label = 'EOSINOFILO'
            elif wbc_type in ['LINFOCITO']:
                label = 'LINFOCITO'
            else:
                label = 'MONOCITO'
            for image_filename in os.listdir(folder + wbc_type):
                img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
                if img_file is not None:
                    # Downsample the image to 120, 160, 3
                    img_file = scipy.misc.imresize(arr=img_file, size=(120, 160, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)

    return X,y

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x * 1./255., input_shape=(120, 160, 3), output_shape=(120, 160, 3)))
    model.add(Conv2D(32, (3, 3), input_shape=(120, 160, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    optim = optimizers.Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=1e-6,
        amsgrad=False
    )

    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    return model


if __name__ == "__main__":
    num_classes = 4
    epochs = 20
    BASE_DIR = './'
    batch_size = 32


    model = get_model()
    print(model.summary())

    X_train, y_train = get_data(BASE_DIR + 'base-de-datos/entrenamiento/')
    X_test, y_test = get_data(BASE_DIR + 'base-de-datos/validacion-facil/')

    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)


    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )


    train_generator = datagen.flow(
        X_train,
        y_train,
        batch_size=batch_size
    )

    validation_generator = datagen.flow(
        X_test,
        y_test,
        batch_size=batch_size
    )

    model = get_model()

    filepath = "modelo-version-2"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(X_train),
        validation_data=validation_generator,
        validation_steps=len(X_test),
        epochs=epochs,
        callbacks= callbacks_list
    )

    model.save('modelo-2.h5')  # always save your weights after training or during training
