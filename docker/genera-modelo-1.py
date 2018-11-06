import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import csv
import cv2
import scipy

def get_data(folder):
    X = []
    y = []

    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            for image_filename in os.listdir(folder + wbc_type):
                img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
                img_file = scipy.misc.imresize(arr=img_file, size=(120, 160, 3))
                if img_file is not None:
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(wbc_type)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

def get_filename_for_index(index):
    PREFIX = 'images/BloodImage_'
    num_zeros = 5 - len(index)
    path = '0' * num_zeros + index
    return PREFIX + path + '.jpg'

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(120, 160, 3), output_shape=(120, 160, 3)))
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
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return model

if __name__ == "__main__":
    num_classes = 5
    epochs = 20
    BASE_PATH = './'
    batch_size = 32


    reader = csv.reader(open(BASE_PATH + 'labels.csv'))
    # skip the header
    next(reader)

    X = []
    y = []

    for row in reader:
        label = row[2]
        if len(label) > 0 and label.find(',') == -1:
            filename = get_filename_for_index(row[1])
            img_file = cv2.imread(BASE_PATH + filename)
            if img_file is not None:
                img_file = scipy.misc.imresize(arr=img_file, size=(120, 160, 3))
                img_arr = np.asarray(img_file)
                # img_arr = apply_color_mask(img_arr)
                X.append(img_arr)
                y.append(label)


    X = np.asarray(X)
    y = np.asarray(y)

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)

    y = np_utils.to_categorical(encoded_y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


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

    # fits the model on batches with real-time data augmentation:
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(X_train),
        validation_data=validation_generator,
        validation_steps=len(X_test),
        epochs=epochs)
    model.save_weights('modelo_3.h5')  # always save your weights after training or during training
