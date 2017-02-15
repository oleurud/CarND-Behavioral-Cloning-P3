import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

def get_data():
    """
    Read all the training data and returns train_samples and validation_samples
    """

    folders = [
        './myData/center/',
        './myData/clockwise/',
        './myData/recovery/',
        './myData/track2/',
        './data/data/'
    ]

    samples = []

    for folder in folders:
        with open(folder + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if(line[0] != None):
                    samples.append([
                        line[0], #center image
                        line[3], #angle
                        folder
                    ])

    return train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                image_path = batch_sample[2] + 'IMG/' + batch_sample[0].split('/')[-1]
                center_image = np.asarray(Image.open(image_path))
                center_angle = float(batch_sample[1])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield X_train, y_train


def get_nvidia_model():
    """
    Based on http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    model = Sequential()
    
    row, col, ch = 160, 320, 3
    model.add(Cropping2D(cropping=((50, 30), (0, 0)), input_shape=(row, col, ch)))
    row, col, ch = 80, 320, 3
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2,2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1,1)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1164, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    return model

def getStats(history):
    fig = plt.figure()
    plt.plot(history.history['loss'], figure = fig)
    plt.plot(history.history['val_loss'], figure = fig)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'testing'])
    fig.savefig('stats.png', dpi=100)

def training():
    train_samples, validation_samples = get_data()
    train_generator = generator(train_samples)
    validation_generator = generator(validation_samples)

    model = get_nvidia_model()

    print("training...")
    history = model.fit_generator(
        train_generator, 
        samples_per_epoch=len(train_samples), 
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples), 
        nb_epoch=3
    )

    print("getting stats...")
    getStats(history)

    print("saving...")
    model.save('model.h5')

    print("done")


training()
