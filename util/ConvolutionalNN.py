import os
# if you want to use the GPU
#device = 'gpu'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=' + device + ',floatX=float32'

from timeit import default_timer
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from theano import config
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization

def experimentCNN(fileNames, labels):
    np.random.seed(39)
    try:
        dim = (100, 100)
        images = []
        for fileName in fileNames:
            imagePIL = Image.open(fileName).convert('LA')  # Open and Turn to grayscale
            res = imagePIL.resize(dim)
            images.append(np.array(res))  # we convert the images to a Numpy array and store them in a list

        # a list of many 100x100 images is made into 1 big array
        # config.floatX is from Theano configration to enforce float32 precision (needed for GPU computation)
        img_array = np.array(images, dtype=config.floatX)
        # #img_array.shape
        # Standarize
        mean = img_array.mean()
        stddev = img_array.std()
        img_array = (img_array - mean) / stddev

        #CONV NN
        n_channels = 1  # for grey-scale, 3 for RGB, but usually already present in the data

        if keras.backend.image_dim_ordering() == 'th':
            # Theano ordering (~/.keras/keras.json: "image_dim_ordering": "th")
            img_array = img_array.reshape(img_array.shape[0], n_channels, img_array.shape[1], img_array.shape[2])
            # train_img = img_array.reshape(img_array.shape[0], n_channels, img_array.shape[1], img_array.shape[2])
            # test_img = test_images.reshape(test_images.shape[0], n_channels, test_images.shape[1], test_images.shape[2])
        else:
            # Tensorflow ordering (~/.keras/keras.json: "image_dim_ordering": "tf")
            img_array = img_array.reshape(img_array.shape[0], img_array.shape[1], img_array.shape[2], n_channels)
            # train_img = img_array.reshape(img_array.shape[0], img_array.shape[1], img_array.shape[2], n_channels)
            # test_img = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], n_channels)

        inputShape = img_array.shape[1:]
        #1,100, 100
        trainingImages, testingImages, trainingLabels, testingLabels = \
            train_test_split(img_array, labels, test_size=0.3, random_state=39, stratify=labels)


        model = Sequential()
        n_filters = 16
        # this applies n_filters convolution filters of size 5x5 resp. 3x3 each in the 2 layers below
        # Layer 1
        model.add(Convolution2D(n_filters, 3, 3, border_mode='valid', input_shape=inputShape))
        # input shape: 100x100 images with 3 channels -> input_shape should be (3, 100, 100)
        model.add(BatchNormalization())
        model.add(Activation('relu'))  # ReLu activation
        model.add(MaxPooling2D(pool_size=(2, 2)))  # reducing image resolution by half
        model.add(Dropout(0.3))  # random "deletion" of %-portion of units in each batch
        # Layer 2
        model.add(Convolution2D(n_filters, 3, 3))  # input_shape is only needed in 1st layer
        # model.add(BatchNormalization())
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())  # Note: Keras does automatic shape inference.
        # Full Layer
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        # Compiling the model
        loss = 'binary_crossentropy'
        optimizer = 'sgd'
        # optimizer = SGD(lr=0.001)  # possibility to adapt the learn rate
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        epochs = 5

        # TRAINING the model
        history = model.fit(trainingImages, trainingLabels, batch_size=32, nb_epoch=epochs)
        predictions = model.predict_classes(testingImages)
        print(f"Metrics: ")
        print(f"Overall precision: {metrics.precision_score(testingLabels, prediction, average='micro')}")
        print(f"Overall recall: {metrics.recall_score(testingLabels, prediction, average='micro')}")
        print(f"Overall F1 score: {metrics.f1_score(testingLabels, prediction, average='micro')}")

    except Exception as ex:
        print(f"An exception ocurred {ex}")
        exit(-1)
