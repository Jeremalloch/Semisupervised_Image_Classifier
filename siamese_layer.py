from keras.layers import Dense, Concatenate, Dropout, Input, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import (Dense, Input, Activation, Flatten, Conv2D,
                          MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, add)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import keras.backend as K
from time import strftime, localtime
import resnetBottom
from DataGenerator import DataGenerator
import numpy as np
import h5py
#  from keras.utils import plot_model


def basicModel(tileSize=64, numPuzzles=9):
    """
    Returns trivial conv net
    """
    inputShape = (tileSize, tileSize, 3)
    inputTensor = Input(inputShape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputTensor)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)

    model = Model(inputTensor, x, name='Trivial_Model')
    return model


def trivialNet(tileSize=64, numPuzzles=9, hammingSetSize=10):
    """
    Implemented non-siamese
    tileSize - The dimensions of the jigsaw input
    numPuzzles - the number of jigsaw puzzles

    returns a keras model
    """
    inputShape = (tileSize, tileSize, 3)
    modelInputs = [Input(inputShape) for _ in range(numPuzzles)]
    sharedLayer = basicModel()
    sharedLayers = [sharedLayer(inputTensor) for inputTensor in modelInputs]

    def L1_distance(x): return K.concatenate(
        [[K.abs(x[i] - x[j]) for j in range(i, 9)] for i in range(9)])
    both = K.concatenate([[K.abs(x[0] - x[j]) for j in range(9)])
    #  both = K.concatenate([[K.abs(x[i] - x[j]) for j in range(i, 9)] for i in range(9)])
                 #  output_shape=lambda x: x[0])

    x = Concatenate()(sharedLayers)  # Reconsider what axis to merge
    x = Dense(512, activation='relu')(x)
    x = Dense(hammingSetSize, activation='softmax')(x)
    model = Model(inputs=modelInputs, outputs=x)

    return model


def contextFreeNetwork(tileSize=64, numPuzzles=9, hammingSetSize=100):
    """
    Implemented non-siamese
    tileSize - The dimensions of the jigsaw input
    numPuzzles - the number of jigsaw puzzles

    returns a keras model
    """
    inputShape = (tileSize, tileSize, 3)
    modelInputs = [Input(inputShape) for _ in range(numPuzzles)]
    sharedLayer = resnetBottom.ResNet34Bottom(inputShape)
    sharedLayers = [sharedLayer(inputTensor) for inputTensor in modelInputs]
    x = Concatenate()(sharedLayers)  # Reconsider what axis to merge
    #  x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    # permutations
    x = Dense(hammingSetSize, activation='softmax')(x)
    model = Model(inputs=modelInputs, outputs=x)

    return model


model = trivialNet()
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
