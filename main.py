from keras.layers import (Dense, Dropout, Concatenate, Input, Activation, Flatten, Conv2D,
                          MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, add)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras import optimizers
from time import strftime, localtime
import warnings
import os
import pickle
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
    # doubles model paramter count
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(hammingSetSize, activation='softmax')(x)
    model = Model(inputs=modelInputs, outputs=x)

    return model


TEST = False
USE_MULTIPROCESSING = False

if USE_MULTIPROCESSING:
    n_workers = 8
    warnings.warn('Generators are not thread safe!', UserWarning)
else:
    n_workers = 1

# Determine if the full, ~125k image dataset, or the 200 image test
# dataset should be used
if TEST:
    hdf5_path = 'Datasets/COCO_2017_unlabeled_test_subset.hdf5'
    batch_size = 16
    num_epochs = 10
    hamming_set_size = 10
    model = contextFreeNetwork(hammingSetSize=10)
    #  model = trivialNet()
else:
    hdf5_path = 'Datasets/COCO_2017_unlabeled.hdf5'
    batch_size = 64
    num_epochs = 100
    hamming_set_size = 100
    model = contextFreeNetwork()

# Open up the datasets
hdf5_file = h5py.File(hdf5_path)
normalize_mean = np.array(hdf5_file['train_mean'])
normalize_std = np.array(hdf5_file['train_std'])
train_dataset = hdf5_file['train_img']
val_dataset = hdf5_file['val_img']
test_dataset = hdf5_file['test_img']
max_hamming_set = hdf5_file['max_hamming_set']

dataGenerator = DataGenerator(batchSize=batch_size, meanTensor=normalize_mean,
                              stdTensor=normalize_std, maxHammingSet=max_hamming_set[:hamming_set_size])

# Output all data from a training session into a dated folder
outputPath = './model_data/{}'.format(strftime('%b_%d_%H:%M:%S', localtime()))
os.makedirs(outputPath)
checkpointer = ModelCheckpoint(
    outputPath +
    '/weights_improvement.hdf5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True)
reduce_lr_plateau = ReduceLROnPlateau(
    monitor='val_loss', patience=3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
# tBoardLogger = TensorBoard(log_dir=outputPath, histogram_freq=5,
# batch_size=batch_size, write_graph=True, write_grads=True,
# write_images=True)

opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(generator=dataGenerator.generate(train_dataset),
                              epochs=num_epochs,
                              steps_per_epoch=train_dataset.shape[0] // batch_size,
                              validation_data=dataGenerator.generate(
                                  val_dataset),
                              validation_steps=val_dataset.shape[0] // batch_size,
                              use_multiprocessing=USE_MULTIPROCESSING,
                              workers=n_workers,
                              callbacks=[checkpointer, reduce_lr_plateau, early_stop])

scores = model.evaluate_generator(
    dataGenerator.generate(test_dataset),
    steps=test_dataset.shape[0] //
    batch_size,
    workers=n_workers,
    use_multiprocessing=USE_MULTIPROCESSING)

# Output the test loss and accuracy
print("Test loss: {}".format(scores[0]))
print("Test accuracy: {}".format(scores[1]))

# Save the train and val accuracy history to file
with open(outputPath + '/history.pkl', 'wb') as history_file:
    pickle.dump(history.history, history_file)
# Save the test score accuracy
with open(outputPath + '/test_accuracy.pkl', 'wb') as test_file:
    pickle.dump(scores, test_file)

model.save(outputPath + '/model.hdf5')
