from keras.layers import Dense, Concatenate, Dropout, Input, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import (Dense, Input, Activation, Flatten, Conv2D, 
        MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, add)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
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

    x = Conv2D(64, (7,7), strides=(2, 2), padding='same')(inputTensor)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = Conv2D(64, (3,3), strides=(2,2), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3,3), strides=(2,2), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3,3), strides=(2,2), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3,3), strides=(2,2), padding='same')(x)
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
    x = Concatenate()(sharedLayers) # Reconsider what axis to merge
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
    # TODO: Determine if euclidian distance 9x9 grid should be used
    x = Concatenate()(sharedLayers) # Reconsider what axis to merge
    # TODO: Determine how this first 2048 layer affects performance, since it doubles model paramter count
    #  x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    # TODO: Make sure that the number of outputs is equal to the number of permutations
    x = Dense(hammingSetSize, activation='softmax')(x)
    model = Model(inputs=modelInputs, outputs=x)

    return model


TEST = True
USE_MULTIPROCESSING = False
# Determine if the full, ~125k image dataset, or the 200 image test dataset should be used
if TEST:
    hdf5_path = "Datasets/COCO_2017_unlabeled_test_subset.hdf5"
    batch_size = 16
    num_epochs = 1000
    hamming_set_size = 10
    model = trivialNet()
else:
    hdf5_path = "Datasets/COCO_2017_unlabeled.hdf5"
    batch_size = 64
    num_epochs = 100
    hamming_set_size = 100
    model = contextFreeNetwork()

# Open up the datasets
hdf5_file = h5py.File(hdf5_path)
normalize_mean = np.array(hdf5_file["train_mean"])
normalize_std = np.array(hdf5_file["train_std"])
train_dataset = hdf5_file["train_img"]
val_dataset = hdf5_file["val_img"]
test_dataset = hdf5_file["test_img"]
max_hamming_set = hdf5_file["max_hamming_set"]

# TODO: Better name required
# TODO: Add max hamming set to generator
thisGen = DataGenerator(batchSize=batch_size, meanTensor=normalize_mean, stdTensor=normalize_std, maxHammingSet=max_hamming_set[:hamming_set_size])

#  TODO: Add csv logger to save validation and more data at each iteration
#  weights_filepath="./weights/weights_{}_improvement-{epoch:02d}-{val_acc:.2f}.hdf5".format(strftime("%b %d %H:%M:%S", localtime()))
#  checkpointer = ModelCheckpoint(weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#  # Log data to view on tensorboard
#  tBoardLogger = TensorBoard(log_dir='./logs', histogram_freq=5, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True)

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


if USE_MULTIPROCESSING:
    # TODO: optimize how many workers
    n_workers = 8
else:
    n_workers = 1

model.fit_generator(generator = thisGen.generate(train_dataset),
                    epochs = num_epochs,
                    steps_per_epoch = train_dataset.shape[0]//batch_size,
                    validation_data = thisGen.generate(val_dataset),
                    validation_steps = val_dataset.shape[0]//batch_size,
                    max_queue_size = 5,
                    use_multiprocessing = USE_MULTIPROCESSING,
                    workers = n_workers)
                    #  callbacks = [checkpointer, tBoardLogger])

# TODO: Put generator in here
#  model.evaluate_generator(generator, steps=test_dataset.shape[0]//batch_size, max_queue_size=5, workers=n_workers, use_multiprocessing=USE_MULTIPROCESSING)

# TODO: Install pydot and graphviz to visualize model
#  plot_model(model, to_file='model.png')
model.save("./saved_model.hdf5")
