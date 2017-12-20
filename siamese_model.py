from keras.layers import Dense, Concatenate, Dropout, Input, Conv2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import resnetBottom
from DataGenerator import DataGenerator
import numpy as np
import h5py

def contextFreeNetwork(tileSize=64, numPuzzles=9):
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
    x = Concatenate()(sharedLayers) # Reconsider what axis to merge
    # TODO: Determine how this first 2048 layer affects performance, since it doubles model paramter count
    #  x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='softmax')(x)
    model = Model(inputs=modelInputs, outputs=x)

    return model


TEST = True
# Determine if the full, ~125k image dataset, or the 200 image test dataset should be used
if TEST:
    hdf5_path = "Datasets/COCO_2017_unlabeled_test_subset.hdf5"
    batch_size = 1
else:
    hdf5_path = "Datasets/COCO_2017_unlabeled.hdf5"
    batch_size = 32

# Open up the datasets
hdf5_file = h5py.File(hdf5_path)
normalize_mean = np.array(hdf5_file["train_mean"])
normalize_std = np.array(hdf5_file["train_std"])
train_dataset = hdf5_file["train_img"]
val_dataset = hdf5_file["val_img"]
test_dataset = hdf5_file["test_img"]

# TODO: Better name required
thisGen = DataGenerator(batchSize=batch_size, meanTensor=normalize_mean, stdTensor=normalize_std)

#  TODO: Add csv logger to save validation and more data at each iteration
#  checkpointer = ModelCheckpoint(filepath='/weights/weights.hdf5', verbose=1, save_best_only=True)

model = contextFreeNetwork()
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

n_workers = 1

model.fit_generator(generator = thisGen.generate(train_dataset),
                    epochs = 150,
                    steps_per_epoch = train_dataset.shape[0]//batch_size,
                    validation_data = thisGen.generate(val_dataset),
                    validation_steps = val_dataset.shape[0]//batch_size,
                    workers = n_workers) # How many threads to use on the data generators
                    #  callbacks = [checkpointer])
