from keras.layers import Dense, Concatenate, Dropout, Input, Conv2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import resnetBottom

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
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='softmax')(x)
    model = Model(inputs=modelInputs, outputs=x)

    return model


#  inputShape = (224,224,3)
#  inputTensor = Input(inputShape)
#  model = resnetBottom.ResNet34Bottom(inputTensor,inputShape)
model = contextFreeNetwork()
model.summary()

#  checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
#
#  training_generator =
#
#  n_workers = 2
#
#  model.fit_generator(generator = training_generator,
#                      steps_per_epoch = len(partition['train'])//batch_size, # TODO fix
#                      validation_data = validation_generator,
#                      validation_steps = len(partition['validation'])//batch_size, #TODO: fix
#                      workers = n_workers, # How many threads to use on the data generators
#                      callbacks = [checkpointer])
