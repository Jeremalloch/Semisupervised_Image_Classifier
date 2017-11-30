from keras.layers import Dense, Concatenate, Dropout
from keras.models import Model
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
    sharedLayers = [ResNet34Bottom(inputTensor,inputShape) for inputTensor in modelInputs]
    x = Concatenate(sharedLayers, axis=-1) # Reconsider what axis to merge
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100,activation='softmax')(x)
    model = Model(inputs=modelInputs, outputs = x)

    return model

model.train_on_batch()

checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])
