import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model

# Size of jigsaw piece
TILE_SIZE = 64
NUM_PUZZLES = 9


input_shape = (TILE_SIZE, TILE_SIZE, 3)
input_images = [Input(input_shape) for _ in range(NUM_PUZZLES)]

# model to use is going to be preferably ResNet-18, then maybe Inception-V1, then maybe ResNet-50 if I can't get
# either of the first ones working
#  conv_model = VGG16(include_top=False, weights=None, input_tensor=input_image, input_shape=input_shape)



# To train on data set too large to put in memory
model.train_on_batch()

checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])
