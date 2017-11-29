from keras.layers import Dense
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD,Adam
#  import warnings


class Model(inputs):
    """
    Creates a Keras model for a multi-input CNN used for
    jigsaw puzzle unsupervised learning
    """
    x = Conv2D(64,(7,7), strides=(2,2), padding='same', name='input_conv')(input)
    x = BatchNormalization()(x)
    model = Model()


def residualMapping(inputTensor, filters):
    """
    Residual building block where input and output tensor are the same 
    dimensions
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(filters, (3,3))
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3,3))
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = layers.add([x, inputTensor])
    x = Activation('relu')(x)

    return x


def downsizeMapping(inputTensor, filters):
    """
    Residual building block where input tensor dimensions are halved, but 
    feature map dimensions double
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    x = Conv2D(filters, (3,3), strides=(2,2))
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3,3))
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    inputTensor = Conv2D(filters, (1,1), strides=(1,1))
    x = layers.add([x, inputTensor])
    x = Activation('relu')(x)

    return x


def ResNet34Bottom(inputTensor, inputShape):
    """
    Creates a stack of layers equivalent to ResNet-34 architecture up until the
    average pool and 1000-d fully connected layer.
    Assumes that the input is a square patch, so that the output is 1x1x512 
    regardless of input dimensions (using average pooling)
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    #  if (inputShape[0] != inputShape[1]):
    #      warnings.warn("Image input shape was non-square", Warning)

    x = Conv2D(64, (7,7), strides=(2, 2), padding='same')(inputTensor)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = downsizeMapping(x, 64)
    x = residualMapping(x, 64)
    x = residualMapping(x, 64)

    x = downsizeMapping(x, 128)
    x = residualMapping(x, 128)
    x = residualMapping(x, 128)
    x = residualMapping(x, 128)

    x = downsizeMapping(x, 256)
    x = residualMapping(x, 256)
    x = residualMapping(x, 256)
    x = residualMapping(x, 256)
    x = residualMapping(x, 256)
    x = residualMapping(x, 256)

    x = downsizeMapping(x, 512)
    x = residualMapping(x, 512)
    x = residualMapping(x, 512)

    dimReduc = 32 # We halve dimensions 5 times, hence 2^5 = 32
    x = AveragePooling2D((inputShape[0]/dimReduc, inputShape[1]/dimReduc))(x)

    return x
