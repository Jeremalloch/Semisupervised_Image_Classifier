import h5py
import random
import numpy as np
from image_preprocessing import image_transform
import itertools
import warnings

class DataGenerator(self):
    """
    Class for a generator that reads in data from the HDF5 file, one batch at
    a time, converts it into the jigsaw, and then returns the data
    """
    def __init__(self, xDim=64, yDim=64, numChannels=3, batchSize=32, meanTensor=None, stdTensor=None, maxHammingSet=None):
        """
        meanTensor - rank 3 tensor of the mean for each pixel for all colour channels, used to normalize the data
        stdTensor - rank 3 tensor of the std for each pixel for all colour channels, used to normalize the data
        maxHammingSet - a 
        """
        self.xDim = xDim
        self.yDim = yDim
        self.numChannels = numChannels
        self.batchSize = batchSize
        self.meanTensor = meanTensor
        self.stdTensor = stdTensor
        if maxHammingSet == None:
            warnings.warn("Did not pass a set of jigsaw orientations", UserWarning)
            temp = list(itertools.permutations(range(9),9))
            self.maxHammingSet = np.array(temp[:100], dtype=np.uint8)
        else:
            self.maxHammingSet = np.array(maxHammingSet, dtype=np.uint8)
        # Determine how many possible jigsaw puzzle arrangements there are
        self.numJigsawTypes = self.maxHammingSet.shape[0]
        # Use default options for JigsawCreator
        self.jigsawCreator = image_transform.JigsawCreator()

    def __data_generation_normalize(self, dataset, batchIndex):
        """
        Internal method used to help generate data, used when 
        dataset - an HDF5 dataset (either train or validation)
        """
        # Determine which jigsaw permutation to use
        jigsawPermutationIndex = random.randint(self.numJigsawTypes)
        #  X = []
        #  X = np.empty((self.batchSize, self.xDim, self.yDim, self.numChannels), dtype=np.uint8)
        #  Y = np.zeros((self.batchSize, self.numJigsawTypes), dtype=np.uint8)

        X = np.empty((self.batchSize, self.xDim, self.yDim, self.numChannels), dtype=np.uint8)
        X = dataset[batchIndex*self.batchSize:(batchIndex+1)*self.batchSize,...]
        for i in range(self.batchSize):
            # subtract mean first and divide by std from training set to 
            # normalize the image then jitter, etc
            X -= self.meanTensor
            X /= self.stdTensor
            self.jigsawCreator.create_croppings(

        return X, Y

    #  def sparsify(self, y):
    #      """
    #      Returns labels in binary NumPy array
    #      """
    #      return np.array([[1 if y[i] == j else 0 for j in range(self.numJigsawTypes)]
    #                       for i in range(y.shape[0])])


    def generate(self, dataset):
        """
        dataset - an HDF5 dataset (either train or validation)
        """
        numBatches = dataset.shape[0]//self.batchSize
        batchIndex = 0
        while True:
             # Load data
            X, Y = __data_generation_normalize(dataset, batchIndex)
            batchIndex += 1 # Increment the batch index
            if batchIndex == numBatches:
                # so that batchIndex wraps back and loop goes on indefinitely
                batchIndex = 0 
            yield X, Y



