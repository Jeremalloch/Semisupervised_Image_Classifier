import h5py
import random
import time
import numpy as np
from image_preprocessing import image_transform
import itertools
import warnings
import threading

#  class threadsafe_iter:
#      """Takes an iterator/generator and makes it thread-safe by
#      serializing call to the `next` method of given iterator/generator.
#      """
#      def __init__(self, it):
#          self.it = it
#          self.lock = threading.Lock()
#
#      def __iter__(self):
#          return self
#
#      def __next__(self):
#          with self.lock:
#              return self.it()
#
#
#  def threadsafe_generator(f):
#      """A decorator that takes a generator function and makes it thread-safe.
#      """
#      def g(*a, **kw):
#          return threadsafe_iter(f(*a, **kw))
#      return g


class DataGenerator:
    """
    Class for a generator that reads in data from the HDF5 file, one batch at
    a time, converts it into the jigsaw, and then returns the data
    """

    def __init__(self, maxHammingSet, xDim=64, yDim=64, numChannels=3,
                 numCrops=9, batchSize=32, meanTensor=None, stdTensor=None):
        """
        meanTensor - rank 3 tensor of the mean for each pixel for all colour channels, used to normalize the data
        stdTensor - rank 3 tensor of the std for each pixel for all colour channels, used to normalize the data
        maxHammingSet - a
        """
        self.xDim = xDim
        self.yDim = yDim
        self.numChannels = numChannels
        self.numCrops = numCrops
        self.batchSize = batchSize
        self.meanTensor = meanTensor.astype(np.float32)
        self.stdTensor = stdTensor.astype(np.float32)
        #  if not maxHammingSet.any():
        #      warnings.warn("Did not pass a set of jigsaw orientations", UserWarning)
        #      temp = list(itertools.permutations(range(9),9))
        #      self.maxHammingSet = np.array(temp[:100], dtype=np.uint8)
        #  else:
        self.maxHammingSet = np.array(maxHammingSet, dtype=np.uint8)
        # Determine how many possible jigsaw puzzle arrangements there are
        self.numJigsawTypes = self.maxHammingSet.shape[0]
        # TODO: Consider inheriting from JigsawCreator
        # Use default options for JigsawCreator
        self.jigsawCreator = image_transform.JigsawCreator(
            maxHammingSet=maxHammingSet)

    def __data_generation_normalize(self, dataset, batchIndex):
        """
        Internal method used to help generate data, used when
        dataset - an HDF5 dataset (either train or validation)
        """
        # Determine which jigsaw permutation to use
        jigsawPermutationIndex = random.randrange(self.numJigsawTypes)
        x = np.empty((self.batchSize, 256, 256, self.numChannels),
                     dtype=np.float32)
        # TODO: consider using read_direct(array, source_sel=None, dest_sel=None) method to avoid
        # creating intermediary numpy array (HDF5 method)
        x = dataset[batchIndex *
                    self.batchSize:(batchIndex + 1) * self.batchSize, ...].astype(np.float32)
        # subtract mean first and divide by std from training set to
        # normalize the image
        x -= self.meanTensor
        x /= self.stdTensor
        # TODO: Implementation below creates custom cropping for each image in batch. Consider if
        # better to apply same transformation to all images worth performance increase. This would entail
        # getting croppings_creator to work on 4D tensors for a batch instead of looping
        # This implementation modifies each image individually
        X = np.empty((self.batchSize, self.xDim, self.yDim,
                      self.numCrops), dtype=np.float32)
        y = np.empty(self.batchSize)
        #  X_i = np.empty((self.xDim, self.yDim, 3, self.numCrops), dtype=np.float32)
        # Python list of 4D numpy tensors for each channel
        X = [np.empty((self.batchSize, self.xDim, self.yDim,
                       self.numChannels), np.float32) for _ in range(self.numCrops)]
        #  pdb.set_trace()
        for image_num in range(self.batchSize):
            # Transform the image into its nine croppings
            single_image, y[image_num] = self.jigsawCreator.create_croppings(
                x[image_num])
            for image_location in range(self.numCrops):
                X[image_location][image_num, :, :,
                                  :] = single_image[:, :, :, image_location]
        return X, y

    def sparsify(self, y):
        """
        Returns labels in binary NumPy array
        """
        return np.array([[1 if y[i] == j else 0 for j in range(self.numJigsawTypes)]
                         for i in range(y.shape[0])])

    #  @threadsafe_generator
    def generate(self, dataset):
        """
        dataset - an HDF5 dataset (either train or validation)
        """
        numBatches = dataset.shape[0] // self.batchSize
        batchIndex = 0
        while True:
             # Load data
            X, y = self.__data_generation_normalize(dataset, batchIndex)
            batchIndex += 1  # Increment the batch index
            if batchIndex == numBatches:
                # so that batchIndex wraps back and loop goes on indefinitely
                batchIndex = 0
            #  pdb.set_trace()
            yield X, self.sparsify(y)
            #  yield X, y
            #  yield X


def test():
    hdf5_path = "Datasets/COCO_2017_unlabeled_test_subset.hdf5"
    hdf5_file = h5py.File(hdf5_path)
    normalize_mean = np.array(hdf5_file["train_mean"])
    normalize_std = np.array(hdf5_file["train_std"])
    train_dataset = hdf5_file["train_img"]
    max_hamming_set = hdf5_file["max_hamming_set"]
    batch_size = 32

    test_gen = DataGenerator(maxHammingSet=max_hamming_set, batchSize=batch_size,
                             meanTensor=normalize_mean, stdTensor=normalize_std)

    start = time.time()
    num_epochs = 25
    for _ in range(num_epochs * (train_dataset.shape[0] // batch_size)):
        X, Y = next(test_gen.generate(train_dataset))
    end = time.time()

    print("Elapsed time {} for {} batches".format(end - start,
                                                  num_epochs * (train_dataset.shape[0] // batch_size)))
    print("Time per batch: {} milliseconds".format(
        1000 * (end - start) / (num_epochs * (train_dataset.shape[0] // batch_size))))


if __name__ == "__main__":
    test()
