from PIL import Image
import numpy as np
import sys
import random
import time
import itertools
import warnings


class JigsawCreator:
    """
    Creates an image processor that converts an image passed as a numpy array
    into 9 subimages, applies processing to them to improve the generalization
    of the learned weights (moving the colour channels independantly in order
    to prevent the network just learning to use chromatic aberation).
    The nine sub-images are then passed passe
    """

    def __init__(self, maxHammingSet, cropSize=225, cellSize=75, tileSize=64,
                 tileLocationRange=10, colourJitter=2):
        """
        cropSize - the size of the square crop used
        cellSize - the dimensions of each subcell of the crop. Dimensions are
        cropSize/3
        tileSize - size of the image cropped from within each cell
        maxHammingSet - 2D array, row is each permutation, column is the elements
        """
        self.cropSize = cropSize
        self.cellSize = cellSize
        self.tileSize = tileSize
        self.colourJitter = colourJitter
        self.tileLocationRange = tileLocationRange
        #  if not maxHammingSet.any():
        #      warnings.warn("Did not pass a set of jigsaw orientations", UserWarning)
        #      temp = list(itertools.permutations(range(9),9))
        #      self.maxHammingSet = np.array(temp[:100], dtype=np.uint8)
        #  else:
        self.maxHammingSet = np.array(maxHammingSet, dtype=np.uint8)
        self.numPermutations = self.maxHammingSet.shape[0]

    def colour_channel_jitter(self, numpy_image):
        """
        Takes in a 3D numpy array and then jitters the colour channels by
        between -2 and 2 pixels (to deal with overfitting to chromatic
        aberations).
        Input - a WxHx3 numpy array
        Output - a (W-4)x(H-4)x3 numpy array (3 colour channels for RGB)
        """
        # Determine the dimensions of the array, minus the crop around the border
        # of 4 pixels (threshold margin due to 2 pixel jitter)
        x_dim = numpy_image.shape[0] - self.colourJitter * 2
        y_dim = numpy_image.shape[1] - self.colourJitter * 2
        # Determine the jitters in all directions
        R_xjit = random.randrange(self.colourJitter * 2 + 1)
        R_yjit = random.randrange(self.colourJitter * 2 + 1)
        G_xjit = random.randrange(self.colourJitter * 2 + 1)
        G_yjit = random.randrange(self.colourJitter * 2 + 1)
        B_xjit = random.randrange(self.colourJitter * 2 + 1)
        B_yjit = random.randrange(self.colourJitter * 2 + 1)
        # Seperate the colour channels
        return_array = np.empty((x_dim, y_dim, 3), np.float32)
        for colour_channel in range(3):
            return_array[:,:,colour_channel] = numpy_image[R_xjit:x_dim +
                                            R_xjit, R_yjit:y_dim + R_yjit, colour_channel]
        #  green_channel_array = numpy_image[G_xjit:x_dim +
        #                                    G_xjit, G_yjit:y_dim + G_yjit, 1]
        #  blue_channel_array = numpy_image[B_xjit:x_dim +
        #                                   B_xjit, B_yjit:y_dim + B_yjit, 2]
        # Put the arrays back together and return it
        #  np.stack((red_channel_array, green_channel_array,
        #                   blue_channel_array), axis=-1)
        return return_array

    #  @jit(u1[:](u1[:],u1[:]))
    def create_croppings(self, numpy_array):
        """
        Take in a 3D numpy array and a 4D numpy array containing 9 "jigsaw" puzzles.
        Dimensions of array is 64 (height) x 64 (width) x 3 (colour channels) x 9
        (each cropping)

        The 3x3 grid is numbered as follows:
        0    1    2
        3    4    5
        6    7    8
        """
        # Jitter the colour channel
        numpy_array = self.colour_channel_jitter(numpy_array)

        y_dim, x_dim = numpy_array.shape[:2]
        # Have the x & y coordinate of the crop
        crop_x = random.randrange(x_dim - self.cropSize)
        crop_y = random.randrange(y_dim - self.cropSize)
        # Select which image ordering we'll use from the maximum hamming set
        perm_index = random.randrange(self.numPermutations)
        final_crops = np.zeros(
            (self.tileSize, self.tileSize, 3, 9), dtype=np.float32)
        for row in range(3):
            for col in range(3):
                x_start = crop_x + col * self.cellSize + \
                    random.randrange(self.cellSize - self.tileSize)
                y_start = crop_y + row * self.cellSize + \
                    random.randrange(self.cellSize - self.tileSize)
                # Put the crop in the list of pieces randomly according to the
                # number picked
                final_crops[:, :, :, self.maxHammingSet[perm_index, row * 3 + col]
                            ] = numpy_array[y_start:y_start + self.tileSize, x_start:x_start + self.tileSize, :]
        return final_crops, perm_index


def benchmark(loop_count=250):
    final_crops = np.zeros(
        (self.tileSize, self.tileSize, 3, 9), dtype=np.float32)

    file_ = "resized/"
    file_paths = [file_ + "HOCO.jpg", file_ + "car_1.jpg",
                  file_ + "car_2.jpg", file_ + "car_3.jpg"]

    images = np.array([Image.open(path)
                       for path in file_paths], dtype=np.uint8)

    # Run benchmark for how long it takes to convert to create jigsaw from an
    # image
    start_time = time.time()
    for _ in range(loop_count):
        for index in range(4):
            cropped_array = create_croppings(
                colour_channel_jitter(images[index]))
    end_time = time.time()
    del cropped_array
    print("Time elapsed {}".format(end_time - start_time))
    print("Time per image {}".format((end_time - start_time) / (loop_count * 4)))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        benchmark(sys.argv[1])
    else:
        print("No loop count supplied to benchmark function, using default of 250")
        benchmark()
