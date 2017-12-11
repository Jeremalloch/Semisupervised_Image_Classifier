from PIL import Image
import numpy as np
import sys, random
import time
from numba import vectorize, njit, prange, jit, u1

# TODO: Consider adding numba to speed up some functions if need be


class JigsawCreator(self):
    """
    Creates an image processor that converts an image passed as a numpy array
    into 9 subimages, applies processing to them to improve the generalization
    of the learned weights (moving the colour channels independantly in order
    to prevent the network just learning to use chromatic aberation).
    The nine sub-images are then passed passe
    """
    def __init__(self, cropSize = 225, cellSize = 75, tileSize = 64,
            tileLocationRange = 10):
        """
        cropSize - the size of the square crop used
        cellSize - the dimensions of each subcell of the crop. Dimensions are
        cropSize/3
        tileSize - size of the image cropped from within each cell
        """
        self.cropSize = cropSize
        self.cellSize = cellSize
        self.tileSize = tileSize
        self.tileLocationRange = tileLocationRange


    # TODO - consider making the jitter modular, adjust how much each image is jittered
    def colour_channel_jitter(self, numpy_image):
        """
        Takes in the entire image, converts it into a 3D numpy array, and then 
        jitters the colour channels by between -2 and 2 pixels (to deal with 
        overfitting to chromatic aberations).
        Input - a PIL image object with HxW dimensions
        Output - a (H-4)x(W-x)x3 numpy array (3 colour channels for RGB)
        """
        # Determine the dimensions of the array, minus the crop around the border
        # of 4 pixels (threshold margin due to 2 pixel jitter)
        x_dim = numpy_image.shape[0] - 4
        y_dim = numpy_image.shape[1] - 4
        # Determine the jitters in all directions
        R_xjit = random.randrange(5) 
        R_yjit = random.randrange(5) 
        G_xjit = random.randrange(5)
        G_yjit = random.randrange(5)
        B_xjit = random.randrange(5)
        B_yjit = random.randrange(5)
        # Sep
        red_channel_array = numpy_image[R_xjit:x_dim + R_xjit,R_yjit:y_dim + R_yjit,0]
        green_channel_array = numpy_image[G_xjit:x_dim + G_xjit,G_yjit:y_dim + G_yjit,1]
        blue_channel_array = numpy_image[B_xjit:x_dim + B_xjit,B_yjit:y_dim + B_yjit,2]
        # Put the arrays back together and return it
        return np.stack((red_channel_array,green_channel_array,blue_channel_array), axis=-1)


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
        y_dim, x_dim = numpy_array.shape[:2]
        # Have the x & y coordinate of the crop
        crop_x = random.randrange(x_dim - self.cropSize)
        crop_y = random.randrange(y_dim - self.cropSize)
        #  box = (crop_x, crop_y, crop_x + self.cropSize, crop_y + self.cropSize)
        #  cropped_region = im.crop(box)
        final_crops = np.zeros((self.tileSize, self.tileSize, 3, 9), dtype='uint8')
        for row in range(3):
            for col in range(3):
                x_start = crop_x + col*self.cellSize + random.randrange(self.cellSize - self.tileSize)
                y_start = crop_y + row*self.cellSize + random.randrange(self.cellSize - self.tileSize)
                final_crops[:,:,:,row*3 + col] = numpy_array[y_start:y_start + self.tileSize, x_start:x_start + self.tileSize,:]
        return final_crops


    def permutate_images(images, permutation):
        """
        Receives a list of images and a list of integers describing the permutation
        to put the images into
        """
        # Sorting in place a WIP, just making new array now
        #  for index, perm_index in range(9):
        #      if index != perm_index:
        #          temp = images[index]
        #          temp_index = permutation.index(index)
        #          images[index] = images[perm_index]
        #          images[temp_index] = temp
        #          # Set the permutation index equal
        #          permutation[perm_index] = index
        permutate_images = []
        for index in permutation:
            permutate_images.append(images[index])
        return permutated_images


def main():
    start_time = time.time()
    final_crops = np.zeros((self.tileSize, self.tileSize, 3, 9), dtype='uint8')
    loop_count = 250
    file_ = "resized/"
    for _ in range(loop_count):
        im = Image.open(file_ + "HOCO.jpg")
        cropped_array = create_croppings(colour_channel_jitter(im))
        im = Image.open(file_ + "car_1.jpg")
        cropped_array = create_croppings(colour_channel_jitter(im))
        im = Image.open(file_ + "car_2.jpg")
        cropped_array = create_croppings(colour_channel_jitter(im))
        im = Image.open(file_ + "car_3.jpg")
        cropped_array = create_croppings(colour_channel_jitter(im))
    end_time = time.time()
    del cropped_array
    print("Time elapsed {}".format(end_time-start_time))
    print("Time per image {}".format((end_time-start_time)/(loop_count*4)))
    #  for i in range(9):
    #      im2 = Image.fromarray(cropped_array[:,:,:,i])
    #      im2.show()


main()
#  Can remove this line now that testing is over
#  for infile in sys.argv[1:]:
#      im = Image.open(infile)
        

