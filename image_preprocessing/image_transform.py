from PIL import Image
import os, sys, random


CROP_SIZE = 225
CELL_SIZE = 75
PRE_JITTER_TILE_SIZE = 68
TILE_SIZE = 64
TILE_LOCATION_RANGE = 10


for infile in sys.argv[1:]:
    im = Image.open(infile)
        

def color_channel_jitter(image):
    """
    Jitters the color channels between -2 and 2 pixels (selected from uniform 
    distribution).
    Input image is a 68x68 image (leaving room for up to 2 pixel jitter).
    Returned image is cropped down to tile size of 64x64 pixels used to
    train the network.
    """
    #  np.fromstring(image.tobytes(), dtype=np.uint8)
    source = image.split()
    RGB = [0,1,2]
    jitter_image = []
    for channel in RGB:
        # Choose a random jitter factor between -2 and 2
        jitter = random.randrange(5) - 2
        jitter_image.append(source[channel].point(lambda i: i + jitter))
    return_image = Image.merge("RGB",jitter_image)
    # Crop out the 2 pixel border we preserved for this operation
    return return_image.crop((2,2,66,66))


def create_croppings(image):
    """
    Take in a PIL.Image object and return 9
    "jigsaw" puzzle subimages that are 64x64
    pixels

    The 3x3 grid is numbered as follows:
    0    1    2
    3    4    5
    6    7    8
    """
    x_dim, y_dim = image.size
    # Have the x & y coordinate of the crop
    crop_x = random.randrange(x_dim - CROP_SIZE)
    crop_y = random.randrange(y_dim - CROP_SIZE)
    #  box = (crop_x, crop_y, crop_x + CROP_SIZE, crop_y + CROP_SIZE)
    #  cropped_region = im.crop(box)
    final_crops = []
    for row in range(3):
        for col in range(3):
            x_start = crop_x + col*CELL_SIZE + random.randrange(CELL_SIZE - PRE_JITTER_TILE_SIZE )
            y_start = crop_y + row*CELL_SIZE + random.randrange(CELL_SIZE - PRE_JITTER_TILE_SIZE)
            box = (x_start, y_start, x_start + PRE_JITTER_TILE_SIZE, y_start + PRE_JITTER_TILE_SIZE)
            final_crops.append(color_channel_jitter(image.crop(box)))
    return final_crops


def main(file_names):
    for infile in file_names:
        croppings = create_croppings(Image.open(infile))
        for i in range(9):
            croppings[i].save("cropped_{}.jpg".format(i))
    


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


if __name__ == "main":
    main(sys.argv[1:])
