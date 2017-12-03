from time import strftime, localtime, time
import h5py
import numpy as np
from PIL import Image
import glob
import random


# Make size slightly larger than the crop size in order to add a little
# more variation
SIZE = (256,256)
TEST_RUN = True

# Determined from running over entire unlabed COCO 2017 image dataset overnight
MEANS = np.array([119.60528554557193, 113.80782733624036, 103.89157246448366], dtype=np.float32)
STD = np.array([62.043584831055576, 60.796926785629495, 61.61505847060693], dtype=np.float32)

if TEST_RUN:
    files = glob.glob("./images/*.jpg")
else:
    files = glob.glob("./Datasets/unlabeled2017/*.jpg")
print("Length of files array: {}".format(len(files)))

start_time = time()
#  for fileName in files:
fileName = "./image_preprocessing/HOCO.jpg"
im = Image.open(fileName)
# Discard black and white images, breaks rest of pipeline
if (im.mode == 'RGB'):
    # If its taller than it is wide, crop first
    #  print("File {} has dimensions {}".format(fileName,im.size))
    if (im.size[1] > im.size[0]):
        crop_shift = random.randrange(im.size[1] - im.size[0])
        im = im.crop((0, crop_shift, im.size[0], im.size[0] + crop_shift))
    elif (im.size[0] > im.size[1]):
        crop_shift = random.randrange(im.size[0] - im.size[1])
        im = im.crop((crop_shift, 0, im.size[1] + crop_shift, im.size[1]))
    im.resize(SIZE, resample=Image.LANCZOS)
    numpy_image = np.array(im, dtype=np.float32)
    for i in range(3):
        numpy_image[:,:,i] -= MEANS[i]
        numpy_image[:,:,i] /= STD[i]

end_time = time()

print("Elapsed time: {} seconds".format(end_time - start_time))
