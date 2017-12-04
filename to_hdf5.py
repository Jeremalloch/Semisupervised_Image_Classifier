from time import strftime, localtime, time
import random
import h5py
import numpy as np
from PIL import Image
import glob
import random


# Make size slightly larger than the crop size in order to add a little
# more variation
SIZE = (256,256)
TEST_RUN = False

# Determined from running over entire unlabed COCO 2017 image dataset overnight
MEANS = np.array([119.60528554557193, 113.80782733624036, 103.89157246448366], dtype=np.float32)
STD = np.array([62.043584831055576, 60.796926785629495, 61.61505847060693], dtype=np.float32)

if TEST_RUN:
    directory = "./images/"
else:
    directory = "./Datasets/unlabeled2017/"

files = glob.glob(directory + "*.jpg")

# Shuffle the data
random.shuffle(files)
# Split data 70% train, 15% validation, 15% test
files_dict = {}
files_dict["train_img"] = files[:int(0.7*len(files))]
files_dict["val_img"] = files[int(0.7*len(files)):int(0.85*len(files))]
files_dict["test_img"] = files[int(0.85*len(files)):]
print("Length of files array: {}".format(len(files)))

# Create the HDF5 output file
hdf5_output = h5py.File(directory + "COCO_2017_unlabeled.hdf5", mode='w')
hdf5_output.create_dataset("train_img", (len(files_dict["train_img"]), *SIZE, 3), np.float32)
hdf5_output.create_dataset("val_img", (len(files_dict["val_img"]), *SIZE, 3), np.float32)
hdf5_output.create_dataset("test_img", (len(files_dict["test_img"]), *SIZE, 3), np.float32)


start_time = time()
for img_type, img_list in files_dict.items():
    for index, fileName in enumerate(img_list):
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
            im = im.resize(SIZE, resample=Image.LANCZOS)
            numpy_image = np.array(im, dtype=np.float32)
            for i in range(3):
                numpy_image[:,:,i] -= MEANS[i]
                numpy_image[:,:,i] /= STD[i]
            hdf5_output[img_type][index, ...] = numpy_image
            if index % 1000 == 0:
                print("Saved {} {}s to hdf5 file".format(index,img_type))

end_time = time()

print("Elapsed time: {} seconds".format(end_time - start_time))
