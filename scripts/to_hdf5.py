from time import strftime, localtime, time
import random
import h5py
import numpy as np
from PIL import Image
import glob
import random


# Make size slightly larger than the crop size in order to add a little
# more variation
SIZE = (256, 256)
TEST_RUN = False
TEST_SUBSET_DATA = True

if TEST_RUN:
    directory = "./images/"
else:
    directory = "./Datasets/unlabeled2017/"

files = glob.glob(directory + "*.jpg")

# Shuffle the data
random.shuffle(files)
# Split data 70% train, 15% validation, 15% test
files_dict = {}

# Using Welford method for online calculation
training_mean_new = np.zeros((*SIZE, 3), dtype=np.float32)
training_mean_old = np.zeros((*SIZE, 3), dtype=np.float32)
training_variance = np.zeros((*SIZE, 3), dtype=np.float32)

# Create the HDF5 output file
if TEST_SUBSET_DATA:
    # Create small list of files for training subset of data
    files_dict["train_img"] = files[:500]
    files_dict["val_img"] = files[500:750]
    files_dict["test_img"] = files[750:1000]
    print("Length of files array: {}".format(len(files)))

    hdf5_output = h5py.File(
        directory + "COCO_2017_unlabeled_test_dataset.hdf5", mode='w')
    hdf5_output.create_dataset("train_img", (len(
        files_dict["train_img"]), *SIZE, 3), np.uint8, compression="gzip")
    hdf5_output.create_dataset(
        "val_img", (len(files_dict["val_img"]), *SIZE, 3), np.uint8, compression="gzip")
    hdf5_output.create_dataset("test_img", (len(
        files_dict["test_img"]), *SIZE, 3), np.uint8, compression="gzip")
    hdf5_output.create_dataset(
        "train_mean", (*SIZE, 3), np.float32, compression="gzip")
    hdf5_output.create_dataset(
        "train_std", (*SIZE, 3), np.float32, compression="gzip")
else:
    files_dict["train_img"] = files[:int(0.7 * len(files))]
    files_dict["val_img"] = files[int(0.7 * len(files)):int(0.85 * len(files))]
    files_dict["test_img"] = files[int(0.85 * len(files)):]
    print("Length of files array: {}".format(len(files)))

    hdf5_output = h5py.File(directory + "COCO_2017_unlabeled.hdf5", mode='w')
    hdf5_output.create_dataset(
        "train_img", (len(files_dict["train_img"]), *SIZE, 3), np.uint8)
    hdf5_output.create_dataset(
        "val_img", (len(files_dict["val_img"]), *SIZE, 3), np.uint8)
    hdf5_output.create_dataset(
        "test_img", (len(files_dict["test_img"]), *SIZE, 3), np.uint8)
    hdf5_output.create_dataset("train_mean", (*SIZE, 3), np.float32)
    hdf5_output.create_dataset("train_std", (*SIZE, 3), np.float32)

start_time = time()
small_start = start_time
for img_type, img_list in files_dict.items():
    for index, fileName in enumerate(img_list):
        im = Image.open(fileName)
        # Discard black and white images, breaks rest of pipeline
        if (im.mode == 'RGB'):
            # If its taller than it is wide, crop first
            if (im.size[1] > im.size[0]):
                crop_shift = random.randrange(im.size[1] - im.size[0])
                im = im.crop(
                    (0, crop_shift, im.size[0], im.size[0] + crop_shift))
            elif (im.size[0] > im.size[1]):
                crop_shift = random.randrange(im.size[0] - im.size[1])
                im = im.crop(
                    (crop_shift, 0, im.size[1] + crop_shift, im.size[1]))
            im = im.resize(SIZE, resample=Image.LANCZOS)
            numpy_image = np.array(im, dtype=np.uint8)
            # Calculate per feature mean and variance on training data only
            if img_type == "train_img":
                # Using Welford method for online calculation of mean and
                # variance
                if index > 0:
                    training_mean_new = training_mean_old + \
                        (numpy_image - training_mean_old) / (index + 1)
                    training_variance = training_variance + \
                        (numpy_image - training_mean_old) * \
                        (numpy_image - training_mean_new)
                    training_mean_old = training_mean_new
                else:
                    training_mean_new = numpy_image
                    training_mean_old = numpy_image
            # Save the image to the HDF5 output file
            hdf5_output[img_type][index, ...] = numpy_image
            if index % 1000 == 0 and index > 0:
                small_end = time()
                print("Saved {} {}s to hdf5 file in {} seconds".format(
                    index, img_type, small_end - small_start))
                small_start = time()

# Calculate the std using the variance array
training_std = np.zeros((*SIZE, 3), dtype=np.float32)
np.sqrt(training_variance / (len(files_dict["train_img"]) - 1), training_std)
hdf5_output["train_mean"][...] = training_mean_new
hdf5_output["train_std"][...] = training_std

end_time = time()
print("Elapsed time: {} seconds".format(end_time - start_time))
