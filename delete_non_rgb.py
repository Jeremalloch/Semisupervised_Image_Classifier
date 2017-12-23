from PIL import Image
from time import time
import glob
import os


directory = "./Datasets/unlabeled2017/"
files = glob.glob(directory + "*.jpg")

num_removed = 0
a = time()
for fileName in files:
    im = Image.open(fileName)
    if (im.mode != "RGB"):
        os.remove(fileName)
        num_removed += 1

b = time()

print("{} files removed in {} seconds".format(num_removed, b - a))
