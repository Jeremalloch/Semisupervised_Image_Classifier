import time
from time import strftime, localtime
import glob
import numpy as np
from PIL import Image

# Make size slightly larger than the crop size in order to add a little
# more variation
SIZE = (256,256)
TO_FILE = True

#  files = glob.glob("./images/*.jpg")
files = glob.glob("./Datasets/unlabeled2017/*.jpg")
print("Length of files array: {}".format(len(files)))

mean_sum = np.zeros(3, dtype=np.float64)
std_sum = np.zeros(3, dtype=np.float64)
total_non_RGB = 0
start_time = time.time()
for fileName in files:
    im = Image.open(fileName)
    # Discard black and white images, breaks rest of pipeline
    if (im.mode == 'RGB'):
        # TODO: Want to resize such that smallest axis is still greater than 225
        #  im.thumbnail(SIZE)
        numpy_image = np.array(im)
        for i in range(3):
            mean_sum[i] += np.mean(numpy_image[:,:,i])
            std_sum[i] += np.std(numpy_image[:,:,i])
    else:
        total_non_RGB += 1

end_time = time.time()
mean_sum /= len(files)
std_sum /= len(files)

if TO_FILE:
    output_name = "image_stats/image_means_{}.txt".format(strftime("%b %d %H:%M:%S", localtime()))
    with open(output_name, 'w') as f:
        for i in range(3):
            f.write("For colour channel {}\n".format(i))
            f.write("Mean: {} STD: {}\n".format(mean_sum[i],std_sum[i]))
        f.write("There were {} non RGB images out of a total dataset size of {}\n".format(total_non_RGB,len(files)))
        f.write("Elapsed time: {}\n".format(end_time - start_time))
        f.close()
else:
    for i in range(3):
        print("For colour channel {}".format(i))
        print("Mean: {} STD: {}".format(mean_sum[i],std_sum[i]))
    print("Elapsed time: {} seconds".format(end_time - start_time))
