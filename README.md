# Semisupervised Image Classifier Pretraining
Can low level features be learnt (or at least pre-trained) from an unsupervised image dataset?
This project explores using spatial context to learn feature representations from unlabeled images.
It works by dividing an image into 9 sub-croppings, as seen in the image below.
These images are then shuffled into one of 100 different permutations, selected from a set computed to maximize the hamming distance each element.
A CNN based on the Resnet-34 architecture is then tasked with reassembling the pieces into the correct order.

This project was inspired by the paper [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246).
This modifies the original work by using a Resnet-34 like CNN in lieu of the Alexnet architecture used in the original implementation.
Additionally, this project was trained on the unlabeled portion of the COCO 2017 dataset, consisting of only ~123k images, while the original paper trained their network on Imagenet, with the labels removed.
![alt text](https://github.com/Jeremalloch/Semisupervised_Image_Classifier/blob/master/writeup_images/tiger_patches.png "Large image")
##### An image with nine sub-images selected
![alt text](https://github.com/Jeremalloch/Semisupervised_Image_Classifier/blob/master/writeup_images/tiger_puzzle.png "Large image")
##### The nine sub-images in a shuffled order before they are fed into the neural network
![alt text](https://github.com/Jeremalloch/Semisupervised_Image_Classifier/blob/master/writeup_images/tiger_solved.png "Large image")
##### The nine sub-images after being reassembled into the correct order
