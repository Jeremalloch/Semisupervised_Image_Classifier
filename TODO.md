#TODO
- Settle on models.  [Link](https://github.com/jcjohnson/cnn-benchmarks) by Justin Johnson (CS231n) makes me think ResNet-34/50 is best bet
input channels with shared weights

### Hamming Distance Code
- Profile the hamming distance code (low priority, only will be run at start of training)

### Image Processing
- Use the guidelines in the Alexnet paper (scale image to size, then randomly crop), subtract colour mean

---
### Datasets
- First explore Deep learning by training various architectures on PASCAL 2007, supervised learning (Pascal is much smaller dataset, computationally much easier)
- Then explore pre-training model accuracy unsupervised on COCO, then supervised learning on PASCAL
- Compare to classification no learning of Resnet/VGG/Alex on PASCAL, and random weights, pre-trained weights of those architecture on PASCAL
- Train the jigsaw classifier on unlabeled2017.zip originally
- Perform image classification transfer learning on train, val and test 2017 dataset

### Style
- Convert all code to common style
