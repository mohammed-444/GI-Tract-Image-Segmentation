# GI-Tract-Image-Segmentation
image segmentation problem which try to segment stomach and intestines from MRI image
# UW-Madison GI Tract Image Segmentation Competition

## Introduction

This is our submission for the UW-Madison GI Tract Image Segmentation Competition. The goal of the competition is to develop a model that can accurately segment images of the gastrointestinal tract from MRi scans.

## **Table Of Contents**
- [GI-Tract-Image-Segmentation](#gi-tract-image-segmentation)
- [UW-Madison GI Tract Image Segmentation Competition](#uw-madison-gi-tract-image-segmentation-competition)
  - [Introduction](#introduction)
  - [**Table Of Contents**](#table-of-contents)
  - [Approach](#approach)
  - [Reproducing Results](#reproducing-results)
  - [Additional Resources](#additional-resources)
  - [Conclusion](#conclusion)

## Approach

My approach for this competition was to use a fully convolutional neural network (FCN) with a ResNet-50 backbone. I trained the model using the provided dataset of images and annotation masks. DuriI used data augmentation techniques during training as random rotation and flipping to improve the model's robustness.

## Reproducing Results

To reproduce my results, you will need to install the following dependencies:
- Python 3.8+
- TensorFlow 2.0+
- Numpy
- OpenCV

You can install the dependencies by running the following command:
```
pip install -r requirements.txt
```

The code for training and evaluating the model can be found in the `train.py` and `eval.py` files respectively. You can train the model by running the following command:
```
python train.py --dataset_path path/to/dataset --output_path path/to/output/folder
```

You can evaluate the model by running the following command:
```
python eval.py --model_path path/to/trained/model --dataset_path path/to/dataset
```

## Additional Resources

- [UW-Madison GI Tract Image Segmentation Competition](https://github.com/uw-madison-github/gi-tract-image-segmentation-competition)
- [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

## Conclusion

This is my submission for the UW-Madison GI Tract Image Segmentation Competition. I hope that my approach and code will be helpful for others who are interested in this field.


