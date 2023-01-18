# **UW-Madison GI Tract Image Segmentation Competition**

## **Introduction**

This is our submission for the UW-Madison GI Tract Image Segmentation Competition. The goal of the competition is to develop a model that can accurately segment images of the gastrointestinal tract from MRi scans.

## **Table Of Contents**
- [**UW-Madison GI Tract Image Segmentation Competition**](#uw-madison-gi-tract-image-segmentation-competition)
  - [**Introduction**](#introduction)
  - [**Table Of Contents**](#table-of-contents)
  - [**Data**](#data)
  - [**Preprocessing**](#preprocessing)
  - [**Model creation**](#model-creation)
    - [Our U-Net Model](#our-u-net-model)
    - [**EFFICIENT Net B7 , Pre-trained Model**](#efficient-net-b7--pre-trained-model)
  - [Visualizing Results](#visualizing-results)
  - [Reproducing Results](#reproducing-results)
  - [Resources](#resources)

## **Data**
  - The data can be found on the [Data Tab](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data) on kaggle competition page.

  - This competition involves segmenting organs cells in images using RLE-encoded masks and 16-bit grayscale PNG format. The test set is entirely unseen and consists of roughly 50 cases with varying numbers of days and slices. The goal is to generalize to both partially and wholly unseen cases.
## **Preprocessing**
  - Each slice were connected to it’s available masks that were found in the Train.csv
  - A New generator `DataGenerator1D()` was  inherited from `tf.keras.utils.Sequence` class, where we adjusted the `__init__()` for the use of our data
  - decoded the RLE masks through `rle_dcode()` function that were inserted to the generator getter `__getitem__()`
  - Data were read through the `DataGenerator()` for training


## **Model creation**
  ### Our U-Net Model
  - UNet model that consists of 4 2D-Convolutional blocks and 4 2D-Deconvolutional (Convolution transpose)
  - Model Input was defined was `n_calsses` = 4 and `input_shape` = (128,128,3)
  - the UNet model was compiled with `adam` optimizer while monitoring the following metrics, accuracy,  `jacard_coef` that we defined in our notebook and `iou_score, f1_score, f2_score, precision, recall` ,that were imported from  `segmentation_models.metrics`
  - while the loss was defined by `bce_dice_loss` that was imported from `segmentation_models.losses`
  - the model achieved the flowing results after `20` epochs of training with best results callback
      
      
      | **Training Params** | **Value** | **Testing Params** | **Value** |
      | --- | --- | --- | --- |
      | **loss** | 0.0917  | **val_accuracy** | 0.1662 |
      | **accuracy** | 0.7223 | **val_accuracy** | 0.8184 |
      | **jacard_coef** | 0.7347 | **val_jacard_coef** | 0.6051 |
      | **iou_score** | 0.7407 | **val_iou_score** | 0.4967 |
      | **precision** | 0.8478 | **val_precision** | 0.6733 |
      | **recall** | 0.8529  | **val_recall** | 0.7160 |
      | **f1-score** | 0.8492 | **val_f1-score** | 0.6121 |
      | **f2-score** | 0.8511 | **val_f2-score** | 0.6005 |
### **EFFICIENT Net B7 , Pre-trained Model**
        
  - The `Unet`model was imported from `segmentation_models` and was defined was the following arguments Model = `efficientnetb7` , `input_shape` = (128,128,3) , `classes` = 3, `activation` = “sigmoid” and `encoder_weights` = “imagenet” which are the weights of the model pre-trained on Image net dataset
  
- the EFFICIENT Net B7model was compiled with the same`adam` optimizer while monitoring the metrics, `dice_coef` which is the same as `jacard_coef` , we didn’t focus on tracking the accuracy this time though.
- IT was then trained on the same Train and validation generators used on our previous model but it achieved slightly better results on just `13` epochs where the best model was saved through the callback
  
        
    | **Training Params** | **Value** | **Testing Params** | **Value** |
    | --- | --- | --- | --- |
    | **loss** | 0.0719 | **val_loss** | 0.1472 |
    | **dice_coef** | 0.8786 | **val_dice_coef** | 0.7732 |
    | **iou_score** | 0.7897 | **val_iou_score** | 0.5312 |
    | **precision** | 0.8772 | **val_precision** | 0.6978 |
    | **recall** | 0.8870 | **val_recall** | 0.7464 |
    | **f1-score** | 0.8813  | **val_f1-score** | 0.6457 |
    | **f2-score** | 0.8845 | **val_f2-score** | 0.6334 |
    ---

## Visualizing Results
check our [Results](Results.ipynb) notebook to see the results of our model 

## Reproducing Results

To reproduce our results, you will need to install the following dependencies:
- Python
- TensorFlow 
- plotly, matplotlib
- segmentation_models

we made it easier by creating Utils folder that contains all the function needed to reprodude the results


## Resources

- [UW-Madison GI Tract Image Segmentation Competition](https://github.com/uw-madison-github/gi-tract-image-segmentation-competition)
- [Segmentation models](github.com/qubvel/segmentation_models)
- [Our Paper](Pdfs\FinalPaperSubmission.pdf)




