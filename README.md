# Pytorch-ENet-Nice
Pytorch to train ENet of Cityscapes datasets and CamVid datasets nicely

# Nice discovery

coming soon!

# Environment

    pytorch1.0
    python3.6

# Train Cityscapes datasets
  Please download the fine labeled Cityscapes dataset **leftImg8bit_trainvaltest.zip (11GB)** and the corresponding ground truth **gtFine_trainvaltest.zip (241MB)** from the [Cityscapes website](https://www.cityscapes-dataset.com/). In addition, clone the cityscapesScripts repository:

    $ git clone https://github.com/mcordts/cityscapesScripts.git

After that, run the `/preparation/createTrainIdLabelImags.py` script, to convert annotations in polygonal format to png images with label IDs, where pixels encode "train IDs" (that you can define in `labels.py`). Since we use the default 19 classes, you do not need to change anything in the `labels.py` script. The input data layer which is used requires a text file of white-space separated paths to the images and the corresponding ground truth. 

## Data preprocess

Accelerated training (coming soon!)

## Train ENet encoder architecture

## Train ENet encoder-decoder architecture

pretrain model (coming soon!)

# Train CamVid datasets

## Data preprocess

Accelerated training(coming soon!)

## Train ENet encoder architecture

## Train ENet encoder-decoder architecture

pretrain model(coming soon!)
