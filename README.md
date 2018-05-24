# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Certificate

![Screenshot](certificate.png)

## Getting Started
### Environment
Tested on the following environment:
- Python 3.6.5:
    - Numpy
    - PyTorch
    - TorchVision

- NVIDAI GPU (driver version 390.48)
- CUDA 9.1

### Sample Data
Download sample data using curl:

```
curl -O https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz
```

And extract using tar:
```
mkdir flowers
tar -xvzf flower_data.tar.gz -C flowers
```

## Training
### Usage
```
python train.py [-h] [--save_dir SAVE_DIR]
                [--arch {densenet161,vgg16_bn,resnet18,resnet34,resnet50,densenet201,vgg13_bn,densenet121,vgg19_bn}]
                [--learning_rate LEARNING_RATE] [--hidden_units HIDDEN_UNITS]
                [--epochs EPOCHS] [--gpu] 
                data_dir

positional arguments:
  data_dir              Directory used to locate source images

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Directory used to save checkpoints
  --arch {densenet161,vgg16_bn,resnet18,resnet34,resnet50,densenet201,vgg13_bn,densenet121,vgg19_bn}
                        Model architecture to use for training
  --learning_rate LEARNING_RATE
                        Learning rate hyperparameter
  --hidden_units HIDDEN_UNITS
                        Number of hidden units hyperparameter
  --epochs EPOCHS       Number of epochs used to train model
  --gpu                 Use GPU for training
```

### Examples
1. The following will train a densenet model on the GPU for 9 epochs:

```
python train.py flowers --arch densenet121 --epochs 9 --gpu --pin_memory --num_workers 4
```

2. The following will train a vgg model on the GPU for 7 epochs and place checkpoint in checkpoints directory:

```
python train.py flowers --arch vgg13_bn --epochs 7 --gpu --pin_memory --num_workers 4 --save_dir checkpoints
```

3. The following will train a resnet model on the CPU for 3 epochs and place checkpoint in checkpoints directory:

```
python train.py flowers --arch resnet18 --epochs 3 --save_dir checkpoints
```

## Inference
### Usage
```
python predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu] [--verbose]
                  input checkpoint

positional arguments:
  input                 Input image
  checkpoint            Model checkpoint file to use for prediction

optional arguments:
  -h, --help            show this help message and exit
  --top_k TOP_K         Return top k most likely classes
  --category_names CATEGORY_NAMES
                        Mapping file used to map categories to real names
  --gpu                 Use GPU for prediction
  --verbose             Display additional processing information
```
### Examples
1. The following will return the most likely class using a densenet checkpoint executing on the GPU:

```
python predict.py flowers/test/28/image_05230.jpg densenet121_checkpoint.pth --gpu
```

2. The following will return the top 5 most likely classes using a vgg checkpoint executing on the GPU and map classes to categories using a mapping file:

```
python predict.py flowers/test/28/image_05230.jpg checkpoints/vgg13_bn_checkpoint.pth --gpu --top_k 5 --category_names cat_to_name.json
```

3. The following will return the most likely class using a resnet checkpoint executing on the CPU:
```
python predict.py flowers/test/28/image_05230.jpg checkpoints/resnet18_checkpoint.pth
```

## Unit Testing
## Training
Run unit tests for training using the GPU (required to generate checkpoints for inference GPU test cases):

```
python -m unittest test_model_helper.TrainingGpuTestCase
```

Run unit tests for training using the CPU (required to generate checkpoints for inference CPU test cases):

```
python -m unittest test_model_helper.TrainingCpuTestCase
```
## Inference
Run unit tests for inference using the GPU (assumes checkpoints have been generated):

```
python -m unittest test_model_helper.InferenceGpuTestCase
```

Run unit tests for inference using the CPU (assumes checkpoints have been generated):

```
python -m unittest test_model_helper.InferenceCpuTestCase
```
