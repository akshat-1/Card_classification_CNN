# CARD CLASSIFICATION CNN

This project aims to real time classify playing cards using Convolutional Neural Networks (CNNs). The model is trained on a dataset containing images of various playing cards.

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [References](#references)
- [Improvements](#improvements)


## Dataset Overview

The dataset used for training and evaluation contains a diverse collection of playing card images. It includes images of cards from different decks, suits, and ranks. The dataset is preprocessed to ensure consistency in image size and format.

- Dataset Size: 7794 images
- Classes: 53 (One class for each type of playing card)
- Image Size: 224 x 224 pixels (RGB format)
- Train-Validation-Test Split: 7624 images / 265 images / 265 images

For more information about the dataset, refer to [this link](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification).

## Model Architecture

The model architecture consists of a series of convolutional layers followed by fully connected layers. Here's an overview of the model architecture:

<img src="image.png">

### Convolutional Layers:

- The model begins with two convolutional layers (`conv1` and `conv2`), which high level extract features from the input images.
- The next two convolutional layers (`conv3` and `conv4`) are used to extract low level features
- Each convolutional layer is followed by a Rectified Linear Unit (ReLU) activation function (`relu1` to `relu5`) to introduce non-linearity to the model, and relu choosen to avoid vanishing gradient problem.
- Max-pooling layers (`pool1` to `pool3`) are applied after each convolutional layer to downsample the feature maps (by half in both dimensions) and reduce spatial dimensions, except for the last layer



### Fully Connected Layers:

- Following the convolutional layers, the feature maps are flattened and passed through two fully connected layers (`fc1` and `fc2`).
- The first fully connected layer (`fc1`) has 512 neurons and applies a ReLU activation function (`relu5`).
- The final fully connected layer (`fc2`) outputs logits for each class without applying an activation function.

### Input and Output:

- Input images are assumed to have three channels (RGB).
- The output layer has `num_classes` neurons, where `num_classes` represents the number of classes for classification (default: 53).

### Forward Pass:

- During the forward pass, input images (`x`) undergo convolutional operations, followed by activation functions and max-pooling.
- The resulting feature maps are flattened and passed through fully connected layers to generate class logits.

Overall, the `CardClassifierCNN` architecture employs convolutional and fully connected layers to learn hierarchical representations of playing card images and make predictions based on these representations.

For more details, refer to the Model Architecture section in the code.

## Training Process

The model is trained using the Adam optimizer with the Cross-Entropy Loss function. Training is performed over multiple epochs, with early stopping implemented to prevent overfitting. Training progress and performance metrics are monitored using validation data.

For detailed information about the training process, refer to the Training Process section in the code.

<img src="loss.png">


## Usage

To use the model for inference, follow these steps:

1. Install the required dependencies (specified in the Dependencies section).
2. Clone the repository to your local machine.
3. Download the dataset and place it in the appropriate directory.
4. Run main script in src folder to train the model
5. Run test scipt in src folder in order to test the model on test dataset
6. run Card_detector script for real time card detection.

## Dependencies

Ensure you have the following dependencies installed:

- Python (version 3.9)
- PyTorch (version 2.1.2)
- Matplotlib
- NumPy (version 1.26.3)
- scikit-learn

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification)
- [Related Paper](https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier)

## improvements

Current test accuracy 92.08%

- Implement inception module and residual blocks with identity connections and make model deeper.