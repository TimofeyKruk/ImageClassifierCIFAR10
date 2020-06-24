# Convolutional NN for CIFAR10 image classification
Used CIFAR10 image (32,32,3) dataset for training and testing.
During the investigation process different structures were trained.  
The architectures of neural networks and results (loss and accuracy) are listed below:  


## 2conv + 3fc NN
The first attempt was with a simple model structure:
'''bash
MyNet(
  (conv1): Conv2d(3, 8, kernel_size=(5, 5), stride=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
  (pool2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (fc1): Linear(in_features=576, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=32, bias=True)
  (fc3): Linear(in_features=32, out_features=10, bias=True)
)
'''
*Accuracy* on testing data is *54%*.
Confusion matrix is in Jupyter Notebook or below.
![изображение](https://user-images.githubusercontent.com/43128663/85545429-9dc17280-b624-11ea-834c-256fa8f47f89.png)

## Deeper CNN with batch normalization and dropout
The structure is the following (Autogenerated by PyTorch):
'''bash
....Work in progress...
'''
*Accuracy* on testing data is *@@@@@@%*.

## ResNet18
A famous build-in architecture of a model that was trained from the beginning.

*Accuracy* on testing data is *@@@@@@%*.