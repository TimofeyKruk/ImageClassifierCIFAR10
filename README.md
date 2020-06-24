# Convolutional NN for CIFAR10 image classification
**Written with PyTorch API**  

Used CIFAR10 image (32,32,3) dataset for training and testing.
During the investigation process trained different structures.  
*The architectures of neural networks and results (accuracy) are listed below:*  

## 2 Convolutional + 3 Dense(FC) model
The first attempt was with quite simple model structure:

```bash
MyNet(
  (conv1): Conv2d(3, 8, kernel_size=(5, 5), stride=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
  (pool2): AvgPool2d(kernel_size=2, stride=2, padding=0)
  (fc1): Linear(in_features=576, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=32, bias=True)
  (fc3): Linear(in_features=32, out_features=10, bias=True)
)
```

**Accuracy** on testing data is **54%**.
A confusion matrix is in Jupyter Notebook or below.
![изображение](https://user-images.githubusercontent.com/43128663/85545429-9dc17280-b624-11ea-834c-256fa8f47f89.png)

## Deeper CNN with batch normalization and dropout
Added more convolutional and fully connected layers.  
**+Batch Normalization and Dropout.  
+Mini-batch gradient descent.  
+SGD Optimizer with learning rate scheduler (custom LR decay analogy).**  
  
Tried different structures and hyperparameters *(learning rate, epochs number, kernel sizes and etc.)*.  
The best results were achieved with the following structure:

```bash
MyNet(
  (conv1): Conv2d(3, 20, kernel_size=(3, 3), stride=(1, 1))
  (bn_conv1): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
  (conv2): Conv2d(20, 24, kernel_size=(3, 3), stride=(1, 1))
  (conv3): Conv2d(24, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv4): Conv2d(32, 40, kernel_size=(3, 3), stride=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv5): Conv2d(40, 48, kernel_size=(3, 3), stride=(1, 1))
  (bn_conv5): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
  (conv6): Conv2d(48, 56, kernel_size=(3, 3), stride=(1, 1))
  (conv7): Conv2d(56, 64, kernel_size=(3, 3), stride=(1, 1))
  (conv8): Conv2d(64, 72, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=1152, out_features=256, bias=True)
  (bn_fc1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
  (dropout1): Dropout(p=0.3, inplace=False)
  (fc2): Linear(in_features=256, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=32, bias=True)
  (dropout3): Dropout(p=0.1, inplace=False)
  (fc4): Linear(in_features=32, out_features=16, bias=True)
  (fc5): Linear(in_features=16, out_features=10, bias=True)
)
```

**Accuracy** on testing data is **69%**.
Also used Tensorboard for training analysis.  
An image below shows test accuracy of current **(green: 69%)** versus less deep NN **(red: < 60%)**  
(x axis is for training process)

![изображение](https://user-images.githubusercontent.com/43128663/85561352-2dbae880-b634-11ea-89db-c9db96f82b2c.png)

Losses while training(orange-test, gray-train)

![изображение](https://user-images.githubusercontent.com/43128663/85562468-2e07b380-b635-11ea-8c42-51dc2db0d3c3.png)

## ResNet18
The famous build-in architecture of the model that was trained from the beginning (not pretrained).  
**Accuracy** on testing data is **71%**.

Losses (test&train) while learning and test accuracy:  

![изображение](https://user-images.githubusercontent.com/43128663/85579579-a83f3480-b643-11ea-994a-0ed9ee4e24ba.png)


#### Why is ResNet18 performance so unsatisfying?
**My own assumptions:**  
* For a deep model 8 epochs can be not enough to grasp the data distribution.
* After 4th epoch (custom scheduler is responsible for this) learning rate decreases to 0.001.  
Due to this it will take longer to converge to the local/global(/just small enough) minimum of cost function.
* Data: 50 000 train images can be still not enough to learn the dependencies of data.

#### Contacts
**Created by Timofey Kruk**  
kruktimofey@gmail.com  
[Skype](https://join.skype.com/invite/o4jkACciB3Nh)

 
