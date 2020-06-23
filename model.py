import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

"""
Will write some information about structure later.
"""


class MyNet(nn.Module):

    def __init__(self, classes_number, in_channels=3):
        super(MyNet, self).__init__()

        # Initializing learning layers
        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=1)
        self.bn_conv1 = nn.BatchNorm2d(16, affine=False)

        self.conv2 = nn.Conv2d(16, 24, 3)
        self.conv3 = nn.Conv2d(24, 32, 3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(32, 40, 3)
        self.bn_conv4 = nn.BatchNorm2d(40, affine=True)
        self.conv5 = nn.Conv2d(40, 48, 3)
        self.conv6 = nn.Conv2d(48, 56, 3)
        self.conv7 = nn.Conv2d(56, 64, 3)

        # self.pool2 = nn.MaxPool2d(2, stride=2)

        # Here will be Flatten layer (tensor.view) later while building structure of model(forward())
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.bn_fc1 = nn.BatchNorm1d(256, affine=False)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, classes_number)

    def forward(self, x):
        x = self.bn_conv1(self.conv1(x))
        x = F.relu(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)

        x = F.relu(self.bn_conv4(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        # Analogy to Flatten()
        x = x.view(-1, 64 * 5 * 5)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x


def train_model(train, PATH, cuda=False, epochs=10, save=True):
    # Optimizer, model, criterion initialization
    classes_number = len(train.dataset.classes)
    channels_number = train.dataset.data.shape[3]
    myNet = MyNet(classes_number=classes_number, in_channels=channels_number)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(myNet.parameters(), lr=0.01, weight_decay=0)

    # NEW: Trying to wrap optimizer with sheduler (for lr decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[3, 8], gamma=0.1)

    device = None
    if cuda is True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", device)
        myNet.to(device)

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(train):
            if cuda is True:
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data

            # Don't forget to make gradients zeros
            optimizer.zero_grad()

            outputs = myNet(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 200 == 0:
                print("epoch: ", epoch, ", batch: ", i, ", loss: ", running_loss / 200)
                running_loss = 0.0

        scheduler.step()

    if save is True:
        torch.save(myNet.state_dict(), PATH)
        print("Model was successfully saved!")

    return myNet


def load_model(PATH, classes_number=10, channels_number=3):
    trained_model = MyNet(classes_number=classes_number, in_channels=channels_number)
    trained_model.load_state_dict(torch.load(PATH))
    return trained_model


if __name__ == '__main__':
    net = MyNet(classes_number=10, in_channels=3)
    print(net)
