import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Will write some information about structure later.
"""


class MyNet(nn.Module):

    def __init__(self, classes_number, in_channels=3):
        super(MyNet, self).__init__()

        # Initializing learning layers
        self.conv1 = nn.Conv2d(in_channels, 8, 3, stride=1)
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 4)
        self.conv3 = nn.Conv2d(16, 32, 3)
        # self.conv4 = nn.Conv2d(24, 32, 3)
        # self.conv5 = nn.Conv2d(32, 40, 3)
        self.pool2 = nn.AvgPool2d(2, stride=2)

        # Here will be Flatten layer (tensor.view) later while building structure of model(forward())
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, classes_number)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        # Analogy to Flatten()
        x = x.view(-1, 32 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train_model(train, PATH, cuda=False, epochs=10, save=True):
    # Optimizer, model, criterion initialization
    classes_number = len(train.dataset.classes)
    channels_number = train.dataset.data.shape[3]
    myNet = MyNet(classes_number=classes_number, in_channels=channels_number)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(myNet.parameters(), lr=0.01)

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
