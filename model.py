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
        self.conv1 = nn.Conv2d(in_channels, 8, 7, stride=1)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 24, 3)
        self.conv4 = nn.Conv2d(24, 32, 3)
        self.conv5 = nn.Conv2d(32, 40, 3)
        self.pool5 = nn.MaxPool2d(2, stride=2)

        # Here will be Flatten layer (tensor.view) later while building structure of model(forward())
        self.fc1 = nn.Linear(40 * 8* 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, classes_number)

    def forward(self, x):
        x = F.relu(self.con1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        # Analogy to Flatten()
        x = x.view(-1, 24 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # TODO: Why is there no Softmax?
        x = self.fc4(x)
        return x


def train_model(train, PATH, epochs=10, save=True):
    # Optimizer, model, criterion initialization
    classes_number = len(train.dataset.classes)
    channels_number = train.dataset.data.shape[3]
    model = MyNet(classes_number=classes_number, in_channels=channels_number)

    criterion = nn.CrossEntropyLoss()
    print(criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(train):
            inputs, labels = data

            # Don't forget to make gradients zeros
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 200 == 0:
                print("epoch: ", epoch, ", batch: ", i, ", loss: ", running_loss / 200)
                running_loss = 0.0

    if save is True:
        torch.save(model.state_dict(), PATH)

    return model


def load_model(PATH, classes_number=10, channels_number=3):
    trained_model = MyNet(classes_number=classes_number, in_channels=channels_number)
    trained_model.load_state_dict(torch.load(PATH))
    return trained_model


if __name__ == '__main__':
    net = MyNet(classes_number=10, in_channels=3)
    print(net)
