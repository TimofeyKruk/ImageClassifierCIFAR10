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
        self.conv1 = nn.Conv2d(in_channels, 8, 5)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool2 = nn.AvgPool2d(2, stride=2)

        # Here will be Flatten layer (tensor.view) later while building structure of model(forward())
        self.fc1 = nn.Linear(16 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, classes_number)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Analogy to Flatten()
        x = x.view(-1, 16 * 6 * 6)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # TODO: Why is there no Softmax?
        x = F.softmax(self.fc3(x))

        return x


def train_model(train,PATH, epochs=10,save=True):
    # Optimizer, model, criterion initialization
    classes_number = len(train.dataset.classes)
    channels_number = train.dataset.data.shape[3]
    model = MyNet(classes_number=classes_number, in_channels=channels_number)

    criterion = nn.CrossEntropyLoss()
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

            running_loss+=loss.item()

        print("epoch: ",epoch,", loss: ",running_loss)

    if save is True:
        torch.save(model.state_dict(),PATH)


    return model


if __name__ == '__main__':
    net = MyNet(classes_number=10, in_channels=3)
    print(net)
