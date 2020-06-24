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
        self.conv1 = nn.Conv2d(in_channels, 20, 3, stride=1)
        self.bn_conv1 = nn.BatchNorm2d(20, affine=False)

        self.conv2 = nn.Conv2d(20, 24, 3)
        self.conv3 = nn.Conv2d(24, 32, 3)
        self.conv4 = nn.Conv2d(32, 40, 3)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(40, 48, 3)
        self.bn_conv5 = nn.BatchNorm2d(48, affine=False)

        self.conv6 = nn.Conv2d(48, 56, 3)
        self.conv7 = nn.Conv2d(56, 64, 3)
        self.conv8 = nn.Conv2d(64, 72, 3)

        # self.pool2 = nn.MaxPool2d(2, stride=2)

        # Here will be Flatten layer (tensor.view) later while building structure of model(forward())
        self.fc1 = nn.Linear(72 * 4 * 4, 256)
        self.bn_fc1 = nn.BatchNorm1d(256, affine=False)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, classes_number)

    def forward(self, x):
        x = self.bn_conv1(self.conv1(x))
        x = F.relu(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)

        x = F.relu(self.bn_conv5(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        # Analogy to Flatten()
        x = x.view(-1, 72 * 4 * 4)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x


def train_model(myNet, train, test, PATH, tensorboard, cuda=False, epochs=10, save=True) -> MyNet:
    # Optimizer, criterion initialization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(myNet.parameters(), lr=0.01)

    # NEW: Trying to wrap optimizer with sheduler (for lr decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[5, 7, 9], gamma=0.1)

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

            if (i + 1) % 300 == 0:
                test_loss = 0.0

                correctly_classified = 0
                total = 0
                # Accuracy counting for test
                with torch.no_grad():
                    for batch in test:
                        images, labels = batch[0].to(device), batch[1].to(device)

                        predictions = myNet(images)

                        # Gathering test loss among mini-batches
                        test_loss += criterion(predictions, labels).item

                        _, argmax_predictions = torch.max(predictions.data, 1)

                        correctly_classified += (argmax_predictions == labels).sum().item()
                        total += labels.size(0)

                test_loss /= len(test)

                # loss
                tensorboard.add_scalar("loss", {
                    "training_loss": running_loss / 300,
                    "test_loss": test_loss
                }, epoch * 11 + i // 300)
                
                tensorboard.add_scalar("test_accuracy", correctly_classified / total, epoch * 11 + i // 300)

                print("epoch: ", epoch, ", batch: ", i, ", loss: ", running_loss / 300)
                running_loss = 0.0

        print("Current LR: ", scheduler.get_last_lr())
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
