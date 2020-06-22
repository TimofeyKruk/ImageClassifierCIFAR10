import torchvision
from torch.utils.data import dataloader
import torchvision.transforms as transforms

"""""
Working with CIFAR10 dataset.
Loading and normalizing dataset.
"""


def downloadData(download=False):
    """
    Downloading data and normalizing images from CIFAR10.

    :return: train,test <- datasets for feature loading
    """

    # Adding transform (as i get it is a pipeline analogy)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Something new for me. First parameter is means for each channel
        # TODO: why did I decide that there is 3 channels, better parameterize
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = torchvision.datasets.CIFAR10(root="./data_CIFAR10",
                                         train=True,
                                         download=download,
                                         transform=transform)

    test = torchvision.datasets.CIFAR10(root="./data_CIFAR10",
                                        train=False,
                                        download=download,
                                        transform=transform)

    return train, test


def loadData(train, test):
    """"Loading the dataset from memory"""

    train_loader = dataloader.DataLoader(train,
                                         batch_size=16,
                                         shuffle=True,
                                         num_workers=2)

    test_loader = dataloader.DataLoader(test,
                                        batch_size=16,
                                        shuffle=False,
                                        num_workers=2)

    return train_loader, test_loader


# Only for local checking of feasibility
if __name__ == "__main__":
    train, test = downloadData(download=False)
    train_l, test_l = loadData(train, test)
    classes_names = train_l.dataset.classes

    print(classes_names)
