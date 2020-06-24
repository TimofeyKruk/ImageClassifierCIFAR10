import data_preparation
import model
import torch
import torchvision
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

"""Main module that loads data, trains and validates the model.
 If I don't forget, also would like to save model parameters for 
 future use."""

# Add command variable whether to learn or load model
if __name__ == "__main__":
    PATH = "./SavedModel"
    model_architecture = "Custom"
    # Command variables parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to save/load model weights",
                        default=PATH)
    parser.add_argument("--train", help="boolean value whether train or load the model",
                        default=False)
    parser.add_argument("--model", help="for using custom structure enter \"Custom\"",
                        default=model_architecture)

    args = parser.parse_args()
    PATH = args.path
    model_architecture = args.model
    train_bool = bool(args.train)

    train, test = data_preparation.downloadData(download=False)
    train_l, test_l = data_preparation.loadData(train, test)

    trained_model = None

    writer = SummaryWriter("runs/cifar10_SGD_resNet18")

    if train_bool is True:
        # TRAINING
        print("Training model:", model_architecture)
        print("Path to save: ", PATH)

        classes_number = len(train_l.dataset.classes)
        channels_number = train_l.dataset.data.shape[3]

        if model_architecture == "Custom":
            myNet = model.MyNet(classes_number=classes_number, in_channels=channels_number)
            trained_model = model.train_model(myNet, train_l, test_l, PATH, writer, cuda=True, epochs=10, save=True)
            print("Model has been trained!")
            print(trained_model)
        else:
            resNet=torchvision.models.resnet18()
            resNet.fc=torch.nn.Linear(in_features=512, out_features=10)
            print(resNet)

            trained_model = model.train_model(resNet, train_l, test_l, PATH, writer, cuda=True, epochs=7, save=True)


    else:
        # Loading previously trained model
        print("Loading model")
        trained_model = model.load_model(PATH)
        print(trained_model)

    # Make prediction on test data and saving them for future analysis in jupyter notebook
    correctly_classified = 0
    total = 0

    all_labels = []
    all_predictions = []

    # Accuracy counting for test
    with torch.no_grad():
        for batch in test_l:
            images, labels = batch

            predictions = trained_model(images)
            _, argmax_predictions = torch.max(predictions.data, 1)

            correctly_classified += (argmax_predictions == labels).sum().item()
            total += labels.size(0)

            all_labels = np.append(all_labels, labels.numpy())
            all_predictions = np.append(all_predictions, argmax_predictions.numpy())

    writer.add_graph(trained_model, images)
    print("Model graph was added to tensorboard!")
    writer.close()

    np.savetxt("LabelsTest2.csv", all_labels, delimiter=',')
    np.savetxt("PredictionsTest2.csv", all_predictions, delimiter=',')

    accuracy = correctly_classified / total * 100
    print("The model accuracy is: ", accuracy)
