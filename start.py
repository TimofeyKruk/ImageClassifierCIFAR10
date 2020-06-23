import data_preparation
import model
import torch
import argparse
import numpy as np

"""Main module that loads data, trains and validates the model.
 If I don't forget, also would like to save model parameters for 
 future use."""

# Add command variable whether to learn or load model
if __name__ == "__main__":
    PATH = "./SavedModel"

    # Command variables parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="boolean value whether train or load the model",
                        default=False)
    args = parser.parse_args()

    train, test = data_preparation.downloadData(download=False)
    train_l, test_l = data_preparation.loadData(train, test)

    train_bool = bool(args.train)
    trained_model = None

    if train_bool is True:
        # TRAINING
        print("Training model")
        trained_model = model.train_model(train_l, PATH, cuda=True, epochs=5, save=True)
        print("Model has been trained!")
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

    with torch.no_grad():
        for batch in test_l:
            images, labels = batch

            predictions = trained_model(images)
            _, argmax_predictions = torch.max(predictions.data, 1)

            correctly_classified += (argmax_predictions == labels).sum().item()
            total += labels.size(0)

            all_labels = np.append(all_labels, labels.numpy())
            all_predictions = np.append(all_predictions, argmax_predictions.numpy())

    np.savetxt("LabelsTest1.csv", all_labels, delimiter=',')
    np.savetxt("PredictionsTest1.csv", all_predictions, delimiter=',')

    accuracy = correctly_classified / total * 100
    print("The model accuracy is: ", accuracy)

    print(torch.cuda.is_available())
