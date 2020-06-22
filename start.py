import data_preparation
import model
import torch
import argparse

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
        trained_model = model.train_model(train_l, PATH, epochs=4, save=True)
    else:
        # Loading previously trained model
        print("Loading model")

        classes_number = len(train_l.dataset.classes)
        channels_number = train_l.dataset.data.shape[3]

        trained_model = model.MyNet(classes_number=classes_number, in_channels=channels_number)
        trained_model.load_state_dict(torch.load(PATH))
        print(trained_model)

    # Make prediction on test data and saving them for future analysis in jupyter notebook
    correctly_classified = 0
    total = 0

    with torch.no_grad():
        for batch in test_l:
            images, labels = batch

            total += labels.size(0)

            predictions = trained_model(images)
            _, max_predictions = torch.max(predictions.data, 1)
            print(max_predictions)
            print("______________________________")
            correctly_classified += (max_predictions == labels).sum().item()

    accuracy = correctly_classified / total * 100
    print("The accuracy of a model is: ", accuracy)
