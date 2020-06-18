import data_preparation
import model


"""Main module that loads data, trains and validates the model.
 If I don't forget, also would like to save model parameters for 
 future use."""

if __name__ == "__main__":
    train, test = data_preparation.downloadData(download=False)
    train_l, test_l = data_preparation.loadData(train, test)

    trained_model=model.train_model(train_l,"./SavedModel",epochs=15,save=True)