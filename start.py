import torch
import model

"""Main module that loads data, trains and validates the model.
 If I don't forget, also would like to save model parameters for 
 future use."""

if __name__ == "__main__":
    a = torch.empty(5, 3)
    print(a)

    c = torch.tensor([[5.3, 3, 22], [5.3, 3, 22]])

    c = c.view(-1)
    print(c[1:4])
