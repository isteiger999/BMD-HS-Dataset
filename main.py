import torch

from data_loader import load_data_simple, split_data, loaders
from models.ANN import ANN, train_ann

def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    X, y = load_data_simple()
    iterations = 5
    stride = 1/iterations

    for iteration in range(iterations):
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, iteration, stride, split=[0.70, 0.15, 0.15])
        train_loader, val_loader, test_loader = loaders(X_train, y_train, X_val, y_val, X_test, y_test)

        ann = ANN().to(device)
        train_ann(ann, device, train_loader, val_loader)



if __name__ == '__main__':
    main()