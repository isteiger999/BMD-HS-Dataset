import torch
from data_loader import load_data_simple, split_data, loaders, mean_std
from models.ANN import ANN, train_ann, test_ann

def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    metrics = {"Final_loss": [], "Acc": []}

    X, y = load_data_simple()
    iterations = 5
    stride = 1/iterations

    for iteration in range(iterations):
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, iteration, stride, split=[0.70, 0.15, 0.15])
        train_loader, val_loader, test_loader = loaders(X_train, y_train, X_val, y_val, X_test, y_test)

        ann = ANN().to(device)
        train_ann(ann, device, train_loader, val_loader)
        metrics = test_ann(ann, device, val_loader, test_loader, metrics, mode="val") # mode = "val" vs. "test"

    mean_std(metrics)

if __name__ == '__main__':
    main()