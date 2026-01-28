import torch
from data_loader import load_data_simple, split_data, loaders, mean_std, load_pcg_data
from models.ANN import ANN, train_ann, test_ann
from models.transformer import Transformer, train_transformer, test_transformer

def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    metrics = {"Final_loss": [], "Acc": []}

    #X, y = load_data_simple()
    win_len = 8000
    stride = 2000
    X, y, nr_windows = load_pcg_data(device, win_len, stride)
    print(X.shape)
    print(y.shape)

    iterations = 5
    stride_splits = 1/iterations

    for iteration in range(iterations):
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, iteration, stride_splits, split=[0.70, 0.15, 0.15])
        train_loader, val_loader, test_loader = loaders(X_train, y_train, X_val, y_val, X_test, y_test)
        
        transformer = Transformer(nr_windows, win_len).to(device)
        train_transformer(transformer, train_loader, val_loader, device, epochs = 150)
        test_transformer(transformer, val_loader, device, metrics)
        
        '''
        ann = ANN().to(device)
        train_ann(ann, device, train_loader, val_loader)
        metrics = test_ann(ann, device, val_loader, test_loader, metrics, mode="val") # mode = "val" vs. "test"
        '''
    
    mean_std(metrics)

if __name__ == '__main__':
    main()