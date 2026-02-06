import torch
from data_loader import load_data_simple, split_data, loaders, mean_std, load_pcg_data, load_spectograms, ensure_deterministic
from models.transformer import Transformer, train_transformer, test_transformer
from models.late_fusion import LateFusion, train_lf, test_lf
from models.ANN import ANN, train_ann


def main():
    ensure_deterministic()
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    metrics = {"Final_loss": [], "Acc": []}

    #win_len, stride = 4000, 1000
    #X, y, nr_windows = load_pcg_data(device, win_len, stride)
    X, y = load_spectograms(device)
    Xs = load_data_simple() # Xs = 108,4 || ys = 108,5

    iterations = 5
    stride_splits = 1/iterations
    
    for iteration in range(iterations):
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, iteration, stride_splits, split=[0.70, 0.15, 0.15])
        train_loader, val_loader, test_loader = loaders(X_train, y_train, X_val, y_val, X_test, y_test)

        Xs_train, ys_train, Xs_val, ys_val, Xs_test, ys_test = split_data(Xs, y, iteration, stride_splits, split=[0.70, 0.15, 0.15])
        train_loader2, val_loader2, test_loader2 = loaders(Xs_train, ys_train, Xs_val, ys_val, Xs_test, ys_test)
        
        # 1. Train Transformer alone
        transformer = Transformer().to(device)
        train_transformer(transformer, train_loader, val_loader, device, epochs = 150)
        #metrics = test_transformer(transformer, val_loader, device, metrics)

        # 2. Train ANN alone
        ann = ANN().to(device)
        train_ann(ann, device, train_loader2, val_loader2, epochs = 100)

        # 3. Train Late Fusion Layer alone
        lf = LateFusion(transformer, ann).to(device)
        train_lf(lf, device, train_loader, val_loader, train_loader2, val_loader2, epochs=100)
        
        #metrics = test_lf(lf, train_loader, train_loader2, device, metrics)
        metrics = test_lf(lf, val_loader, val_loader2, device, metrics)

    
    mean_std(metrics)
    

if __name__ == '__main__':
    main()