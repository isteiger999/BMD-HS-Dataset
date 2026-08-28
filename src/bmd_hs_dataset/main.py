import torch  # pyright: ignore[reportMissingImports]
import os, sys
from create_spectrorgrams import create_spectrograms
from bmd_hs_dataset.data_loader import loaders, mean_std, ensure_deterministic, create_train_val_test_split, create_simple
from bmd_hs_dataset.models.transformer import Transformer, train_transformer
from bmd_hs_dataset.models.late_fusion import LateFusion, train_lf, test_lf
from bmd_hs_dataset.models.ANN import ANN, train_ann


def main():
    ensure_deterministic()
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    if not os.path.exists('src/bmd_hs_dataset/data/spectrograms'):
        create_spectrograms()
    
    metrics = {"Final_loss": [], "Acc": [], "F1": []}

    iterations = 5
    stride_splits = 1/iterations
    split=[0.70, 0.15, 0.15]
    
    for iteration in range(iterations):
        # X_train.shape: [608, 1, 64, 401], y_train.shape: [608, 5]
        X_train, y_train, X_val, y_val, X_test, y_test = create_train_val_test_split(split, stride_splits, device, iteration)
        X_train_meta, X_val_meta, X_test_meta = create_simple(split, stride_splits, device, iteration)
        train_loader, val_loader, test_loader = loaders(X_train, y_train, X_val, y_val, X_test, y_test, X_train_meta, X_val_meta, X_test_meta)
        
        # 1. Train Transformer alone
        transformer = Transformer(device).to(device)
        train_transformer(transformer, train_loader, val_loader, device, epochs = 150)
        #metrics = test_transformer(transformer, val_loader, device, metrics)

        # 2. Train ANN alone
        ann = ANN().to(device)
        train_ann(ann, device, train_loader, val_loader, epochs = 100)

        # 3. Train Late Fusion Layer alone
        lf = LateFusion(transformer, ann).to(device)
        train_lf(lf, device, train_loader, val_loader, epochs=100)
        
        #metrics = test_lf(lf, train_loader, train_loader2, device, metrics)
        metrics = test_lf(lf, val_loader, device, metrics)

    
    mean_std(metrics)
    

if __name__ == '__main__':
    main()