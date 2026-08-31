import torch  # pyright: ignore[reportMissingImports]
import os, sys
from create_spectrorgrams import create_spectrograms
import bmd_hs_dataset.data_loader as data_laoder
from bmd_hs_dataset.models.transformer import Transformer, train_transformer
from bmd_hs_dataset.models.late_fusion import LateFusion, train_lf, test_lf
from bmd_hs_dataset.models.ANN import ANN, train_ann


def main():
    data_laoder.ensure_deterministic()
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    if not os.path.exists('src/bmd_hs_dataset/data/spectrograms'):
        create_spectrograms()
    
    metrics = {"Final_loss": [], "Acc": [], "F1": []}

    iterations = 5
    stride_splits = 1/iterations
    split=[0.70, 0.15, 0.15]
    
    for iteration in range(iterations):
        # 1. Create X_train, X_val, X_test
        # X_train.shape: [608, 1, H, W], y_train.shape: [608, 5]
        X_train, y_train, X_val, y_val, X_test, y_test = data_laoder.create_train_val_test_split(split, stride_splits, device, iteration)
        X_train_meta, X_val_meta, X_test_meta = data_laoder.create_simple(split, stride_splits, device, iteration)
        assert X_train.shape[0] == X_train_meta.shape[0], f"X_train.shape[0] {X_train.shape[0]} =! X_train_meta.shape[0] {X_train_meta.shape[0]}"
        print(X_train.shape, X_val.shape, X_test.shape)
        train_loader, val_loader, test_loader = data_laoder.loaders(X_train, y_train, X_val, y_val, X_test, y_test, X_train_meta, X_val_meta, X_test_meta)

        # 3. Train Transformer alone
        transformer = Transformer(device).to(device)
        train_transformer(transformer, train_loader, val_loader, device, epochs = 150)
        #metrics = test_transformer(transformer, val_loader, device, metrics)

        # 4. Train ANN alone
        ann = ANN().to(device)
        train_ann(ann, device, train_loader, val_loader, epochs = 100)

        # 5. Train Late Fusion Layer alone
        lf = LateFusion(transformer, ann).to(device)
        train_lf(lf, device, train_loader, val_loader, epochs=100)
        
        #metrics = test_lf(lf, train_loader, train_loader2, device, metrics)
        metrics = test_lf(lf, val_loader, device, metrics)

    
    data_laoder.mean_std(metrics)
    

if __name__ == '__main__':
    main()