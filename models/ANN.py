import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy


n_classes = 5

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128,n_classes)
        )
    
    def forward(self, x):
        meta_logits = self.mlp(x)
        return meta_logits
    
def train_ann(ann, device, train_loader, val_loader, epochs):
    optimizer = optim.Adam(ann.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion_train = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=5e-6)

    early_st_patience=8
    best_val = math.inf
    bad_epochs = 0
    best_state = None

    for epoch in range(epochs):
        ann.train()
        train_loss = 0
        total_train, correct_train = 0, 0
        for x,_,y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = ann(x)
            loss = criterion_train(preds, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train += x.shape[0]
            probs = torch.sigmoid(preds)
            y_pred = (probs > 0.5).int()
            for row in range(x.shape[0]):
                if torch.equal(y_pred[row, :], y[row, :]):
                    correct_train += 1
        
        train_loss /= total_train
        acc_train = correct_train/total_train

        # LR update
        total_val, correct_val = 0, 0
        ann.eval()
        criterion_val = nn.BCEWithLogitsLoss()
        val_loss = 0

        with torch.no_grad():
            for xv,_,yv in val_loader:
                xv, yv = x.to(device), y.to(device)
                pred = ann(xv)
                loss = criterion_val(pred, yv)
                val_loss += loss.item()

                y_pred = (torch.sigmoid(preds)>0.5).int()
                total_val += xv.shape[0]
                for row in range(xv.shape[0]):
                    correct_val += (y_pred[row, :] == yv[row, :]).float().mean().item()
        
        val_loss /= total_val
        acc_val = correct_val/total_val
        scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            best_state = copy.deepcopy(ann.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= early_st_patience:
                if best_state is not None:
                    ann.load_state_dict(best_state)
                break

        if epoch%10==0: print(f"ANN Epoch {epoch}: train_acc: {round(acc_train, 2)} || train_loss: {round(train_loss, 3)} || val_acc: {round(acc_val, 2)} || val_loss: {round(val_loss, 3)}, lr: {optimizer.param_groups[0]['lr']:.6f}")


def test_ann(ann, device, val_loader, test_loader, metrics, mode):
    
    if mode=="val":
        loader = val_loader 
    else:
        loader = test_loader

    criterion = nn.BCEWithLogitsLoss()

    test_loss = 0
    total_test, correct_test = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = ann(x)
        y_pred = (torch.sigmoid(preds)>0.5).int()
        test_loss += criterion(preds, y).item()
        total_test += x.shape[0]
        for row in range(x.shape[0]):
            if torch.equal(y_pred[row, :], y[row, :]):
                correct_test += 1

    test_loss /= total_test
    acc = correct_test/total_test
    metrics["Final_loss"].append(test_loss)
    metrics["Acc"].append(acc)

    return metrics