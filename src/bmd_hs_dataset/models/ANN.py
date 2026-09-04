import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy
from sklearn.metrics import f1_score

n_classes = 5

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64,n_classes)
        )
    
    def forward(self, x):
        meta_logits = self.mlp(x)
        return meta_logits 
    
def train_ann(ann, device, train_loader, val_loader, epochs):
    optimizer = optim.Adam(ann.parameters(), lr = 1e-3, weight_decay=1e-5)
    weights = torch.tensor([1.9189189189189189, 1.5116279069767442, 1.8421052631578947, 1.8421052631578947, 4.142857142857143]).to(device)
    criterion_train = nn.BCEWithLogitsLoss(pos_weight=weights)
    criterion_val = nn.BCEWithLogitsLoss(pos_weight=weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=5e-6)

    early_st_patience=10
    best_val = -math.inf
    bad_epochs = 0
    best_state = None

    for epoch in range(epochs):
        ann.train()
        train_loss = 0
        total_train, correct_train = 0, 0

        # preparation for F1 score:
        all_preds, all_labels = [], []
        for _,x,y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = ann(x)
            loss = criterion_train(preds, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train += x.shape[0]
            #probs = torch.sigmoid(preds)
            y_pred = (torch.sigmoid(preds) > 0.5).int()
            for row in range(x.shape[0]):
                correct_train += (y_pred[row, :] == y[row, :]).float().mean().item()
            
            all_preds.append(y_pred.detach().cpu())
            all_labels.append(y.detach().cpu())

        y_pred_mt = torch.cat(all_preds).numpy()
        y_mt = torch.cat(all_labels).numpy()
        f1_train = f1_score(y_mt, y_pred_mt, average='macro')
        train_loss /= total_train
        acc_train = correct_train/total_train

        # LR update
        total_val, correct_val = 0, 0
        ann.eval()
        val_loss = 0

        # preparation for F1 score:
        all_predsv, all_labelsv = [], [] 
        with torch.no_grad():
            for _,xv,yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = ann(xv)
                loss = criterion_val(pred, yv)
                val_loss += loss.item()

                y_pred = (torch.sigmoid(pred) > 0.5).int()
                total_val += xv.shape[0]
                for row in range(xv.shape[0]):
                    correct_val += (y_pred[row, :] == yv[row, :]).float().mean().item()
                all_predsv.append(y_pred.detach().cpu())
                all_labelsv.append(yv.detach().cpu())

        y_pred_mv = torch.cat(all_predsv).numpy()
        y_mv = torch.cat(all_labelsv).numpy()
        f1_val = f1_score(y_mv, y_pred_mv, average='macro')
        val_loss /= total_val
        acc_val = correct_val/total_val
        scheduler.step(val_loss)

        # early stopping
        if f1_val > best_val:
            best_val = f1_val
            bad_epochs = 0
            best_state = copy.deepcopy(ann.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= early_st_patience:
                if best_state is not None:
                    ann.load_state_dict(best_state)
                break

        if epoch%10==0: print(f"ANN Epoch {epoch}: train_acc: {round(acc_train, 2)} || f1_train: {round(f1_train, 2)} || train_loss: {round(train_loss, 3)} || val_acc: {round(acc_val, 2)} || f1_val: {round(f1_val, 2)} || val_loss: {round(val_loss, 3)}, lr: {optimizer.param_groups[0]['lr']:.6f}")

