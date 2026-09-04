import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy
import torch.nn.functional as F
from sklearn.metrics import f1_score

n_classes = 5

class LateFusion(nn.Module):
    def __init__(self, transformer, ann):
        super().__init__()
        self.transformer = transformer
        self.ann = ann

        #self.lf1 = nn.Linear(2*n_classes, n_classes)
        self.mlp = nn.Sequential(
            nn.Linear(2*n_classes, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, audio_data, meta_data):
        audio_logits = self.transformer(audio_data)
        meta_logits = self.ann(meta_data)

        concat = torch.concat([audio_logits, meta_logits], dim=1)   # dim 0 = batch, dim 1 = rows, dim = 2 columns
        output = self.mlp(concat)
        return output

   
    
def train_lf(lf, device, train_loader, val_loader, epochs):
    lf.transformer.eval()
    lf.ann.eval()

    optimizer = optim.Adam(lf.parameters(), lr = 5e-4, weight_decay=1e-5)
    weights = torch.tensor([1.9189189189189189, 1.5116279069767442, 1.8421052631578947, 1.8421052631578947, 4.142857142857143]).to(device)
    criterion_train = nn.BCEWithLogitsLoss(pos_weight=weights)
    criterion_val = nn.BCEWithLogitsLoss(pos_weight=weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor = 0.5, min_lr=5e-6)

    early_st_patience = 10
    best_val = -math.inf
    bad_epochs = 0
    best_state = None

    for epoch in range(epochs):
        lf.train()
        train_loss = 0
        total_train, correct_train = 0, 0

        all_preds, all_labels = [], []
        for x, xs, y in train_loader:
            x,y,xs = x.to(device), y.to(device), xs.to(device)
            preds = lf(x, xs)
            # NO AVERAGING NEEDED IN LATE FUSION
            loss = criterion_train(preds, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            total_train += x.shape[0]
            y_pred = (torch.sigmoid(preds) > 0.5).int()
            for row in range(int(x.shape[0])): 
                    correct_train += (y_pred[row, :] == y[row, :]).float().mean().item()
            all_labels.append(y.detach().cpu())
            all_preds.append(y_pred.detach().cpu())

        y_pred_mt = torch.cat(all_preds).numpy()
        y_mt = torch.cat(all_labels).numpy()
        f1_train = f1_score(y_mt, y_pred_mt, average='macro')
        acc_train = correct_train/total_train
        train_loss /= total_train

        lf.eval()
        val_loss = 0
        correct_val, total_val = 0, 0

        all_predsv, all_labelsv = [], []
        with torch.no_grad():
            for xv,xsv,yv in val_loader:
                xv, xsv, yv = xv.to(device), xsv.to(device), yv.to(device)
                pred = lf(xv, xsv)
                # NO AVERAGING NEEDED IN LATE FUSION
                lossv = criterion_val(pred, yv) 
                val_loss += lossv.item()

                y_predv = (torch.sigmoid(pred) > 0.5).int()
                total_val += xv.shape[0]
                for row in range(int(xv.shape[0])): 
                    correct_val += (y_predv[row, :] == yv[row, :]).float().mean().item() # bc. y_pred now only has one column

                all_predsv.append(y_predv.detach().cpu())
                all_labelsv.append(yv.detach().cpu())
        
        y_pred_mv = torch.cat(all_predsv).numpy()
        y_mv = torch.cat(all_labelsv).numpy()
        f1_val = f1_score(y_mv, y_pred_mv, average='macro')
        acc_val = correct_val/total_val
        val_loss /= total_val

        scheduler.step(val_loss)

        #early stopping
        if f1_val > best_val:
            best_val = f1_val
            bad_epochs = 0
            best_state = copy.deepcopy(lf.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= early_st_patience:
                if best_state is not None:
                    lf.load_state_dict(best_state)
                break

        print(f"Epoch {epoch}: train_acc: {round(acc_train, 2)} || f1_train: {round(f1_train, 2)} || train_loss: {round(train_loss, 3)} || val_acc: {round(acc_val, 2)} || f1_val: {round(f1_val, 2)} || val_loss: {round(val_loss, 3)}, lr: {optimizer.param_groups[0]['lr']:.6f}")



def test_lf(lf, train_loader, val_loader, device, metrics):
    lf.eval()
    weights = torch.tensor([1.9189189189189189, 1.5116279069767442, 1.8421052631578947, 1.8421052631578947, 4.142857142857143]).to(device)
    criterion_val = nn.BCEWithLogitsLoss(pos_weight=weights)
    total_val, correct_val = 0, 0
    val_loss = 0

    all_predsv, all_labelsv = [], []
    with torch.no_grad():
        for xv,xsv,yv in val_loader:
            xv, xsv, yv = xv.to(device), xsv.to(device), yv.to(device)
            pred = lf(xv, xsv)
            # NO AVERAGING NEEDED IN LATE FUSION
            lossv = criterion_val(pred, yv) 
            val_loss += lossv.item()

            y_predv = (torch.sigmoid(pred) > 0.5).int()
            total_val += xv.shape[0]
            for row in range(int(xv.shape[0])): 
                correct_val += (y_predv[row, :] == yv[row, :]).float().mean().item() # bc. y_pred now only has one column

            all_predsv.append(y_predv.detach().cpu())
            all_labelsv.append(yv.detach().cpu())
        
    y_pred_mv = torch.cat(all_predsv).numpy()
    y_mv = torch.cat(all_labelsv).numpy()
    f1_val = f1_score(y_mv, y_pred_mv, average='macro')
    val_acc = correct_val/total_val
    val_loss /= total_val

    metrics["Final_loss_val"].append(val_loss)
    metrics["Acc_val"].append(val_acc)
    metrics["F1_val"].append(f1_val)


    # Now do same for train_loader
    criterion_train = nn.BCEWithLogitsLoss(pos_weight=weights)
    total_train, correct_train = 0, 0
    train_loss = 0
    all_predst, all_labelst = [], []
    with torch.no_grad():
        for xv,xsv,yv in train_loader:
            xv, xsv, yv = xv.to(device), xsv.to(device), yv.to(device)
            pred = lf(xv, xsv)
            # NO AVERAGING NEEDED IN LATE FUSION
            lossv = criterion_train(pred, yv) 
            train_loss += lossv.item()

            y_predv = (torch.sigmoid(pred) > 0.5).int()
            total_train += xv.shape[0]
            for row in range(int(xv.shape[0])): 
                correct_train += (y_predv[row, :] == yv[row, :]).float().mean().item() # bc. y_pred now only has one column

            all_predst.append(y_predv.detach().cpu())
            all_labelst.append(yv.detach().cpu())
        
    y_pred_mv = torch.cat(all_predst).numpy()
    y_mv = torch.cat(all_labelst).numpy()
    f1_train = f1_score(y_mv, y_pred_mv, average='macro')
    train_acc = correct_train/total_train
    train_loss /= total_train

    metrics["Final_loss_train"].append(train_loss)
    metrics["Acc_train"].append(train_acc)
    metrics["F1_train"].append(f1_train)

    return metrics





