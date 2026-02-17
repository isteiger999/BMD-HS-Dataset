import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy


n_classes = 5

class LateFusion(nn.Module):
    def __init__(self, transformer, ann):
        super().__init__()
        self.transformer = transformer
        self.ann = ann

        self.late_fusion = nn.Linear(2*n_classes, n_classes)

    def forward(self, audio_data, meta_data):
        audio_logits = self.transformer(audio_data)
        meta_logits = self.ann(meta_data)

        concat = torch.concat([audio_logits, meta_logits], dim=1)   # dim 0 = batch, dim 1 = rows, dim = 2 columns
        output = self.late_fusion(concat)
        return output

   
    
def train_lf(lf, device, train_loader, val_loader, epochs):
    lf.transformer.eval()
    lf.ann.eval()

    optimizer = optim.Adam(lf.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion_train = nn.BCEWithLogitsLoss()
    criterion_val = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor = 0.5, min_lr=5e-6)

    early_st_patience = 8
    best_val = math.inf
    bad_epochs = 0
    best_state = None

    for epoch in range(epochs):
        lf.train()
        train_loss = 0
        total_train, correct_train = 0, 0
        #for (x,y), (xs,_) in zip(train_loader, train_loader2):
        for x, xs, y in train_loader:
            x,y,xs = x.to(device), y.to(device), xs.to(device)
            preds = lf(x, xs)
            loss = criterion_train(preds, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train += x.shape[0]
            y_pred = torch.sigmoid(preds)
            y_pred = (y_pred > 0.5).int()
            for row in range(x.shape[0]):
                correct_train += (y_pred[row,:] == y[row,:]).float().mean().item()

        acc_train = correct_train/total_train
        train_loss /= total_train

        lf.eval()
        val_loss = 0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            #for (xv,yv), (xsv,_) in zip(val_loader, val_loader2):
            for xv,xsv,yv in val_loader:
                xv, xsv, yv = xv.to(device), xsv.to(device), yv.to(device)
                preds = lf(xv, xsv)
                loss = criterion_val(preds, yv)
                val_loss += loss.item()

                y_pred = (preds > 0.5).int()
                total_val += xv.shape[0]
                for row in range(xv.shape[0]):
                    correct_val += (y_pred[row, :] == yv[row, :]).float().mean().item()
        
        acc_val = correct_val/total_val
        val_loss /= total_val

        scheduler.step(val_loss)

        #early stopping
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            best_state = copy.deepcopy(lf.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= early_st_patience:
                if best_state is not None:
                    lf.load_state_dict(best_state)
                break

        print(f"Epoch {epoch}: train_acc: {round(acc_train, 2)} || train_loss: {round(train_loss, 3)} || val_acc: {round(acc_val, 2)} || val_loss: {round(val_loss, 3)}, lr: {optimizer.param_groups[0]['lr']:.6f}")



def test_lf(lf, val_loader, device, metrics):
    lf.eval()
    criterion_val = nn.BCEWithLogitsLoss()
    total_val, correct_val = 0, 0
    val_loss = 0

    with torch.no_grad():
        for xv, xsv, yv in val_loader:
            xv, yv, xsv = xv.to(device), yv.to(device), xsv.to(device)
            pred = lf(xv, xsv)
            loss = criterion_val(pred, yv)
            val_loss += loss.item()
            y_pred = torch.sigmoid(pred)
            y_pred = (y_pred > 0.5).int()
            
            total_val += xv.shape[0]
            for row in range(xv.shape[0]):
                correct_val += (y_pred[row, :] == yv[row, :]).float().mean().item() # y_pred.shape = 10,5
                
    val_loss /= total_val
    val_acc = correct_val / total_val

    metrics["Final_loss"].append(val_loss)
    metrics["Acc"].append(val_acc)

    return metrics





