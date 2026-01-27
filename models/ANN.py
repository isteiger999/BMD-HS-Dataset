import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
def train_ann(ann, device, train_loader, val_loader):
    optimizer = optim.Adam(ann.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion_train = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=5e-6)

    early_st_patience=10
    best_val = -math.inf
    bad_epochs = 0
    best_state = None

    epochs = 100
    for epoch in range(epochs):
        ann.train()
        train_loss = 0
        total_train, correct_train = 0, 0
        for x,y in train_loader:
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

        # LR update
        total_val, correct_val = 0, 0
        ann.eval()
        criterion_val = nn.BCEWithLogitsLoss()
        val_loss = 0

        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = x.to(device), y.to(device)
                pred = ann(xv)
                loss = criterion_val(pred, yv)
                val_loss += loss.item()
                total_val += xv.shape[0]
                y_pred = (torch.sigmoid(preds)>0.5).int()
                for row in range(xv.shape[0]):
                    if torch.equal(y_pred[row, :], y[row, :]):
                        correct_val += 1
        
        val_loss /= total_val
        scheduler.step(val_loss)

        if epoch%10==0: print(f"Epoch {epoch} Train Loss: {train_loss}, train_acc: {correct_train/total_train} || Val Loss: {val_loss}, train_acc: {correct_val/total_val}, lr: {optimizer.param_groups[0]['lr']}")

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