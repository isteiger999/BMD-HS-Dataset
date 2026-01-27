import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    criterion = nn.BCEWithLogitsLoss()

    epochs = 100
    for epoch in range(epochs):
        ann.train()
        train_loss = 0
        total, correct = 0, 0
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = ann(x)
            loss = criterion(preds, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += x.shape[0]
            probs = torch.sigmoid(preds)
            y_pred = (probs > 0.5).int()
            for row in range(x.shape[0]):
                if torch.equal(y_pred[row, :], y[row, :]):
                    correct += 1
        
        train_loss /= total
        if epoch%10 == 0: print(f"Train Loss: {train_loss}, accuracy: {correct/total}")

