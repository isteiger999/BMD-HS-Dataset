import torch
import torch.nn as nn
import torch.optim as optim
from models.transformer import Transformer
from models.ANN import ANN

n_classes = 5

class LateFusion(nn.Module):
    def __init__(self, nr_windows, win_len):
        super().__init__()
        self.transformer = Transformer(nr_windows, win_len)
        self.ann = ANN()

        self.late_fusion = nn.Linear(2*n_classes, n_classes)

    def forward(self, audio_data, meta_data):
        audio_logits = self.transformer(audio_data)
        meta_logits = self.ann(meta_data)

        concat = torch.concat([audio_logits, meta_logits], dim=1)   # dim 0 = batch, dim 1 = rows, dim = 2 columns
        output = self.late_fusion(concat)
        return torch.sigmoid(output)

   
    
def train_lf(lf, transformer, ann, device, train_loader, val_loader, train_loader2, val_loader2, epochs):
    transformer.eval()
    ann.eval()

    optimizer = optim.Adam(lf.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        lf.train()
        train_loss = 0
        total_train, correct_train = 0, 0
        for (x,y), (xs,_) in zip(train_loader, train_loader2):
            x,y,xs = x.to(device), y.to(device), xs.to(device)
            audio_logits = transformer(x)
            meta_logits = ann(xs)

            preds = lf(audio_logits, meta_logits)
            train_loss += criterion(preds, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_train += x.shape[0]
            y_pred = torch.sigmoid(preds)
            y_pred = (y_pred > 0.5).int()
            for row in range(x.shape[0]):
                correct_train += (y_pred[row,:] == y[row,:]).float().sum().item()

        acc_train = correct_train/total_train

        if epoch%10==0: print(f"Epoch {epoch}: train_acc: {round(acc_train, 3)}")


        



