import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy

num_channels = 8
embed_dim = 100
kernel_size = 1000 # 4000 = 1sec
stride = kernel_size//4
num_classes = 5
attention_heads = 4
transformer_blocks = 4
mlp_nodes = 256
nr_tokens = (80000 - kernel_size) // stride + 1

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_channels=num_channels, out_channels=embed_dim, kernel_size=kernel_size, stride = stride)

    def forward(self, x):
        embedding = self.patch_embed(x)             # outputs [batch, embed_dim, #tokens]
        embedding = embedding.transpose(1,2)        # transformer expects [batch, #tokens, embed_dim]
        return embedding
    
class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, attention_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_nodes),
            nn.GELU(),
            nn.Linear(mlp_nodes, embed_dim)
        )

    def forward(self, x):
        residual1 = x
        x = self.ln1(x)
        x = self.multi_head_attention(x, x, x)[0] + residual1
        residual2 = x
        x = self.ln2(x)
        x = self.mlp(x) + residual2
        return x

class MLP_Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)          # takes in cls token only
        self.fc1 = nn.Linear(embed_dim, mlp_nodes)
        self.fc2 = nn.Linear(mlp_nodes, mlp_nodes)
        self.fc3 = nn.Linear(mlp_nodes, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(self.ln1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, nr_tokens+1, embed_dim))
        self.transformer_block = nn.Sequential(*[TransformerEncoder() for _ in range(transformer_blocks)])
        self.mlp_head = MLP_Head()

    def forward(self, x):
        x = self.embedding(x)
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim = 1)  # dim = 1 is rows (dim=0 is batch, dim = 2 are columns)
        x = x + self.position_embedding
        x = self.transformer_block(x)
        x = x[:, 0]
        x = self.mlp_head(x)
        return x


def train_transformer(transformer, train_loader, val_loader, device, epochs = 150):
    optimizer = optim.Adam(transformer.parameters(), lr = 5e-4, weight_decay=1e-4)
    criterion_train = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=5e-6)

    for epoch in range(epochs):
        transformer.train()
        train_loss = 0
        total_train, correct_train = 0, 0
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = transformer(x)
            loss = criterion_train(preds, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train += x.shape[0]
            y_pred = torch.sigmoid(preds)
            y_pred = (y_pred > 0.5).int()
            for row in range(x.shape[0]):
                if torch.equal(y_pred[row, :], y[row, :]):
                    correct_train += 1


        train_loss /= total_train
        train_acc = correct_train / total_train

        transformer.eval()
        criterion_val = nn.BCEWithLogitsLoss()
        total_val, correct_val = 0, 0
        val_loss = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = transformer(xv)
                loss = criterion_val(pred, yv)
                val_loss += loss.item()
                y_pred = torch.sigmoid(pred)
                y_pred = (y_pred > 0.5).int()
                
                total_val += xv.shape[0]
                for row in range(xv.shape[0]):
                    if torch.equal(y_pred[row, :], yv[row, :]):
                        correct_val += 1
                    
        val_loss /= total_val
        val_acc = correct_val / total_val
        scheduler.step(val_loss)

        # early stopping
        early_st_patience = 8
        best_val = -math.inf
        bad_epochs = 0
        best_state = None
        if val_acc > best_val:
            best_val = val_acc
            bad_epochs = 0
            best_state = copy.deepcopy(transformer.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= early_st_patience:
                if best_state is not None:
                    transformer.load_state_dict(best_state)
                break

        if epoch%10==0: print(f"Epoch {epoch}: train_acc: {train_acc} || train_loss: {train_loss} || val_acc: {val_acc} || val_loss: {val_loss}, lr: {optimizer.param_groups[0]['lr']:.6f}")