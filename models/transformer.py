import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy
from models.ANN import ANN
from sklearn.metrics import f1_score


num_channels = 1
num_recordings = 8
embed_dim = 256
num_classes = 5
attention_heads = 4
transformer_blocks = 6
mlp_nodes = 512
"""
patch size: 1 pixel in the spectrogram in horizontal direction corresponds to 50 ms (since we used hop_length=200pixels
when generating the spectrograms, which corresponds at 4kHz sampling rate a hop of 50ms. patch_size of 8 mean 8*50ms,
which is 400ms window --> transformer looks at frequency content of 400 ms)
"""
patch_size = 8  # meaning sees 8 (8x8) pixels horizontally (in time) 1 pixel is 80000/200 = 400ms
nr_tokens = ((401 - patch_size) // patch_size + 1) * ((64 - patch_size) // patch_size + 1)

# Embedding for raw audio timeseries --> very noisy

class EmbeddingSpectograms(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(patch_size * patch_size, embed_dim)

    def forward(self, x):
        # x shape: [B, 8, 1, 128, 401]
        B, R, C, H, W = x.shape
        
        # 1. Flatten B and R into a single batch dimension
        # New shape: [B*8, 1, 128, 401]
        x = x.view(B * R, C, H, W)
        
        # 2. Crop to be divisible by patch_size (400, 128)
        H_new = (H // patch_size) * patch_size
        W_new = (W // patch_size) * patch_size
        x = x[:, :, :H_new, :W_new]
        
        # 3. Extract patches
        # unfold(2, ...) handles H, unfold(3, ...) handles W
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        
        # 4. Reshape to [Total_Batch, nr_tokens, patch_size^2]
        # Total_Batch is B*8
        x = x.contiguous().view(B * R, -1, patch_size * patch_size)
        
        # 5. Project to embed_dim
        x = self.projection(x) # Result: [B*8, Tokens, Embed_Dim]
        
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, attention_heads, batch_first=True) # batch_first=False
        self.dropout = nn.Dropout(p=0.2)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_nodes),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(mlp_nodes, embed_dim)
        )

    def forward(self, x):
        residual1 = x
        x = self.ln1(x)
        x = x.contiguous()
        x = self.dropout(self.multi_head_attention(x, x, x)[0]) + residual1
        residual2 = x
        x = self.ln2(x)
        x = self.mlp(x) + residual2
        return x

class MLP_Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)          # takes in cls token only
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(embed_dim, mlp_nodes)
        self.fc2 = nn.Linear(mlp_nodes, mlp_nodes)
        self.fc3 = nn.Linear(mlp_nodes, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(self.ln1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        #self.embedding = PatchEmbedding(nr_windows)
        self.embedding = EmbeddingSpectograms()
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, nr_tokens+1, embed_dim))
        self.transformer_block = nn.Sequential(*[TransformerEncoder() for _ in range(transformer_blocks)])
        self.mlp_head = MLP_Head()
        self.threshhold = nn.Parameter(torch.Tensor([0.5, 0.5, 0.5, 0.5, 0.5]))
        self.steepness = 1.5
    
    def forward(self, x):
        x = self.embedding(x)
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim = 1)  # dim = 1 is rows (dim=0 is batch, dim = 2 are columns)
        x = x + self.position_embedding
        x = x.contiguous()
        x = self.transformer_block(x)
        x = x[:, 0].contiguous()  # meaning first row (cls token) of every datapoint in the batch
        x = self.mlp_head(x)
        #reshape
        x = x.view(-1,8,5)
        x = torch.mean(x, dim=1)
        
        #return torch.sigmoid(self.steepness * (x - self.threshhold))
        return x



def train_transformer(transformer, train_loader, val_loader, device, epochs = 150):
    optimizer = optim.Adam(transformer.parameters(), lr = 1e-3, weight_decay=1e-5)
    #weights = torch.tensor([0.1795, 0.1545, 0.1748, 0.1748, 0.3161]).to(device)
    weights = torch.tensor([1.9189189189189189, 1.5116279069767442, 1.8421052631578947, 1.8421052631578947, 4.142857142857143]).to(device)
    criterion_train = nn.BCEWithLogitsLoss(pos_weight=weights)
    criterion_val = nn.BCEWithLogitsLoss(pos_weight=weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=5e-6)    # 'min'

    # for early stopping
    early_st_patience = 10
    best_val = -math.inf
    bad_epochs = 0
    best_state = None

    for epoch in range(epochs):
        transformer.train()
        train_loss = 0
        total_train, correct_train = 0, 0

        # preparation for F1 score:
        all_preds, all_labels = [], []
        for x,_,y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = transformer(x)   # preds shape = (B*8, 5)
            loss = criterion_train(preds, y) # loss = criterion_train(preds, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train += (x.shape[0])
            y_pred = (torch.sigmoid(preds) > 0.5).int()
            for row in range(int(x.shape[0])): 
                correct_train += (y_pred[row, :] == y[row, :]).float().mean().item() # bc. y_pred now only has one column
            
            all_preds.append(y_pred.detach().cpu())
            all_labels.append(y.detach().cpu())


        y_pred_mt = torch.cat(all_preds).numpy()
        y_mt = torch.cat(all_labels).numpy()
        f1_train = f1_score(y_mt, y_pred_mt, average='macro')
        train_loss /= total_train
        train_acc = correct_train / total_train

        # lr scheduler
        transformer.eval()
        total_val, correct_val = 0, 0
        val_loss = 0
        
        # preparation for F1 score calculation
        all_predsv, all_labelsv = [], []
        with torch.no_grad():
            for xv, _, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = transformer(xv)            
                lossv = criterion_val(pred, yv)   #lossv = criterion_val(pred, yv)
                val_loss += lossv.item()
                y_predv = (torch.sigmoid(pred) > 0.5).int() ##

                total_val += (xv.shape[0])
                for row in range(int(xv.shape[0])): 
                    correct_val += (y_predv[row, :] == yv[row, :]).float().mean().item() # bc. y_pred now only has one column

                all_predsv.append(y_predv.detach().cpu())
                all_labelsv.append(yv.detach().cpu())
                
        y_pred_mv = torch.cat(all_predsv).numpy()
        y_mv = torch.cat(all_labelsv).numpy()
        f1_val = f1_score(y_mv, y_pred_mv, average='macro')                        
        val_loss /= total_val
        val_acc = correct_val / total_val
        scheduler.step(val_loss)

        # early stopping
        if f1_val > best_val: # val_loss <
            best_val = f1_val # val_loss
            bad_epochs = 0
            best_state = copy.deepcopy(transformer.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= early_st_patience:
                if best_state is not None:
                    transformer.load_state_dict(best_state)
                break

        print(f"Trans Epoch {epoch}: train_acc: {round(train_acc, 2)} || f1_train: {round(f1_train, 2)} || train_loss: {round(train_loss, 2)} || val_acc: {round(val_acc, 2)} || f1_val: {round(f1_val, 2)} || val_loss: {round(val_loss, 2)}, lr: {optimizer.param_groups[0]['lr']:.6f}")


