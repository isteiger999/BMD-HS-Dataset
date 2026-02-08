import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy
from models.ANN import ANN


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
nr_tokens = ((401 - patch_size) // patch_size + 1) * ((128 - patch_size) // patch_size + 1)

# Embedding for raw audio timeseries --> very noisy

class EmbeddingSpectograms(nn.Module):
    def __init__(self):
        super().__init__()
        # Each patch is (patch_size x patch_size). 
        # For a grayscale spectrogram (1 channel), the input features are patch_size^2.
        self.projection = nn.Linear(patch_size * patch_size, embed_dim)

    def forward(self, x):
        # x shape: [B, 1, 128, 401]
        B, C, H, W = x.shape
        
        # 1. Crop the spectrogram so it's perfectly divisible by patch_size
        # (401 becomes 400, 128 is already divisible by 8)
        H_new = (H // patch_size) * patch_size
        W_new = (W // patch_size) * patch_size
        x = x[:, :, :H_new, :W_new]
        
        # 2. Use unfold to extract patches manually
        # This creates a tensor of patches: [B, 1, H/p, W/p, p, p]
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        
        # 3. Reshape and flatten the patches
        # We want: [Batch, Number_of_Patches, Patch_Pixels]
        x = x.contiguous().view(B, -1, patch_size * patch_size)
        
        # 4. Project pixels to embed_dim
        x = self.projection(x) # Result: [B, Tokens, Embed_Dim]
        
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
        return x


def train_transformer(transformer, train_loader, val_loader, device, epochs = 150):
    optimizer = optim.Adam(transformer.parameters(), lr = 5e-4, weight_decay=1e-3)
    weights = torch.tensor([0.1795, 0.1545, 0.1748, 0.1748, 0.3161]).to(device)
    criterion_train = nn.BCEWithLogitsLoss(weight=weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=5e-6)

    # for early stopping
    early_st_patience = 10
    best_val = math.inf
    bad_epochs = 0
    best_state = None

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
                correct_train += (y_pred[row, :] == y[row, :]).float().mean().item()


        train_loss /= total_train
        train_acc = correct_train / total_train

        # lr scheduler
        transformer.eval()
        criterion_val = nn.BCEWithLogitsLoss()
        total_val, correct_val = 0, 0
        val_loss = 0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                pred = transformer(xv)
                lossv = criterion_val(pred, yv)
                val_loss += lossv.item()
                y_pred = torch.sigmoid(pred)
                y_pred = (y_pred > 0.5).int()
                
                total_val += xv.shape[0]
                for row in range(xv.shape[0]):
                    correct_val += (y_pred[row, :] == yv[row, :]).float().mean().item()
                    
        val_loss /= total_val
        val_acc = correct_val / total_val
        scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            best_state = copy.deepcopy(transformer.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= early_st_patience:
                if best_state is not None:
                    transformer.load_state_dict(best_state)
                break

        print(f"Trans Epoch {epoch}: train_acc: {round(train_acc, 2)} || train_loss: {round(train_loss, 3)} || val_acc: {round(val_acc, 2)} || val_loss: {round(val_loss, 3)}, lr: {optimizer.param_groups[0]['lr']:.6f}")

def test_transformer(transformer, val_loader, device, metrics):
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
                correct_val += (y_pred[row,:]==yv[row,:]).float().mean().item()
                
    val_loss /= total_val
    val_acc = correct_val / total_val

    metrics["Final_loss"].append(val_loss)
    metrics["Acc"].append(val_acc)

    return metrics


