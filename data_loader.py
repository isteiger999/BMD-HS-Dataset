import torch
import pandas as pd
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset

def number(df_x):
    map_gender = {'M': 0, 'F': 1} 
    map_home = {'U': 0, 'R': 1}

    df_x = df_x.copy()

    df_x.iloc[:, 1] = df_x.iloc[:, 1].map(map_gender)
    df_x.iloc[:, 3] = df_x.iloc[:, 3].map(map_home)

    df_x = df_x.apply(pd.to_numeric, errors='coerce')
        
    return df_x

def load_data_simple():
    df_x = pd.read_csv(r"data/additional_metadata.csv")
    df_y = pd.read_csv(r"data/train.csv")

    df_x = df_x.drop(['patient_id'], axis=1)
    df_y = df_y[['AS','AR','MR','MS','N']]

    df_x = number(df_x)

    X, y = torch.tensor(np.array(df_x), dtype=torch.float32), torch.tensor(np.array(df_y), dtype=torch.float32) 
    
    return X, y

def calc_fraction(X, stride, split):
    fract_train = round(X.shape[0] * split[0])
    residue = X.shape[0] - fract_train
    stride = math.floor(X.shape[0] * stride)
    if residue%2 == 0:
        fract_val, fract_test = residue/2, residue/2
        return int(fract_train), int(fract_val), int(fract_test), int(stride)
    else:
        fract_train += 1   
        residue = X.shape[0] - fract_train
        fract_val, fract_test = residue/2, residue/2
        return int(fract_train), int(fract_val), int(fract_test), int(stride)
    
def split_data(X, y, iteration, stride, split):
    X2 = torch.concat([X, X], dim=0)
    y2 = torch.concat([y, y], dim=0) 

    fract_train, fract_val, fract_test, stride  = calc_fraction(X, stride, split)
    X_train, y_train = X2[iteration*stride:(fract_train+iteration*stride), :], y2[iteration*stride:(fract_train+iteration*stride), :]
    X_val, y_val = X2[(fract_train+iteration*stride):(fract_train+iteration*stride+fract_val), :], y2[(fract_train+iteration*stride):(fract_train+iteration*stride+fract_val), :]
    X_test, y_test = X2[(fract_train+iteration*stride+fract_val):(fract_train+iteration*stride+fract_val+fract_test), :], y2[(fract_train+iteration*stride+fract_val):(fract_train+iteration*stride+fract_val+fract_test), :]

    return X_train, y_train, X_val, y_val, X_test, y_test

def loaders(X_train, y_train, X_val, y_val, X_test, y_test):
    train_ds = TensorDataset(X_train.float(), y_train.squeeze())
    val_ds = TensorDataset(X_val.float(), y_val.squeeze())
    test_ds = TensorDataset(X_test.float(), y_test.squeeze())
    
    train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=10, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=10, shuffle=False)

    return train_loader, val_loader, test_loader

def calc_mean_std(list):
    # Calculate mean
    length = len(list)
    mean = sum(list) / length

    # Calculate variance
    squared_diffs = [(x - mean) ** 2 for x in list]
    variance = sum(squared_diffs) / length
    std = np.sqrt(variance)

    return mean, std

def mean_std(metrics):
    list_loss = metrics["Final_loss"]
    list_acc = metrics["Acc"]

    mu_loss, std_loss = calc_mean_std(list_loss)
    mu_acc, std_acc = calc_mean_std(list_acc)

    print(f"Final Loss: {mu_loss}\u00B1{std_loss}")
    print(f"Final Acc: {mu_acc}\u00B1{std_acc}")