import torch
import pandas as pd
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
import librosa
import matplotlib.pyplot as plt
from scipy import signal

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
    
    # X.shape = 108, 4
    # y.shape = 108, 5

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


# ----------------------------

def fix_length(wav_file):
    if wav_file.shape[0] < 80000:
        diff = 80000-wav_file.shape[0]
        wav_file = np.pad(wav_file, (0, diff), mode='edge')
    else: 
        wav_file = wav_file[:80000]
    
    return wav_file

def filter(wav_file):
    b, a = signal.butter(5, 250, 'low', analog = False, fs=4000) #first parameter is signal order and the second one refers to frequenc limit. I set limit 30 so that I can see only below 30 frequency signal component
    output = signal.filtfilt(b, a, wav_file)
    output_copy = output.copy()
    output = torch.tensor(output_copy)
    return output

def load_pcg_data(device):
    X = torch.zeros([108, 8, 80000], dtype=torch.float32, device=device)
    y = torch.zeros([108, 5], dtype=torch.float32, device=device)

    train_csv = pd.read_csv('data/train.csv')
    train = train_csv.to_numpy()

    for row in range(train_csv.shape[0]):
        for index, file_name in enumerate(train_csv.iloc[row, 6:]):
            wav_file, _ = librosa.load(f'data/train/{file_name}.wav', sr=4000)
            if wav_file.shape[0] != 80000:
                wav_file = fix_length(wav_file)

            #wav_file_filtered = filter(wav_file)    
            X[row, index, :] = wav_file
        
        labels = train[row, 1:6]
        labels = torch.tensor(labels.astype(float), dtype=torch.float32)
        labels = labels.view(1, -1)
        y[row, :] = labels

    return X, y