import torch
import pandas as pd
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy import signal
import random

def ensure_deterministic():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

def number(df_x):
    map_gender = {'M': 0, 'F': 1} 
    map_home = {'U': 0, 'R': 1}

    df_x = df_x.copy()
    # Convert Gender strings to integers (M=0, F=1) and Home strings to integers (U=0, R=1)
    df_x['Gender'] = df_x['Gender'].map(map_gender)
    df_x['Lives'] = df_x['Lives'].map(map_home)

    df_x = df_x.apply(pd.to_numeric, errors='coerce')
        
    return df_x

def loaders(X_train, y_train, X_val, y_val, X_test, y_test, Xs_train, Xs_val, Xs_test):
    train_ds = TensorDataset(X_train.float(), Xs_train.float(), y_train.squeeze())
    val_ds = TensorDataset(X_val.float(), Xs_val.float(), y_val.squeeze())
    test_ds = TensorDataset(X_test.float(), Xs_test.float(), y_test.squeeze())
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True) # shuffle = True
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

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
    list_f1 = metrics["F1"]

    mu_loss, std_loss = calc_mean_std(list_loss)
    mu_acc, std_acc = calc_mean_std(list_acc)
    mu_f1, std_f1 = calc_mean_std(list_f1)

    print(f"Final Loss: {mu_loss}\u00B1{std_loss}")
    print(f"Final Acc: {mu_acc}\u00B1{std_acc}")
    print(f"Final F1: {mu_f1}\u00B1{std_f1}")


def fix_length(wav_file):
    if wav_file.shape[0] < 80000:
        diff = 80000-wav_file.shape[0]
        rest = wav_file[:diff]
        wav_file = torch.concat([wav_file, rest], dim=0)
    else: 
        wav_file = wav_file[:80000]
    
    return wav_file

def filter(wav_file):
    b, a = signal.butter(5, 250, 'low', analog = False, fs=4000) #first parameter is signal order and the second one refers to frequenc limit. I set limit 30 so that I can see only below 30 frequency signal component
    output = signal.filtfilt(b, a, wav_file)
    output_copy = output.copy()
    output = torch.tensor(output_copy)
    return output




####### COMPUTER VISION APPROACH USING SPECTROGRAMS ###########

def create_order() -> tuple[dict, dict, dict]:
    '''
    Returns three dictionaries: order, labels, metadata.
    order: keys are patient names, values are lists of the 8 recording names (strings)
    labels: keys are patient names, values are lists of the 5 disease labels (integers)
    metadata: keys are patient names, values are lists of the 4 metadata labels (integers)
    '''
    order, labels, metadata = {}, {}, {}
    train_csv = pd.read_csv(f"src/bmd_hs_dataset/data/train.csv")
    meta_csv = number(pd.read_csv(f"src/bmd_hs_dataset/data/additional_metadata.csv")) # number is function which converts characters to 0/1

    patients = train_csv["patient_id"].tolist()
    rec_names = train_csv.drop(["AS", "AR", "MR", "MS" ,"N"], axis=1)
    labels_dict = train_csv[["patient_id", "AS", "AR", "MR", "MS", "N"]]
    # 1. Add all patients to order dictionary as keys
    for patient in patients:
        order[patient] = []
        labels[patient] = []
        metadata[patient] = []
    
    # 2. Add all 8 recording names as values to each patient
    for row, patient in enumerate(patients):
        for col in range(8):
            rec_name = rec_names.iloc[row, col+1] # col+1 because first column is patient name
            order[patient].append(rec_name)
            if col <= 4:        # because there are only 5 labels but 8 recs
                disease = labels_dict.iloc[row, col+1]
                labels[patient].append(int(disease))
            if col <= 3:
                meta = meta_csv.iloc[row, col+1]
                metadata[patient].append(meta)

    # mix up patients
    keys = list(order.keys())
    random.shuffle(keys)

    order = {k: order[k] for k in keys}
    labels = {k: labels[k] for k in keys}
    metadata = {k: metadata[k] for k in keys}

    return order, labels, metadata

def calc_fractions(patients: list[str], stride: float, split: list) -> tuple[int, int, int, int]:
    patients_tot = int(len(patients)) # 108
    train_fract = int(split[0] * patients_tot - 0.5)
    stride = math.floor(stride * patients_tot)

    if train_fract % 2 == 0:
        residue = patients_tot - train_fract
        val_fract, test_fract = residue // 2, residue // 2
        return train_fract, val_fract, test_fract, stride
    else:
        train_fract += 1
        residue = patients_tot - train_fract
        val_fract, test_fract = residue // 2, residue // 2
        return train_fract, val_fract, test_fract, stride

def plot_spectrogram(spectogram_tensor, patient):
    spec_np = spectogram_tensor.cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram: {patient}')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    plt.show()

def create_tensors_mel(pX_train: list[str], device: torch.device, 
                       order: dict[str, list[str]], 
                       labels: dict[str, list[int]])-> tuple[torch.Tensor, torch.Tensor]:
    nr_recording_per_patient = 8
    spec = torch.load('src/bmd_hs_dataset/data/spectrograms/AR_016_sit_Aor.pt', weights_only=True)
    height = spec.shape[0]              # 128
    width = spec.shape[1]               # 401

    X_train = torch.zeros([len(pX_train)*nr_recording_per_patient, 1, height, width], dtype=torch.float32, device=device)
    y_train = torch.zeros([len(pX_train)*nr_recording_per_patient, 5], dtype=torch.float32, device=device)
    for num, patient in enumerate(pX_train):
        rec8 = order[patient]           # a list of the 8 recording names (strings)
        disease5 = labels[patient]      # a list of 4 integers (diseases)
        for idx, rec in enumerate(rec8):
            spectogram_tensor = torch.load(f"src/bmd_hs_dataset/data/spectrograms/{rec}.pt", weights_only=True)
            X_train[num+idx, 0] = spectogram_tensor
            y_train[num+idx, :] = torch.Tensor(np.array(disease5))
            #plot_spectrogram(spectogram_tensor, patient)

    return X_train, y_train

def create_tensors_meta(pX_train, device, metadata):
    recording_per_patient = 8
    X_train = torch.zeros([len(pX_train)*recording_per_patient, 4], dtype=torch.float32, device=device)

    for num, patient in enumerate(pX_train):
        meta4 = metadata[patient]           # a list of the 4 integers (metadata)
        X_train[num:num+recording_per_patient] = torch.Tensor(np.array(meta4))


    return X_train

def time_freq_masking(X:torch.Tensor, y:torch.Tensor) -> tuple[torch.Tensor, torch.tensor]:
    X_copy1 = X.clone()
    X_copy2 = X.clone()

    B,_,H,W = X.shape
    nr_stripes_t = 5
    stripe_width_t_min = 5
    stripe_width_t_max = 10

    # Time Masking
    nr_stripes_t = 5
    stripe_width_t_min = 5
    stripe_width_t_max = 10
    for mel_nr in range(B):
        mel_spec = X_copy1[mel_nr, 0]
        for _ in range(nr_stripes_t):
            stripe_width = random.randint(stripe_width_t_min, stripe_width_t_max)
            stripe_start_idx = random.randint(5, W - stripe_width_t_max - 1)

            mel_spec[:, stripe_start_idx:(stripe_start_idx+stripe_width)] = 0.0

        X_copy1[mel_nr, 0] = mel_spec

    # Frequency Masking
    nr_stripes_f = 3
    stripe_width_f_min = 1
    stripe_width_f_max = 5
    for mel_nr in range(B):
        mel_spec = X_copy2[mel_nr, 0]
        for _ in range(nr_stripes_f):
            stripe_width = random.randint(stripe_width_f_min, stripe_width_f_max)
            stripe_start_idx = random.randint(5, H - stripe_width_f_max - 1)

            mel_spec[stripe_start_idx:(stripe_start_idx+stripe_width), :] = 0.0

        X_copy2[mel_nr, 0] = mel_spec

    X = torch.cat((X, X_copy1, X_copy2), dim=0)
    y = torch.cat((y, y, y), dim=0)

    return X,y 


def create_train_val_test_split(split: list[float], stride_splits: float, device: torch.device, 
                                iteration: int)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    order, labels, metadata = create_order()
    patients = list(order.keys())
    patients2 = patients + patients

    fract_train, fract_val, fract_test, stride  = calc_fractions(patients, stride_splits, split)

    pX_train = patients2[iteration*stride:(fract_train+iteration*stride)] # a list of the patient names used to construct X_train for this iteration
    pX_val = patients2[(fract_train+iteration*stride):(fract_train+iteration*stride+fract_val)]
    pX_test = patients2[(fract_train+iteration*stride+fract_val):(fract_train+iteration*stride+fract_val+fract_test)]

    X_train, y_train = create_tensors_mel(pX_train, device, order, labels)
    X_val, y_val = create_tensors_mel(pX_val, device, order, labels)
    X_test, y_test = create_tensors_mel(pX_test, device, order, labels)

    # Data augmentation
    X_train, y_train = time_freq_masking(X_train, y_train)

    return X_train, y_train, X_val, y_val, X_test, y_test

def create_simple(split, stride_splits, device, iteration):
    order, _, metadata = create_order()
    patients = list(order.keys())
    patients2 = patients + patients

    fract_train, fract_val, fract_test, stride  = calc_fractions(patients, stride_splits, split)

    pX_train = patients2[iteration*stride:(fract_train+iteration*stride)] # a list of the patient names used to construct X_train for this iteration
    pX_val = patients2[(fract_train+iteration*stride):(fract_train+iteration*stride+fract_val)]
    pX_test = patients2[(fract_train+iteration*stride+fract_val):(fract_train+iteration*stride+fract_val+fract_test)]

    X_train = create_tensors_meta(pX_train, device, metadata)
    X_val = create_tensors_meta(pX_val, device, metadata)
    X_test = create_tensors_meta(pX_test, device, metadata)

    # Expand X_train_meta to same size as X_train carrying spectrograms (data augmentation from before)
    X_train = torch.cat((X_train, X_train, X_train), dim=0)

    return X_train, X_val, X_test

## calculate pos_weight parameter for optimizer
def pos_weight_calc():
    # definition: pos_weight = (#0)/(#1)
    df = pd.read_csv(r"data/train.csv")
    df = df[["AS","AR","MR","MS","N"]]
    
    pos_weights = []
    for col in range(df.shape[1]):
        column = df.iloc[:, col]
        column = np.array(column, dtype=np.int32)
        ones = np.sum(column)
        zeros = 108 - ones
        pos_weight = zeros/ones
        pos_weights.append(pos_weight)

    print(pos_weights)

# pos_weight_calc()
