import torch, librosa
from data_loader import fix_length
import pandas as pd
import torchaudio
from torchaudio import transforms
import os

output_dir = "data/spectrograms"
os.makedirs(output_dir, exist_ok=True)

def create_spectrograms(device):

    train_csv = pd.read_csv('data/train.csv')
    transform = transforms.MelSpectrogram(sample_rate=4000)

    for row in range(train_csv.shape[0]):
        for _, file_name in enumerate(train_csv.iloc[row, 6:]):
            wav_file, _ =  torchaudio.load(f'data/train/{file_name}.wav', normalize=True)
            if wav_file.shape[0] != 80000:
                wav_file = fix_length(wav_file)
            
            # create spectogram
            mel_specgram = transform(wav_file)
            save_path = os.path.join(output_dir, f"{file_name}.pt")
            torch.save(mel_specgram, save_path)
        

# create_scalograms(device)