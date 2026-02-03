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
    transform = transforms.MelSpectrogram(
        sample_rate=4000,
        n_fft=4096,          # Larger FFT for better frequency resolution (1024ms window)
        win_length=4096,     # Window length
        hop_length=200,      # Smaller hop = finer time resolution (80000/200 = 400 time steps)
        f_min=0.5,           # Minimum frequency (0.5 Hz, 30 bpm)
        f_max=200,           # Maximum frequency (200 Hz - covers heart sounds + murmurs)
        n_mels=128,          # Number of mel frequency bins
        power=2.0            # Power spectrogram
    )

    for row in range(train_csv.shape[0]):
        for _, file_name in enumerate(train_csv.iloc[row, 6:]):
            wav_file, _ =  torchaudio.load(f'data/train/{file_name}.wav', normalize=True)
            wav_file = wav_file.squeeze(0)
            if wav_file.shape[0] != 80000:
                wav_file = fix_length(wav_file)
            
            # create spectogram
            mel_specgram = transform(wav_file)
            save_path = os.path.join(output_dir, f"{file_name}.pt")
            torch.save(mel_specgram, save_path)
        
device = torch.device("mps" if torch.mps.is_available() else "cpu")
### create_spectrograms(device)