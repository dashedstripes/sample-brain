import os
import librosa
import torch
from torch.utils.data import Dataset
import numpy as np

class WaveNetDataset(Dataset):
    def __init__(self, directory_path, sequence_length=16000):
        self.file_paths = librosa.util.find_files(directory_path)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        audio = librosa.load(file_path, sr=16000, mono=True)
        data = audio[0]

        # audio[0] = list of numbers, probably the mono signal
        # audio[1] = sample_rate

        # each audio signal needs to be the same length
        # currently, each wav file is a different length
        # therefore, we need to pad the end
        
        # Apply zero-padding or truncation to ensure fixed length
        if len(data) < self.sequence_length:
            padding = np.zeros(self.sequence_length - len(data))
            data = np.concatenate([data, padding])
        elif len(data) > self.sequence_length:
            data = data[:self.sequence_length]
        
        # Convert to tensor and add channel dimension
        # audio = torch.tensor(audio).float().unsqueeze(0)

        input = torch.tensor(data).float().unsqueeze(0)
        target = torch.tensor(data).float().unsqueeze(0)

        input_normalized = (input - input.min()) / (input.max() - input.min())
        target_normalized = (target - target.min()) / (target.max() - target.min())
        
        return input_normalized, target_normalized