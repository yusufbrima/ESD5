import pandas as pd
import torchaudio
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as transforms
import torchaudio.transforms as T
import random


class AugmentAudio:
    def __init__(self):
        self.transforms = [
            T.Vol(gain=0.9),
            T.TimeStretch(fixed_rate=0.8),
            T.FrequencyMasking(freq_mask_param=15),
            T.TimeMasking(time_mask_param=35)
        ]

    def __call__(self, audio):
        if random.random() > 0.5:
            for transform in self.transforms:
                audio = transform(audio)
        return audio

class ESC50Dataset(Dataset):
    """
    A PyTorch Dataset class for loading audio data and labels from the ESC-50 dataset.

    Attributes:
        csv_file (str): Path to the CSV file containing the metadata.
        audio_dir (str): Directory where the audio files are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, csv_file, audio_dir, transform=None):
        """
        Initialize the dataset with metadata and audio directory.

        Parameters:
            csv_file (str): Path to the CSV file containing the metadata.
            audio_dir (str): Directory where the audio files are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.transform = transform
        # self.CLASSES = self.metadata['category'].unique().tolist()

        # Create a mapping from category names to class IDs
        self.label_to_idx = {label: idx for idx, label in enumerate(self.metadata['category'].unique())}
        self.CLASSES = list(self.label_to_idx.keys())

    def get_key_from_value(self, value):
        """
        Returns the key corresponding to the given value in the dictionary.

        Parameters:
            d (dict): The dictionary to search.
            value: The value to find the corresponding key for.

        Returns:
            key: The key corresponding to the given value.
        """
        for key, val in self.label_to_idx.items():
            if val == value:
                return key
        return None  # Return None if the value is not found

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Generates one sample of data.

        Parameters:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: A tuple containing the audio sample (waveform and sample rate) and the label (class ID).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the file name and label
        audio_path = f"{self.audio_dir}/{self.metadata.iloc[idx, 0]}"
        label_name = self.metadata.iloc[idx, 2]  # 'category' column
        label_idx = self.label_to_idx[label_name]  # Convert label name to class ID

        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)

        # print("Sampling rate:",  sample_rate)
        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)

        sample = {'data': waveform, 'sample_rate': sample_rate}

        return sample, label_idx


class ESC50SpectrogramDataset(ESC50Dataset):
    """
    A PyTorch Dataset class for loading audio data, converting it to log spectrograms, and loading labels from the ESC-50 dataset.
    """
    
    def __init__(self, csv_file, audio_dir, transform=None, augmentation=None):
        """
        Initialize the dataset with metadata and audio directory.

        Parameters:
            csv_file (str): Path to the CSV file containing the metadata.
            audio_dir (str): Directory where the audio files are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
            augmentation (callable, optional): Optional augmentation to be applied on the audio waveform.
        """
        super().__init__(csv_file, audio_dir, transform)
        self.augmentation = augmentation
        self.spectrogram_transform = transforms.Spectrogram(n_fft=512, hop_length=256, power=2)
        self.amplitude_to_db_transform = transforms.AmplitudeToDB()

    def __getitem__(self, idx):
        """
        Generates one sample of data and converts it to a log spectrogram.

        Parameters:
            idx (int): Index of the sample to be fetched.

        Returns:
            tuple: A tuple containing the log spectrogram and the label (class ID).
        """
        sample, label_idx = super().__getitem__(idx)
        waveform = sample['data']
        sample_rate = sample['sample_rate']

        # Apply augmentation if provided
        if self.augmentation:
            waveform = self.augmentation(waveform)

        # Convert waveform to spectrogram
        spectrogram = self.spectrogram_transform(waveform)
        log_spectrogram = self.amplitude_to_db_transform(spectrogram)

        # Set the dtype to float32
        log_spectrogram = log_spectrogram.type(torch.float32)

        return {'data': log_spectrogram, 'sample_rate': sample_rate}, label_idx

if __name__ == "__main__":
    pass
