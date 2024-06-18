from pathlib import Path
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader,random_split
from dataset import ESC50Dataset,ESC50SpectrogramDataset,AugmentAudio
from models import CustomResNet18,ESC50Model
from config import FINAL_META_DATA,AUDIO_PATH,BATCH_SIZE,LEARNING_RATE,EPOCHS,SEED,MODELS_PATH,DATA_PATH
from utils import train_model,test_model,EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__=="__main__":

    # Load metadata
    metadata = pd.read_csv(FINAL_META_DATA)

    # Split the metadata into train, validation, and test sets
    train_meta, temp_meta = train_test_split(metadata, test_size=0.2, random_state=42, stratify=metadata['category'])
    val_meta, test_meta = train_test_split(temp_meta, test_size=0.5, random_state=42, stratify=temp_meta['category'])

    # Save the splits as new CSV files (optional, for easy debugging or reuse)
    train_meta.to_csv(f'{DATA_PATH}/meta/train_meta.csv', index=False)
    val_meta.to_csv(f'{DATA_PATH}/meta/val_meta.csv', index=False)
    test_meta.to_csv(f'{DATA_PATH}/meta/test_meta.csv', index=False)

    # Define the dataset and data loader
    dataset = ESC50Dataset(csv_file=f'{DATA_PATH}/meta/train_meta.csv', audio_dir=AUDIO_PATH)
    
    # Create dataset instances
    train_dataset = ESC50SpectrogramDataset(csv_file=f'{DATA_PATH}/meta/train_meta.csv', audio_dir=AUDIO_PATH, augmentation=None)
    val_dataset = ESC50SpectrogramDataset(csv_file=f'{DATA_PATH}/meta/val_meta.csv', audio_dir=AUDIO_PATH)
    test_dataset = ESC50SpectrogramDataset(csv_file=f'{DATA_PATH}/meta/test_meta.csv', audio_dir=AUDIO_PATH)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    sample,label = next(iter(train_loader))

    print(len(train_dataset), len(val_dataset), len(test_dataset))