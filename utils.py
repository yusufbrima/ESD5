import os
import requests
from zipfile import ZipFile
import shutil
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from config import MODELS_PATH,FIG_PATH

class DatasetDownloader:
    """
    A class to download and extract datasets from a given URL.

    Attributes:
        url (str): The URL of the zip file to download.
        download_path (str): The local file path to save the downloaded zip file.
        extract_path (str): The local directory to extract the contents of the zip file.

    Methods:
        download(): Downloads the zip file from the specified URL.
        extract(): Extracts the contents of the downloaded zip file to the specified directory.
        organize_files(): Moves the audio and meta folders to the desired locations and removes the extracted directory.
        clean_up(): Removes the downloaded zip file to clean up.
        download_and_extract(): Downloads, extracts, organizes, and then removes the zip file.
    """
    def __init__(self, url, download_path, extract_path):
        """
        Constructs all the necessary attributes for the DatasetDownloader object.

        Parameters:
            url (str): The URL of the zip file to download.
            download_path (str): The local file path to save the downloaded zip file.
            extract_path (str): The local directory to extract the contents of the zip file.
        """
        self.url = url
        self.download_path = download_path
        self.extract_path = extract_path

    def download(self):
        """Downloads the zip file from the specified URL."""
        # Create the directory if it doesn't exist
        os.makedirs(self.extract_path, exist_ok=True)

        # Download the zip file
        print(f"Downloading {self.url}...")
        response = requests.get(self.url)
        with open(self.download_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded to {self.download_path}")

    def extract(self):
        """Extracts the contents of the downloaded zip file to the specified directory."""
        # Extract the zip file
        print(f"Extracting {self.download_path} to {self.extract_path}...")
        with ZipFile(self.download_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_path)
        print("Extraction complete!")

    def organize_files(self):
        """Moves the audio and meta folders to the desired locations and removes the extracted directory."""
        extracted_dir = os.path.join(self.extract_path, 'ESC-50-master')
        audio_src = os.path.join(extracted_dir, 'audio')
        meta_src = os.path.join(extracted_dir, 'meta')
        audio_dest = os.path.join(self.extract_path, 'audio')
        meta_dest = os.path.join(self.extract_path, 'meta')

        # Move the audio folder
        print(f"Moving {audio_src} to {audio_dest}...")
        shutil.move(audio_src, audio_dest)

        # Move the meta folder
        print(f"Moving {meta_src} to {meta_dest}...")
        shutil.move(meta_src, meta_dest)

        # Remove the extracted directory
        print(f"Removing directory {extracted_dir}...")
        shutil.rmtree(extracted_dir)
        print("File organization complete!")

    def clean_up(self):
        """Removes the downloaded zip file to clean up."""
        # Clean up the zip file
        os.remove(self.download_path)
        print(f"Removed zip file {self.download_path}")

    def download_and_extract(self):
        """Downloads, extracts, organizes, and then removes the zip file."""
        self.download()
        self.extract()
        self.organize_files()
        self.clean_up()


def process_audio_labels(meta_file, esc50_file, audio_path, output_file):
    """
    Processes audio labels and categorizes them based on a given metadata file.
    
    Parameters:
        meta_file (str): Path to the major categories CSV file.
        esc50_file (str): Path to the ESC-50 labels CSV file.
        audio_path (str): Path to the directory containing audio files.
        output_file (str): Path to the output CSV file.
    """
    # Read the major categories CSV file
    df = pd.read_csv(meta_file)

    # Read the ESC-50 labels CSV file
    labels = pd.read_csv(esc50_file)

    # Initialize a dictionary to store the processed data
    data = {"filename": [], "target": [], "category": [], "src_file":[], "take": [], "duration": [], "sr": []}

    # Iterate through each label in the ESC-50 labels file with a progress bar
    for i in tqdm(range(len(labels)), desc="Processing labels"):
        filename = labels.iloc[i, 0]
        label = labels.iloc[i, 3]
        src_file = labels.iloc[i, 5]
        take = labels.iloc[i, 6]
        flag = False  # Flag to check if the label is found in any category

        # Check if the label exists in any column of the major categories dataframe
        for column in df.columns:
            if "_" in label:
                # Replace underscores with spaces in the label
                label = " ".join(label.split("_"))
            if label.capitalize() in df[column].values:
                # Load the audio file
                y, sr = librosa.load(f"{audio_path}/{filename}", sr=None)
                # Append the audio data details to the data dictionary
                data["duration"].append(len(y) / sr)
                data["sr"].append(sr)
                data["filename"].append(filename)
                data["target"].append(label)
                data["category"].append(column)
                data["src_file"].append(src_file)
                data["take"].append(take)
                flag = True  # Set flag to True if the label is found
                break
        
        if not flag:
            # Print the label if it is not found in any category
            print(label)

    # Create a DataFrame from the data dictionary
    f_df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    f_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")


def train_model(model, train_loader, val_loader, criterion, optimizer,scheduler,early_stopping, num_epochs=25, device="cpu", save_path='saved_model.pth'):
    """
    Train the model with the given data loaders, loss function, and optimizer.

    Parameters:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        optimizer (torch.optim.Optimizer): Optimizer.
        early_stopping (EarlyStopping): Early stopping object.
        num_epochs (int): Number of epochs to train the model.
        device (str): Device to use for training ('cpu' or 'cuda').
        save_path (str): Path to save the best model.

    Returns:
        model (nn.Module): The trained model.
        dict: Dictionary containing training and validation loss and accuracy history.
    """
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    min_valid_loss = np.inf

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_loss = 0.0
        train_corrects = 0
        model.train()  # Set model to training mode
        
        for samples, labels in train_loader:
            data, labels = samples["data"].to(device), labels.to(device)
            
            # Clear the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Calculate training loss and accuracy
            train_loss += loss.item() * data.size(0)
            train_corrects += torch.sum(preds == labels.data).item()
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_corrects / len(train_loader.dataset)
        
        valid_loss = 0.0
        valid_corrects = 0
        model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            for samples, labels in val_loader:
                data, labels = samples['data'].to(device), labels.to(device)
                
                # Forward pass
                outputs = model(data)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Calculate validation loss and accuracy
                valid_loss += loss.item() * data.size(0)
                valid_corrects += torch.sum(preds == labels.data).item()
        
        valid_loss /= len(val_loader.dataset)
        valid_acc = valid_corrects / len(val_loader.dataset)
        
        print(f'Training Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Validation Loss: {valid_loss:.4f} Acc: {valid_acc:.4f}')
        
        if valid_loss < min_valid_loss:
            print(f'Validation Loss Decreased ({min_valid_loss:.6f} --> {valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{MODELS_PATH}/{save_path}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(valid_loss)
        history['val_acc'].append(valid_acc)
        
        if early_stopping(valid_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
        # Step the scheduler
        scheduler.step()

    return model, history


def test_model(model, test_loader, criterion, device="cpu"):
    test_loss = 0.0
    test_corrects = 0
    all_labels = []
    all_preds = []
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data['data'].to(device), labels.to(device)
            
            # Forward pass
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Calculate test loss and accuracy
            test_loss += loss.item() * data.size(0)
            test_corrects += torch.sum(preds == labels.data).item()
            
            # Collect all labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    test_acc = test_corrects / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    return test_loss, test_acc, all_labels, all_preds

def plot_confusion_matrix(true_labels, pred_labels, class_names, modelstr="resnet18"):
    cm = confusion_matrix(true_labels, pred_labels)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{FIG_PATH}/{modelstr}_confusion_matrix.png')
    plt.close()
    # plt.show()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

if __name__ == "__main__":
    pass
