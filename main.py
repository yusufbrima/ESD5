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
from dataset import ESC50Dataset,ESC50SpectrogramDataset
from models import CustomCNNModel
from config import FINAL_META_DATA,AUDIO_PATH,BATCH_SIZE,LEARNING_RATE,EPOCHS,SEED,MODELS_PATH,RESULTS_PATH
from utils import train_model,test_model,EarlyStopping,plot_confusion_matrix

# set the seed
# torch.random.manual_seed(SEED)
# np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelstr = 'resnet18'

if __name__ == "__main__":
    
    # Define the dataset and data loader
    # dataset = ESC50Dataset(csv_file=FINAL_META_DATA, audio_dir=AUDIO_PATH)
    dataset = ESC50SpectrogramDataset(csv_file=FINAL_META_DATA, audio_dir=AUDIO_PATH)
    
    # Define the sizes of the splits
    train_size = int(0.8 * len(dataset))
    val_size = int(0.10 * len(dataset))
    test_size = len(dataset) - train_size - val_size


    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    sample,label = next(iter(train_loader))

    input_shape = sample['data'].shape[1:]  

    num_classes = len(dataset.label_to_idx)
    # Define the customized vision model with 5 output classes
    model = CustomCNNModel(num_classes=num_classes, weights=None, modelstr=modelstr)
    # model = ESC50Model(input_shape=input_shape, num_cats=num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    early_stopping = EarlyStopping(patience=5, min_delta=0.01)



    # Train the model
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer,scheduler,early_stopping, num_epochs=EPOCHS, device=device)

    # Save the trained model
    torch.save(model.state_dict(), f'{MODELS_PATH}/custom_{modelstr}.pth')

    # Print the training and validation loss and accuracy history
    print('Training and validation loss and accuracy history:')
    print(history)

    # Example usage:
    # test_loss, test_acc = test_model(model, test_loader, criterion, device=device)
    test_loss, test_acc, all_labels, all_preds = test_model(model, test_loader, criterion, device=device)

    # save the test labels and predictions
    test_results = {'labels': all_labels, 'preds': all_preds}
    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(f'{RESULTS_PATH}/{modelstr}_test_scores.csv', index=False)

    # save the training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'{RESULTS_PATH}/{modelstr}_history.csv', index=False)

    # save the test results
    test_results = {'test_loss': test_loss, 'test_acc': test_acc}
    test_results_df = pd.DataFrame(test_results, index=[0])
    test_results_df.to_csv(f'{RESULTS_PATH}/{modelstr}_test_results.csv', index=False)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')


