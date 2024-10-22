# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import pickle
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Custom PyTorch Dataset class for loading waveform data from CSV files
class WaveformDataset(Dataset):
    def __init__(self, folder_path, metadata_csv, max_length, max_waveform, max_elevation):
        # folder_path: directory containing waveform CSV files
        # metadata_csv: path to the metadata file that has the labels
        # max_length: maximum length of the waveform data
        # max_waveform: maximum value for normalizing waveform data
        # max_elevation: maximum value for normalizing elevation data
        
        self.folder_path = folder_path
        try:
            # Read the metadata CSV file containing labels (target values)
            self.metadata = pd.read_csv(metadata_csv, index_col='SiteID')
        except Exception as e:
            print(f"Error reading metadata CSV: {e}")
            raise
        
        # Filter metadata to include only valid site IDs that exist in the folder
        self.metadata = self.metadata[self.metadata.index.isin(self._get_valid_sites())]
        
        # List of valid site IDs (from the CSV files)
        self.file_names = self.metadata.index.tolist()
        
        self.max_length = max_length  # Maximum waveform length
        self.max_waveform = max_waveform  # Maximum waveform value for normalization
        self.max_elevation = max_elevation  # Maximum elevation value for normalization
    
    def __len__(self):
        # Returns the number of samples in the dataset
        return len(self.file_names)
    
    def __getitem__(self, idx):
        # Fetch the waveform and elevation data for a specific site ID (from its CSV file)
        site_id = self.file_names[idx]
        csv_path = os.path.join(self.folder_path, f"{site_id}.csv")
        data = pd.read_csv(csv_path)
        
        # Extract waveform and elevation data from the CSV file
        waveform_data = data['Rxwaveform'].values
        elevation_data = data['Elevation'].values
        
        # Initialize arrays with zeros and populate with actual data
        waveform = np.zeros(self.max_length)
        waveform[:len(waveform_data)] = waveform_data
        
        elevation = np.zeros(self.max_length)
        elevation[:len(elevation_data)] = elevation_data

        # Normalize the waveform and elevation data
        waveform = waveform / self.max_waveform
        elevation = elevation / self.max_elevation

        # Combine waveform and elevation into a 2D array
        combined = np.zeros((self.max_length, 2))
        combined[:, 0] = waveform
        combined[:, 1] = elevation
        
        # Convert to a PyTorch tensor and rearrange the shape (channels first for 1D CNN)
        combined = torch.tensor(combined, dtype=torch.float32).permute(1, 0)  # Shape: (2, max_length)
        
        # Fetch the label (target value) for this site from the metadata
        target = torch.tensor(self.metadata.loc[site_id, 'Live_Biomass_Density'], dtype=torch.float32)
        
        return combined, target  # Return the combined waveform/elevation data and its corresponding label

    # Private method to get valid site IDs (only CSV files that exist in the folder)
    def _get_valid_sites(self):
        valid_sites = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(".csv"):
                    site_id = file[:-4]  # Extract site ID from the filename (without ".csv")
                    if site_id in self.metadata.index:
                        valid_sites.append(site_id)
        return valid_sites

# Define the 1D Convolutional Neural Network model
class Waveform1DCNN(nn.Module):
    def __init__(self, input_length):
        super(Waveform1DCNN, self).__init__()
        # Convolutional layers with 1D convolution and padding to preserve input length
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Max pooling layer to downsample the data
        self.pool = nn.MaxPool1d(2)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers for regression
        self.fc1 = nn.Linear(256 * (input_length // 16), 256)  # Input dimension based on pooling
        self.fc2 = nn.Linear(256, 1)  # Output is a single value (biomass density)
    
    def forward(self, x):
        # Forward pass through the network
        x = self.pool(F.relu(self.conv1(x)))  # Convolution -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Convolution -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv3(x)))  # Convolution -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv4(x)))  # Convolution -> ReLU -> Pooling
        
        # Flatten the feature maps for the fully connected layers
        x = torch.flatten(x, start_dim=1)
        
        # Pass through fully connected layers with dropout in between
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)  # Final output layer
        
        return x  # Output is the predicted biomass density

# Main training and evaluation function
def main(folder_path, train_metadata_csv, test_metadata_csv, epochs=100, batch_size=16, model_output_path='./model_output/1dcnn'):
    # Load the training and testing metadata
    try:
        train_metadata = pd.read_csv(train_metadata_csv)
        test_metadata = pd.read_csv(test_metadata_csv)
        
        # Calculate mean and standard deviation of the target variable for standardization (optional)
        target_mean = train_metadata['Live_Biomass_Density'].mean()
        target_std = train_metadata['Live_Biomass_Density'].std()
    except Exception as e:
        print(f"Error reading metadata CSV: {e}")
        return

    # Create directories to save model output and metadata
    os.makedirs(model_output_path, exist_ok=True)
    train_metadata_path = os.path.join(model_output_path, 'train_metadata.csv')
    test_metadata_path = os.path.join(model_output_path, 'test_metadata.csv')
    train_metadata.to_csv(train_metadata_path)
    test_metadata.to_csv(test_metadata_path)
    
    # Determine the maximum waveform length and maximum values for normalization
    max_length = 0
    max_waveform = 0
    max_elevation = 0
    files_to_check = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    for file in files_to_check:
        data = pd.read_csv(file)
        waveform_length = len(data['Rxwaveform'])
        if waveform_length > max_length:
            max_length = waveform_length
        max_waveform = max(max_waveform, data['Rxwaveform'].max())
        max_elevation = max(max_elevation, data['Elevation'].max())

    # Initialize the training and testing datasets
    train_dataset = WaveformDataset(folder_path, train_metadata_path, max_length, max_waveform, max_elevation)
    test_dataset = WaveformDataset(folder_path, test_metadata_path, max_length, max_waveform, max_elevation)

    # Create data loaders for batching the data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create the 1D CNN model and move it to GPU (if available)
    model = Waveform1DCNN(input_length=max_length).cuda()
    
    # Define the loss function (Mean Squared Error for regression) and optimizer (Adam)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler to reduce the learning rate if test loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.001, patience=10, verbose=True)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    evaluation_file = os.path.join(model_output_path, 'evaluation_results.txt')

    # Training loop
    training_loss = []
    testing_loss = []
    lowest_test_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        total_train_samples = len(train_loader.dataset)

        # Training step
        for waveforms, targets in train_loader:
            waveforms, targets = waveforms.cuda(), targets.cuda()  # Move data to GPU
            optimizer.zero_grad()  # Zero the gradients
            
            outputs = model(waveforms)  # Forward pass
            loss = criterion(outputs.squeeze(), targets)  # Compute loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model weights
            
            running_loss += loss.item()  # Accumulate the loss

        # Calculate mean training loss per sample
        mean_sample_loss = running_loss / total_train_samples
        training_loss.append((epoch, mean_sample_loss))
        print(f"Epoch {epoch+1}, Mean Sample Loss: {mean_sample_loss}")

        # Validation step
        model.eval()  # Set model to evaluation mode
        total_test_samples = len(test_loader.dataset)
        test_loss = 0.0

        with torch.no_grad():  # Disable gradient calculation during testing
            all_targets = []
            all_outputs = []
            for waveforms, targets in test_loader:
                waveforms, targets = waveforms.cuda(), targets.cuda()
                outputs = model(waveforms)
                loss = criterion(outputs.squeeze(), targets)
                test_loss += loss.item()
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().squeeze().numpy())

        # Calculate mean test loss per sample
        test_mean_sample_loss = test_loss / total_test_samples
        testing_loss.append((epoch, test_mean_sample_loss))
        print(f"Epoch {epoch+1}, Test Mean Sample Loss: {test_mean_sample_loss}")

        # Update the scheduler with the test loss
        scheduler.step(test_mean_sample_loss)

        # Save the model if it has the lowest test loss so far
        if test_mean_sample_loss < lowest_test_loss:
            best_epoch = epoch
            lowest_test_loss = test_mean_sample_loss
            best_model = model.state_dict()  # Save the model's state
            torch.save(best_model, os.path.join(model_output_path, f'best_model_epoch_{epoch}.pth'))
            print(f"Model saved with Test Mean Sample Loss: {test_mean_sample_loss}")

    # Save the final best model for later use
    shutil.copy(os.path.join(model_output_path, f'best_model_epoch_{best_epoch}.pth'), os.path.join(model_output_path, 'best_model.pth'))

    # Save the training and testing loss data for future reference
    with open(os.path.join(model_output_path, 'gedi_biomass_training_loss.pkl'), 'wb') as f:
        pickle.dump(training_loss, f)
    with open(os.path.join(model_output_path, 'gedi_biomass_testing_loss.pkl'), 'wb') as f:
        pickle.dump(testing_loss, f)

    # Load the saved loss data
    with open(os.path.join(model_output_path, 'gedi_biomass_training_loss.pkl'), 'rb') as f:
        training_loss = pickle.load(f)
    with open(os.path.join(model_output_path, 'gedi_biomass_testing_loss.pkl'), 'rb') as f:
        testing_loss = pickle.load(f)

    # Plot and save the training and testing loss curves
    train_epochs, train_losses = zip(*training_loss)
    test_epochs, test_losses = zip(*testing_loss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_losses, label='Training Loss')
    plt.plot(test_epochs, test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Sample Loss')
    plt.title('Training and Testing Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_output_path, 'training_testing_loss_curve.png'))
    plt.close()

    # Rescale the final test predictions and targets
    all_targets_original = np.array(all_targets)
    all_outputs_original = np.array(all_outputs)

    # Plot and save the prediction vs true values scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets_original, all_outputs_original, color='blue')
    plt.plot([min(all_targets_original), max(all_targets_original)], 
             [min(all_targets_original), max(all_targets_original)], color='red', linewidth=2)
    plt.xlabel('True Values (Mg/ha)')
    plt.ylabel('Predicted Values (Mg/ha)')
    plt.title('Prediction vs True Values')
    plt.grid(True)
    plt.savefig(os.path.join(model_output_path, 'prediction_vs_true.png'))
    plt.close()

    # Calculate evaluation metrics on the final test predictions
    mae = mean_absolute_error(all_targets_original, all_outputs_original)
    mse = mean_squared_error(all_targets_original, all_outputs_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets_original, all_outputs_original)
    percentage_rmse = (rmse / np.mean(all_targets_original)) * 100

    # Print the final evaluation results
    print(f'Mean Absolute Error (MAE): {mae:.2f} Mg/ha')
    print(f'R-squared (R2): {r2:.2f}')
    print(f'Percentage RMSE: {percentage_rmse:.2f}%')
    print(f'RMSE: {rmse:.2f}')
    print(f'Number of samples in the testing set: {len(all_targets_original)}')
    print(f'Number of model parameters: {num_parameters}')

    # Save the final evaluation metrics to a text file
    with open(evaluation_file, 'a') as file:
        file.write(f'\nFinal Evaluation Metrics:\n')
        file.write(f'Mean Absolute Error (MAE): {mae:.2f} Mg/ha\n')
        file.write(f'R-squared (R2): {r2:.2f}\n')
        file.write(f'Percentage RMSE: {percentage_rmse:.2f}%\n')
        file.write(f'RMSE: {rmse:.2f}\n')
        file.write(f'Number of samples in the testing set: {len(all_targets_original)}\n')
        file.write(f'Number of model parameters: {num_parameters}\n')

# Entry point for the script to parse command-line arguments and call the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a 1D CNN on waveform data.')
    
    # Add command-line arguments for the script
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing waveform CSV files')
    parser.add_argument('--train_metadata_csv', type=str, required=True, help='Path to the training metadata CSV file')
    parser.add_argument('--test_metadata_csv', type=str, required=True, help='Path to the testing metadata CSV file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--model_output_path', type=str, default='./model_output', help='Directory to save model outputs')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Call the main function with the parsed arguments
    main(args.folder_path, args.train_metadata_csv, args.test_metadata_csv, args.epochs, args.batch_size, args.model_output_path)
