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
import argparse
import matplotlib.pyplot as plt
import pickle
import shutil

# Custom dataset class for handling waveform data from CSV files
class WaveformDataset(Dataset):
    def __init__(self, folder_path, metadata_csv, max_length, max_waveform, max_elevation):
        """
        Initializes the dataset object. 
        :param folder_path: Directory containing waveform CSV files
        :param metadata_csv: Path to metadata file that contains labels for the waveforms
        :param max_length: Maximum length of waveforms (determined from data)
        :param max_waveform: Maximum value of waveform for normalization
        :param max_elevation: Maximum elevation value for normalization
        """
        self.folder_path = folder_path
        try:
            # Read the metadata CSV containing labels (indexed by SiteID)
            self.metadata = pd.read_csv(metadata_csv, index_col='SiteID')
        except Exception as e:
            print(f"Error reading metadata CSV: {e}")
            raise
        
        # Filter metadata to include only valid site IDs with corresponding waveform CSVs
        self.metadata = self.metadata[self.metadata.index.isin(self._get_valid_sites())]
        
        # Store the valid filenames (site IDs)
        self.file_names = self.metadata.index.tolist()
        
        # Store normalization and waveform parameters
        self.max_length = max_length
        self.max_waveform = max_waveform
        self.max_elevation = max_elevation
    
    def __len__(self):
        # Return the total number of waveform samples
        return len(self.file_names)
    
    def __getitem__(self, idx):
        """
        Fetch the waveform and elevation data for a given index.
        :param idx: Index to retrieve the corresponding site data
        """
        # Get site ID based on index
        site_id = self.file_names[idx]
        
        # Load waveform and elevation data from the CSV file
        csv_path = os.path.join(self.folder_path, f"{site_id}.csv")
        data = pd.read_csv(csv_path)
        waveform_data = data['Rxwaveform'].values
        elevation_data = data['Elevation'].values
        
        # Initialize zero-padded arrays and fill with actual data
        waveform = np.zeros(self.max_length)
        waveform[:len(waveform_data)] = waveform_data
        
        elevation = np.zeros(self.max_length)
        elevation[:len(elevation_data)] = elevation_data

        # Normalize the waveform and elevation data
        waveform = waveform / self.max_waveform
        elevation = elevation / self.max_elevation

        # Combine waveform and elevation data into a single 2D array
        combined = np.zeros((self.max_length, 2))
        combined[:, 0] = waveform
        combined[:, 1] = elevation
        
        # Convert to a PyTorch tensor and adjust dimensions for LSTM input (batch_size, sequence_length, num_features)
        combined = torch.tensor(combined, dtype=torch.float32).permute(1, 0)  # Shape: (2, max_length)
        
        # Retrieve the target value (biomass density) from metadata
        target = torch.tensor(self.metadata.loc[site_id, 'Live_Biomass_Density'], dtype=torch.float32)
        
        return combined, target  # Return the input data and the corresponding label

    def _get_valid_sites(self):
        # Retrieve valid site IDs based on the available CSV files in the folder
        valid_sites = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(".csv"):
                    site_id = file[:-4]  # Extract site ID (filename without extension)
                    if site_id in self.metadata.index:
                        valid_sites.append(site_id)
        return valid_sites

# Define the LSTM model architecture
class WaveformLSTM(nn.Module):
    def __init__(self, input_length, hidden_dim=128, num_layers=3):
        """
        Initializes the LSTM model.
        :param input_length: Length of input waveform data
        :param hidden_dim: Number of hidden units in the LSTM layers
        :param num_layers: Number of LSTM layers in the model
        """
        super(WaveformLSTM, self).__init__()
        
        # LSTM network with input_size = 2 (waveform and elevation), hidden_size = hidden_dim, and num_layers
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.5)
        
        # Fully connected layers to map LSTM outputs to a single output (regression)
        self.fc1 = nn.Linear(hidden_dim * input_length, 128)
        self.fc2 = nn.Linear(128, 1)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        :param x: Input tensor of shape (batch_size, 2, max_length)
        """
        # Initialize hidden and cell states for LSTM
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).cuda()
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).cuda()
        
        # Adjust input shape for LSTM (batch_size, max_length, 2)
        x = x.permute(0, 2, 1)
        
        # Pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Flatten the output for the fully connected layer
        out = out.contiguous().view(out.size(0), -1)
        
        # Pass through fully connected layers with dropout in between
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.fc2(out)  # Final output (predicted biomass density)
        
        return out

# Main function for training and evaluating the LSTM model
def main(folder_path, train_metadata_csv, test_metadata_csv, epochs=100, batch_size=16, model_output_path='./model_output/output_lstm'):
    # Load training and testing metadata
    try:
        train_metadata = pd.read_csv(train_metadata_csv)
        test_metadata = pd.read_csv(test_metadata_csv)
        # Compute mean and standard deviation of the target variable for potential normalization
        target_mean = train_metadata['Live_Biomass_Density'].mean()
        target_std = train_metadata['Live_Biomass_Density'].std()
    except Exception as e:
        print(f"Error reading metadata CSV: {e}")
        return

    # Create directories to save model outputs
    os.makedirs(model_output_path, exist_ok=True)
    train_metadata_path = os.path.join(model_output_path, 'train_metadata.csv')
    test_metadata_path = os.path.join(model_output_path, 'test_metadata.csv')
    train_metadata.to_csv(train_metadata_path)
    test_metadata.to_csv(test_metadata_path)
    
    # Determine the maximum waveform length, max waveform, and max elevation for normalization
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

    # Initialize dataset and dataloaders
    train_dataset = WaveformDataset(folder_path, train_metadata_path, max_length, max_waveform, max_elevation)
    test_dataset = WaveformDataset(folder_path, test_metadata_path, max_length, max_waveform, max_elevation)
    
    # DataLoader for batching the data during training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create LSTM model and move it to GPU
    model = WaveformLSTM(input_length=max_length, hidden_dim=128, num_layers=3).cuda()
    
    # Define loss function (MSE for regression) and optimizer (Adam)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler to reduce learning rate if the test loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Initialize variables for tracking training progress
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    evaluation_file = os.path.join(model_output_path, 'evaluation_results.txt')

    # Training and evaluation loop
    training_loss = []
    testing_loss = []
    lowest_test_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        model.train()  # Set model to training mode
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
            running_loss += loss.item()

        # Calculate mean training loss per sample
        mean_sample_loss = running_loss / total_train_samples
        training_loss.append((epoch, mean_sample_loss))
        print(f"Epoch {epoch+1}, Mean Sample Loss: {mean_sample_loss}")

        # Evaluation step
        model.eval()  # Set model to evaluation mode
        total_test_samples = len(test_loader.dataset)
        test_loss = 0.0
        all_targets = []
        all_outputs = []

        with torch.no_grad():  # Disable gradient calculation for testing
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

        # Update learning rate scheduler based on test loss
        scheduler.step(test_mean_sample_loss)

        # Save the best model (lowest test loss)
        if test_mean_sample_loss < lowest_test_loss:
            best_epoch = epoch
            lowest_test_loss = test_mean_sample_loss
            best_model = model.state_dict()  # Save model parameters
            torch.save(best_model, os.path.join(model_output_path, f'best_model_epoch_{epoch}.pth'))
            print(f"Model saved with Test Mean Sample Loss: {test_mean_sample_loss}")

    # Save the final best model
    shutil.copy(os.path.join(model_output_path, f'best_model_epoch_{best_epoch}.pth'), os.path.join(model_output_path, 'best_model.pth'))

    # Save training and testing loss data for future reference
    with open(os.path.join(model_output_path, 'gedi_biomass_training_loss.pkl'), 'wb') as f:
        pickle.dump(training_loss, f)
    with open(os.path.join(model_output_path, 'gedi_biomass_testing_loss.pkl'), 'wb') as f:
        pickle.dump(testing_loss, f)

    # Load saved loss data for plotting
    with open(os.path.join(model_output_path, 'gedi_biomass_training_loss.pkl'), 'rb') as f:
        training_loss = pickle.load(f)
    with open(os.path.join(model_output_path, 'gedi_biomass_testing_loss.pkl'), 'rb') as f:
        testing_loss = pickle.load(f)

    # Plot training and testing loss curves
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

    # Plot predictions vs true values
    all_targets_original = np.array(all_targets)
    all_outputs_original = np.array(all_outputs)

    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets_original, all_outputs_original, color='blue')
    plt.plot([min(all_targets_original), max(all_targets_original)], [min(all_targets_original), max(all_targets_original)], color='red', linewidth=2)
    plt.xlabel('True Values (Mg/ha)')
    plt.ylabel('Predicted Values (Mg/ha)')
    plt.title('Prediction vs True Values')
    plt.grid(True)
    plt.savefig(os.path.join(model_output_path, 'prediction_vs_true.png'))
    plt.close()

    # Calculate evaluation metrics
    mae = mean_absolute_error(all_targets_original, all_outputs_original)
    mse = mean_squared_error(all_targets_original, all_outputs_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets_original, all_outputs_original)
    percentage_rmse = (rmse / np.mean(all_targets_original)) * 100

    # Print evaluation metrics
    print(f'Mean Absolute Error (MAE): {mae:.2f} Mg/ha')
    print(f'R-squared (R2): {r2:.2f}')
    print(f'Percentage RMSE: {percentage_rmse:.2f}%')
    print(f'RMSE: {rmse:.2f}')
    print(f'Number of samples in the testing set: {len(all_targets_original)}')
    print(f'Number of model parameters: {num_parameters}')

    # Save final evaluation metrics to a text file
    with open(evaluation_file, 'a') as file:
        file.write(f'\nFinal Evaluation Metrics:\n')
        file.write(f'Mean Absolute Error (MAE): {mae:.2f} Mg/ha\n')
        file.write(f'R-squared (R2): {r2:.2f}\n')
        file.write(f'Percentage RMSE: {percentage_rmse:.2f}%\n')
        file.write(f'Number of samples in the testing set: {len(all_targets_original)}\n')
        file.write(f'RMSE: {rmse:.2f}\n')
        file.write(f'Number of model parameters: {num_parameters}\n')

# Entry point for the script
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a LSTM on waveform data.')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing waveform CSV files')
    parser.add_argument('--train_metadata_csv', type=str, required=True, help='Path to the training metadata CSV file')
    parser.add_argument('--test_metadata_csv', type=str, required=True, help='Path to the testing metadata CSV file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--model_output_path', type=str, default='./model_output', help='Directory to save model outputs')

    # Call the main function with parsed arguments
    args = parser.parse_args()
    main(args.folder_path, args.train_metadata_csv, args.test_metadata_csv, args.epochs, args.batch_size, args.model_output_path)
