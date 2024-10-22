# Importing necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import timm
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import rasterio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import torch.optim as optim
import argparse
import pickle
import shutil

#%% Setting up the custom data class for loading and processing images and labels
class MyDataset(Dataset):
    def __init__(self, csvpath, ycol, input_dimension):
        # Load the CSV file containing paths to images and their corresponding labels
        self.df = pd.read_csv(csvpath)
        self.ycol = ycol  # Column that contains the label (target variable)
        self.input_dimension = input_dimension  # Input size required by the model
        
    def __len__(self):
        # Return the number of samples in the dataset
        return self.df.shape[0]    
    
    def __getitem__(self, index):
        # Retrieve the image path and the label for the given index
        image_path = self.df.loc[index, 'Filename_0_255']  
        
        # Read the image using rasterio
        image = rasterio.open(image_path).read()[:3,:,:]  # Keep only the first 3 channels (RGB)
        
        # Resize the image to the required input dimensions of the model
        image = cv2.resize(image.transpose(1, 2, 0), (self.input_dimension[1], self.input_dimension[2]))
        
        # Rearrange dimensions for PyTorch (C, H, W)
        image = image.transpose(2, 0, 1)
        
        # Retrieve the label for the current image
        y = self.df.loc[index, self.ycol]
        
        return image, y  # Return the image and its corresponding label

# Importing additional libraries
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Helper function to get input dimension for a specific model using timm
def get_input_dimension(model_name):
    try:
        # Create the model using the specified model name and get its input size
        model = timm.create_model(model_name, pretrained=True)
        input_size = model.default_cfg['input_size']
        return input_size
    except RuntimeError as e:
        print(f"Error loading model {model_name}: {e}")
        return None

# Helper function to count the number of parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Main training and evaluation function
def main(ycol, model_name, batch_size=50, epochs=100):
    # Get the required input dimensions for the specified model
    input_dimension = get_input_dimension(model_name)
    
    # Load the training and testing datasets
    train_dataset = MyDataset('train.csv', ycol, input_dimension)
    test_dataset = MyDataset('test.csv', ycol, input_dimension)

    # Create data loaders for batching and shuffling the data
    traindataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    testdataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Create the model using timm library with pretrained weights
    model = timm.create_model(model_name, pretrained=True, in_chans=3, num_classes=1)
    
    # Get the total number of parameters in the model
    num_parameters = count_parameters(model)

    # Move the model to GPU for faster training
    net = model.cuda()

    # Define the path for saving model checkpoints
    model_output_path = '/model_{}_{}_train_test/'.format(model_name, ycol)
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    
    # Define the loss function (Mean Squared Error) and the optimizer (Adam)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Set up learning rate scheduler to reduce learning rate when the validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    training_loss = []  # To track training loss over epochs
    testing_loss = []   # To track testing loss over epochs
    lowest_test_loss = float('inf')  # Variable to store the lowest test loss
    best_epoch = 0  # To track which epoch had the best model performance

    # Training loop
    for epoch in range(epochs):  # Loop over the dataset multiple times
        net.train()  # Set model to training mode
        running_loss = 0.0  # Reset running loss for each epoch
        total__train_samples = len(traindataloader.dataset)
        
        # Iterate through the training data
        for i, data in enumerate(traindataloader, 0):
            inputs, labels = data  # Get inputs and labels from the batch
            
            # Move data to GPU and set the inputs to float
            inputs = inputs.float().cuda()  
            
            optimizer.zero_grad()  # Zero the parameter gradients
            
            # Forward pass through the model
            outputs = torch.squeeze(net(inputs.cuda()))
            
            # Compute the loss
            loss = criterion(outputs, labels.float().cuda())
            
            # Backpropagate the loss
            loss.backward()
            
            # Update the model parameters
            optimizer.step()
            
            running_loss += loss.item()  # Accumulate the loss for this epoch

        # Compute the mean sample loss for this epoch
        mean_sample_loss = running_loss / total__train_samples
        training_loss.append((epoch, mean_sample_loss))  # Track training loss
        print(f"Epoch {epoch+1}, Mean Sample Loss: {mean_sample_loss}")

        # Validation step at the end of every epoch
        if epoch % 1 == 0:
            net.eval()  # Set model to evaluation mode
            total__test_samples = len(testdataloader.dataset)
            test_loss = 0
            
            # Iterate through the test data
            for i, data in enumerate(testdataloader, 0):
                inputs, labels = data  # Get inputs and labels
                inputs = inputs.float().cuda()  
                
                # Forward pass (no backpropagation in evaluation mode)
                outputs = torch.squeeze(net(inputs.cuda()))
                loss = criterion(outputs, labels.float().cuda())
                
                test_loss += loss.item()  # Accumulate test loss
            
            # Compute mean test loss
            test_mean_sample_loss = test_loss / total__test_samples
            scheduler.step(test_mean_sample_loss)  # Update the learning rate based on validation loss
            
            print(f"Epoch {epoch+1}, Test Mean Sample Loss: {test_mean_sample_loss}")
            testing_loss.append((epoch, test_mean_sample_loss))  # Track test loss
            
            # If the model has improved, save the current best model
            if test_mean_sample_loss < lowest_test_loss:
                best_epoch = epoch
                lowest_test_loss = test_mean_sample_loss
                best_model = net.state_dict()  # Save the best model parameters
                torch.save(best_model, os.path.join(model_output_path, 'best_model_epoch_{}.pth'.format(epoch)))
                print(f"Model saved with Test Mean Sample Loss: {test_mean_sample_loss}")

    # Copy the best model to a separate file for convenience
    shutil.copy(os.path.join(model_output_path, 'best_model_epoch_{}.pth'.format(best_epoch)), 
                os.path.join(model_output_path, 'best_model.pth'))                   

    # Save training and testing loss data to disk for future reference
    with open(os.path.join(model_output_path, 'gedi_biomass_training_loss.pkl'), 'wb') as f:
        pickle.dump(training_loss, f)
    with open(os.path.join(model_output_path, 'gedi_biomass_testing_loss.pkl'), 'wb') as f:
        pickle.dump(testing_loss, f)

    # Plot and save training and testing loss curves
    train_epochs, train_losses = zip(*training_loss)
    test_epochs, test_losses = zip(*testing_loss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_losses, label='Training Loss')
    plt.plot(test_epochs, test_losses, label='Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_output_path, 'training_testing_loss_curve.png'))        

    # Load the best model for final evaluation
    model_path = os.path.join(model_output_path, 'best_model.pth')
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # Testing loop to calculate predictions and true values for visualization
    y_true = []
    y_pred = []
    for i, data in enumerate(testdataloader, 0):
        inputs, labels = data  # Get inputs and labels
        inputs = inputs.float().cuda()  
        
        # Forward pass
        outputs = torch.squeeze(net(inputs.cuda())).cpu().detach().numpy()
        y_true.extend(labels.cpu().detach().numpy())  # Append true values
        y_pred.extend(outputs)  # Append predicted values

    # Plot the true vs predicted values as a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='blue')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red')
    plt.title('Prediction vs True Values')
    plt.xlabel('True Values (Mg/ha)')
    plt.ylabel('Predicted Values (Mg/ha)')
    plt.grid(True)
    plt.savefig(os.path.join(model_output_path, 'my_plot.png'))

    # Evaluation metrics (Mean Absolute Error, R-squared, RMSE, etc.)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    y_test = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    average_actual = np.mean(y_test)
    percentage_rmse = (rmse / average_actual) * 100

    # Print evaluation results
    print(f'Mean Absolute Error (MAE): {mae:.2f} Mg/ha')
    print(f'R-squared (R2): {r2:.2f}')
    print(f'Percentage RMSE: {percentage_rmse:.2f}%')
    print(f'RMSE: {rmse:.2f}')
    print(f'Number of samples in the testing set: {y_test.shape[0]}')
    print(f"Number of model parameters: {num_parameters}")

    # Write evaluation results to a text file
    with open(os.path.join(model_output_path, 'evaluation_results.txt'), 'w') as file:
        file.write(f'Mean Absolute Error (MAE): {mae:.2f} Mg/ha\n')
        file.write(f'R-squared (R2): {r2:.2f}\n')
        file.write(f'Percentage RMSE: {percentage_rmse:.2f}%\n')
        file.write(f'RMSE: {rmse:.2f}\n')
        file.write(f'Number of samples in the testing set: {y_test.shape[0]}\n')
        file.write(f'Number of model parameters: {num_parameters}\n')

# Entry point for running the script
if __name__ == "__main__":
    # Set up argument parsing to handle input parameters
    parser = argparse.ArgumentParser(description='Process some parameters.')
    
    # Define arguments for column name, model name, batch size, and epochs
    parser.add_argument('--ycol', type=str, default='Total_Biomass_Density', required=False, help='The Y Column')
    parser.add_argument('--model_name', type=str, default='resnet50', required=False, help='The model name')
    parser.add_argument('--batch_size', type=int, default=50, required=False, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, required=False, help='Number of training epochs')

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.ycol, args.model_name, args.batch_size, args.epochs)
