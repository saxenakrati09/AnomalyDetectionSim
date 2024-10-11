import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import torch.nn as nn



class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def autoencoder_main():
    # Load the data
    data = pd.read_csv('data.csv')

    # Extract features and labels
    features = data[['Casting_Speed', 'SEN_Depth', 'Tundish_Temperature', 'Mold_Level']]
    labels = data[['Casting_Speed_Label', 'SEN_Depth_Label', 'Tundish_Temperature_Label', 'Mold_Level_Label']]

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split the data into training and testing sets
    X_train, X_test = train_test_split(features_scaled, test_size=0.2, random_state=42)


    # Initialize the model, loss function, and optimizer
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Create DataLoader for batch processing
    train_loader = DataLoader(X_train_tensor, batch_size=32, shuffle=True)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for data in train_loader:
            # Forward pass
            output = model(data)
            loss = criterion(output, data)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Get reconstruction errors for training data
    model.eval()
    with torch.no_grad():
        train_reconstructions = model(X_train_tensor)
        train_errors = torch.mean((train_reconstructions - X_train_tensor) ** 2, dim=1).numpy()

    # Fit a kernel density estimator to the training errors
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(train_errors.reshape(-1, 1))

    # Determine the threshold for anomaly detection
    threshold = np.percentile(train_errors, 95)

    # Get reconstruction errors for test data
    with torch.no_grad():
        test_reconstructions = model(X_test_tensor)
        test_errors = torch.mean((test_reconstructions - X_test_tensor) ** 2, dim=1).numpy()

    # Detect anomalies
    anomalies = test_errors > threshold
    # print(f'Number of anomalies detected: {np.sum(anomalies)}')

    # Calculate error contribution rate for each variable
    def error_contribution_rate(original, reconstructed):
        error = (original - reconstructed) ** 2
        total_error = np.sum(error, axis=1, keepdims=True)
        contribution_rate = error / total_error
        return contribution_rate

    # Calculate contribution rates for test data
    with torch.no_grad():
        test_contribution_rates = error_contribution_rate(X_test_tensor.numpy(), test_reconstructions.numpy())

    # Print contribution rates for anomalies
    # for i, is_anomaly in enumerate(anomalies):
    #     if is_anomaly:
    #         print(f'Anomaly at index {i}: Contribution rates: {test_contribution_rates[i]}')
    return np.sum(anomalies)
