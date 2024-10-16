import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from icecream import ic
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Custom Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, scaler = None):
        self.data = pd.read_csv(csv_file)
        self.features = self.data[['Casting_Speed', 'SEN_Depth', 'Tundish_Temperature', 'Mold_Level']].values
        self.labels = self.data[['Casting_Speed_Label', 'SEN_Depth_Label', 'Tundish_Temperature_Label', 'Mold_Level_Label']].values
        self.label_mapping = {'normal': 0, 'impulse': 1, 'step': 2, 'slowvarying': 3}
        self.labels = self._map_labels(self.labels)
        if scaler:
            self.features = scaler.transform(self.features)

    def _map_labels(self, labels):
        mapped_labels = []
        for label_set in labels:
            mapped_labels.append([self.label_mapping[label] for label in label_set])
        return torch.tensor(mapped_labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), self.labels[idx]

# Scale Conversion Layer
class ScaleConversionLayer(nn.Module):
    def __init__(self):
        super(ScaleConversionLayer, self).__init__()
        self.smoothing = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        self.downsampling = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        identity = x
        smooth = self.smoothing(x)
        downsample = self.downsampling(x)
        return identity, smooth, downsample

# Convolutional Layer with Regularization
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.global_pool(x)
        return x

# LSTM Regularization Layer
class LSTMRegularization(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMRegularization, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1) 
        return x

# Fully Connected Layer
class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x

# Complete Model
class AnomalyDetectionModel(nn.Module):
    def __init__(self):
        super(AnomalyDetectionModel, self).__init__()
        self.scale_conversion = ScaleConversionLayer()
        in_channels, out_channels = 1, 16
        input_size, hidden_size, num_layers = 4, 4, 2
        num_classes = 4
        self.conv1 = ConvLayer(in_channels, out_channels)
        self.conv2 = ConvLayer(in_channels, out_channels)
        self.lstm = LSTMRegularization(input_size, hidden_size, num_layers)
        self.fc = FullyConnectedLayer(out_channels + out_channels + hidden_size, num_classes)  # 4 classes for each label
        
    def forward(self, x):
        identity, smooth, downsample = self.scale_conversion(x)
        identity = self.lstm(identity)
        smooth = self.conv1(smooth)
        downsample = self.conv2(downsample)
        combined = torch.cat((identity, smooth, downsample), dim=1)
        combined = combined.permute(0, 2, 1)
        out = self.fc(combined)
        return out

# Example usage
# if __name__ == "__main__":
#     dataset = TimeSeriesDataset('data.csv')
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#     model = AnomalyDetectionModel()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     # Training loop
#     for epoch in tqdm(range(10)):  # Number of epochs
#         for data, labels in dataloader:
#             optimizer.zero_grad()
#             outputs = model(data.unsqueeze(1))
#             loss = criterion(outputs.view(-1, 4), labels.float())  # Flatten for multi-class classification
#             loss.backward()
#             optimizer.step()
#         print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def cnnlstm():
    data_ = pd.read_csv('data.csv')
    features = data_[['Casting_Speed', 'SEN_Depth', 'Tundish_Temperature', 'Mold_Level']].values
    scaler = StandardScaler().fit(features)
    dataset = TimeSeriesDataset('data.csv') #, scaler=scaler)
    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # 20% for testing
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    # train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = AnomalyDetectionModel()
    # criterion = FocalLoss(alpha=0.1, gamma=3, reduction='mean')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    for epoch in range(10):  # Number of epochs
        model.train()
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data.unsqueeze(1))
            loss = criterion(outputs.view(-1, 4), labels.float())  # Flatten for multi-class classification
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

    # Evaluation loop
    model.eval()
    abnormal_counts = {'Casting_Speed': [0, 0, 0, 0], 'SEN_Depth': [0, 0, 0, 0], 'Tundish_Temperature': [0, 0, 0, 0], 'Mold_Level': [0, 0, 0, 0]}
    feature_labels = ['Casting_Speed', 'SEN_Depth', 'Tundish_Temperature', 'Mold_Level']
    with torch.no_grad():
        total, correct = 0, 0
        for data, labels in test_loader:
            
            outputs = model(data.unsqueeze(1))
            _, predicted = torch.max(outputs, 1)
            total += labels.view(-1).size(0)
            correct += (predicted.view(-1) == labels.view(-1)).sum().item()
            
            # Count the abnormal intervals
            for i, feature in enumerate(feature_labels):
                for j in range(4):  # 4 classes
                    abnormal_counts[feature][j] += (predicted[:, i] == j).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        # print("Abnormal Interval Counts:")
        # for feature, counts in abnormal_counts.items():
        #     print(f"{feature}: {counts}")
    return abnormal_counts, len(data_)