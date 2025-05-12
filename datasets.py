from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

def estimate_quantile(train_loader, quantiles):
    all_data = []

    # Collect all data from the train_loader
    for batch in train_loader:
        inputs, _ = batch
        all_data.append(inputs)

    # Concatenate all data along the first dimension
    all_data = torch.cat(all_data, dim=0)
    quantile_values = torch.quantile(all_data, quantiles, dim=0).transpose(0,1)
    return quantile_values

def get_quantization_thresholds(train_loader, n_bits):
    thresholds = 2 ** n_bits - 1
    quantiles = torch.arange(1 / (thresholds + 1), 1, 1 / (thresholds + 1))
    thresholds = estimate_quantile(train_loader, quantiles)
    return thresholds

def get_min_max_values(train_loader, num_features):
    min_values = torch.tensor([float('inf')] * num_features)
    max_values = torch.tensor([-float('inf')] * num_features)
    for batch in train_loader:
        inputs, _ = batch
        min_values = torch.min(min_values, inputs.min(dim=0).values)
        max_values = torch.max(max_values, inputs.max(dim=0).values)
    return min_values, max_values

def process_data(X,y, batch_size = 64, random_state= None):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    # Further split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Add an extra dimension for regression
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader