from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

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

def estimate_quantile_image(train_loader, quantiles):
    all_data = []

    # Collect all data from the train_loader
    for batch in train_loader:
        inputs, _ = batch ## inputs shape [B, C, H, W]
        b, c, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).reshape(b * h * w, c)
        all_data.append(inputs)

    # Concatenate all data along the first dimension
    all_data = torch.cat(all_data, dim=0)
    quantile_values = torch.quantile(all_data, quantiles, dim=0).transpose(0,1)
    return quantile_values

def get_quantization_thresholds_image(train_loader, n_bits):
    thresholds = 2 ** n_bits - 1
    quantiles = torch.arange(1 / (thresholds + 1), 1, 1 / (thresholds + 1))
    thresholds = estimate_quantile_image(train_loader, quantiles)
    return thresholds

def get_minmax_thresholds(min_values, max_values, n_bits):
    M = 2 ** n_bits - 1
    s = (max_values - min_values) / (M) ## shape [num_features]
    s = s.unsqueeze(1)  # Add a new dimension to match the shape of min_values, shape [num_features, 1]
    thresholds = min_values.unsqueeze(1) + s * (torch.arange(1, M+1).unsqueeze(0) - 0.5) ## shape [num_features, M]
    return thresholds

def get_min_max_values(train_loader, num_features):
    min_values = torch.tensor([float('inf')] * num_features)
    max_values = torch.tensor([-float('inf')] * num_features)
    for batch in train_loader:
        inputs, _ = batch
        min_values = torch.min(min_values, inputs.min(dim=0).values)
        max_values = torch.max(max_values, inputs.max(dim=0).values)
    return min_values, max_values

def get_min_max_values_image(train_loader, num_features):
    min_values = torch.tensor([float('inf')] * num_features)
    max_values = torch.tensor([-float('inf')] * num_features)
    for batch in train_loader:
        inputs, _ = batch
        b, c, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).reshape(b * h * w, c)
        min_values = torch.min(min_values, inputs.min(dim=0).values)
        max_values = torch.max(max_values, inputs.max(dim=0).values)
    return min_values, max_values

def split_data(X,y, random_state= None):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    # Further split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    return X_train, y_train, X_test, y_test, X_val, y_val

def convert_to_tensor(X_train, y_train, X_test, y_test, X_val, y_val, batch_size=64):
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


def load_data(datasetname, scratch, splitdata=True):
    data_folder = scratch
    X_file = os.path.join(data_folder, datasetname + "X.npy")
    y_file = os.path.join(data_folder, datasetname + "Y.npy")
    # Check if the dataset already exists
    if os.path.exists(X_file) and os.path.exists(y_file):
        X = np.load(X_file, allow_pickle=True)
        y = np.load(y_file, allow_pickle=True)
    else:
        import openml
        dataset = openml.datasets.get_dataset(dataset_id=datasetname, version=1)
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        X = X.T[np.array(categorical_indicator) == False].T
        X = X.values
        y = y.values
        # Ensure the data folder exists
        os.makedirs(data_folder, exist_ok=True)
        np.save(X_file, X)
        np.save(y_file, y)
    if splitdata:
        X_train, y_train, X_test, y_test, X_val, y_val = split_data(X, y)
        return convert_to_tensor(X_train, y_train, X_test, y_test, X_val, y_val)
    else:
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)



def load_imagenet(batch_size=64):
    root_dir = '/home/ubuntu/work/saved_data/IMAGENET'
    from torchvision import models

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    from torchvision.datasets import ImageNet
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    train_dataset = ImageFolder(root=root_dir + '/val', transform=preprocess)
    val_dataset = ImageFolder(root=root_dir + '/val', transform=preprocess)
    # test_dataset = ImageFolder(root=root_dir + '/test/test', transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
    return train_loader, val_loader

def load_cifar(batch_size=64):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # CIFAR-10 transforms (standard)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Datasets and loaders
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    print("Creating CIFAR-10 dataloaders, train size :", len(train_set), " test size:", len(test_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    train_sub_loader = torch.utils.data.Subset(train_loader.dataset, range(2000))
    train_sub_loader = torch.utils.data.DataLoader(train_sub_loader, batch_size=train_loader.batch_size, shuffle=True)

    test_sub_loader = torch.utils.data.Subset(test_loader.dataset, range(1000))
    test_sub_loader = torch.utils.data.DataLoader(test_sub_loader, batch_size=test_loader.batch_size, shuffle=False)
    # return train_sub_loader, test_sub_loader
    return train_loader, test_loader