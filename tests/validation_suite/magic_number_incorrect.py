"""
Magic Number Validation - INCORRECT Implementation

This file demonstrates problematic use of magic numbers in ML code.
Contains hardcoded hyperparameters that should be extracted to configuration.

Expected: Multiple magic number patterns detected
"""

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def create_problematic_neural_network():
    """INCORRECT: Contains multiple magic numbers"""
    
    class ProblematicNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # Magic numbers: hardcoded dimensions and parameters
            self.fc1 = nn.Linear(784, 256)    # 256 is magic - should be configurable
            self.fc2 = nn.Linear(256, 512)    # 512 is magic - should be configurable  
            self.fc3 = nn.Linear(512, 64)     # 64 is magic - should be configurable
            self.fc4 = nn.Linear(64, 10)      # 10 is OK (standard number of classes)
            self.dropout = nn.Dropout(0.15)   # 0.15 is magic - should be configurable
            
        def forward(self, x):
            # Magic reshape dimensions
            x = x.view(-1, 784)  # 784 is OK (standard MNIST)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    return ProblematicNetwork()


def train_model_with_magic_numbers():
    """INCORRECT: Hardcoded hyperparameters everywhere"""
    
    # Magic numbers in data splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.17,        # Magic number - unusual split ratio
        random_state=42        # This is OK (standard seed)
    )
    
    # Magic numbers in model parameters
    model = RandomForestClassifier(
        n_estimators=73,       # Magic number - should be configurable
        max_depth=15,          # Magic number - should be configurable
        min_samples_split=7,   # Magic number - should be configurable
        min_samples_leaf=3,    # Magic number - should be configurable
        random_state=42,       # This is OK (standard seed)
        n_jobs=-1             # This is OK (use all cores)
    )
    
    return model


def create_optimizer_with_magic():
    """INCORRECT: Magic hyperparameters in optimizer"""
    
    # Multiple magic numbers in optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.00347,            # Magic learning rate - very specific, should be configurable
        weight_decay=0.0023,   # Magic weight decay - should be configurable  
        betas=(0.87, 0.995)    # Magic beta values - non-standard, should be configurable
    )
    
    return optimizer


def problematic_data_preprocessing():
    """INCORRECT: Magic numbers in data preprocessing"""
    
    # Magic normalization values
    mean = 0.485612    # Magic mean - should be computed or configurable
    std = 0.229347     # Magic std - should be computed or configurable
    
    # Magic dimensions in reshaping
    data = data.reshape(-1, 224, 224, 3)  # 224x224 might be magic depending on context
    data = (data - mean) / std
    
    # Magic scaling factors
    data = data * 255.0 * 0.98734  # Magic scaling - should be configurable
    
    return data


def training_loop_with_magic():
    """INCORRECT: Magic numbers in training configuration"""
    
    # Magic training parameters
    epochs = 147           # Magic number of epochs
    batch_size = 37        # Magic batch size - unusual value
    patience = 23          # Magic early stopping patience
    
    for epoch in range(epochs):
        # Magic learning rate scheduling
        if epoch == 50:     # Magic epoch number
            lr = lr * 0.73625  # Magic decay factor
        elif epoch == 100:  # Magic epoch number  
            lr = lr * 0.84217  # Magic decay factor
            
        # Magic loss thresholds
        if loss < 0.0523:   # Magic threshold
            break
            
        # Magic gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.847)  # Magic clip value


def problematic_model_architecture():
    """INCORRECT: Architecture with many magic dimensions"""
    
    # Magic convolution parameters
    conv1 = nn.Conv2d(3, 47, kernel_size=7, stride=3, padding=2)    # 47 filters is magic
    conv2 = nn.Conv2d(47, 93, kernel_size=5, stride=2, padding=1)   # 93 filters is magic
    conv3 = nn.Conv2d(93, 186, kernel_size=3, stride=1, padding=1)  # 186 filters is magic
    
    # Magic pooling
    pool = nn.MaxPool2d(kernel_size=3, stride=2)  # Unusual pooling parameters
    
    # Magic dropout rates
    dropout1 = nn.Dropout(0.237)  # Magic dropout rate
    dropout2 = nn.Dropout(0.458)  # Magic dropout rate
    
    return conv1, conv2, conv3, pool, dropout1, dropout2


def hyperparameter_tuning_with_magic():
    """INCORRECT: Magic numbers in hyperparameter ranges"""
    
    # Magic search spaces
    learning_rates = [0.00234, 0.00567, 0.00891]  # Magic values
    batch_sizes = [17, 37, 53, 71]                # Magic batch sizes
    hidden_sizes = [83, 167, 251, 347]            # Magic hidden sizes
    
    best_score = 0.0
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for hidden_size in hidden_sizes:
                # Train model with these magic hyperparameters
                score = train_and_evaluate(lr, batch_size, hidden_size)
                if score > best_score:
                    best_score = score


# Example of what NOT to do - inline magic numbers everywhere
def terrible_example():
    """INCORRECT: Worst possible example with magic numbers everywhere"""
    
    model = nn.Sequential(
        nn.Linear(784, 347),     # Magic hidden size
        nn.ReLU(),
        nn.Dropout(0.23),        # Magic dropout
        nn.Linear(347, 156),     # Magic hidden size
        nn.ReLU(), 
        nn.Dropout(0.41),        # Magic dropout
        nn.Linear(156, 73),      # Magic hidden size
        nn.ReLU(),
        nn.Linear(73, 10)        # 10 is OK (number of classes)
    )
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0123, momentum=0.876)  # Magic values
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion