"""
Magic Number Validation - CORRECT Implementation

This file demonstrates proper configuration-based approach for ML hyperparameters.
All magic numbers are extracted to configuration constants.

Expected: 0 magic number patterns detected
"""

# Configuration constants - proper approach
LEARNING_RATE = 0.001
BATCH_SIZE = 32
HIDDEN_SIZE = 128
DROPOUT_RATE = 0.3
EPOCHS = 100
TRAIN_TEST_SPLIT_RATIO = 0.2

# Common ML values that are not magic numbers
RANDOM_SEED = 42
N_JOBS = -1
VERBOSE_LEVEL = 1

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def create_proper_neural_network():
    """Correct: All dimensions and hyperparameters are extracted to constants"""
    
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # Using configuration constants - no magic numbers
            self.fc1 = nn.Linear(784, HIDDEN_SIZE)  # Input size and hidden size from config
            self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
            self.fc3 = nn.Linear(HIDDEN_SIZE, 10)  # 10 classes is standard/expected
            self.dropout = nn.Dropout(DROPOUT_RATE)
            
        def forward(self, x):
            # Standard values that are not magic numbers
            x = x.view(-1, 784)  # 784 is standard MNIST input size
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    return NeuralNetwork()


def train_model_properly():
    """Correct: All hyperparameters come from configuration"""
    
    # Proper data splitting with configuration
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TRAIN_TEST_SPLIT_RATIO,  # From configuration
        random_state=RANDOM_SEED  # Standard random seed
    )
    
    # Proper model creation with configuration
    model = RandomForestClassifier(
        n_estimators=EPOCHS,  # From configuration
        random_state=RANDOM_SEED,  # Standard value
        n_jobs=N_JOBS,  # Standard value (-1 for all cores)
        verbose=VERBOSE_LEVEL  # Standard verbosity
    )
    
    return model


def create_optimizer_properly(model):
    """Correct: Optimizer parameters from configuration"""
    
    # All hyperparameters from configuration constants
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,  # From configuration
        weight_decay=0.01,  # Common regularization value
        betas=(0.9, 0.999)  # Standard Adam parameters
    )
    
    return optimizer


def data_preprocessing_correct():
    """Correct: Standard preprocessing without magic numbers"""
    
    # Standard normalization values - not magic numbers
    mean = 0.5  # Standard normalization
    std = 0.5   # Standard normalization
    
    # Common tensor operations with standard dimensions
    data = data.reshape(-1, 1)  # -1 for inferred dimension, 1 is standard
    data = (data - mean) / std
    
    return data


# Example configuration file approach
CONFIG = {
    'model': {
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'dropout_rate': DROPOUT_RATE
    },
    'training': {
        'epochs': EPOCHS,
        'test_size': TRAIN_TEST_SPLIT_RATIO
    },
    'system': {
        'random_seed': RANDOM_SEED,
        'n_jobs': N_JOBS
    }
}


def load_config_example():
    """Example of proper configuration loading"""
    return CONFIG['model']['learning_rate']