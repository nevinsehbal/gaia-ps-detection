import sys
import os
# Add the src_dataset directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../src_dataset/')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../utils/')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../src_model/')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../src_loss/')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from customDataset import SeismicDataset, get_dataset
from customLoss import CustomLoss
import matplotlib.pyplot as plt
import os
import time
from logUtils import printCustom

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


class CNN_LSTM(nn.Module):
    def __init__(self, input_channels, hidden_size, output_dim, dropout_prob=0.3):
        super(CNN_LSTM, self).__init__()

        # Encoder (CNN layers)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            #nn.Dropout(p=dropout_prob),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            #nn.Dropout(p=dropout_prob),
            nn.MaxPool1d(kernel_size=3, stride=3),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            #nn.Dropout(p=dropout_prob),
            nn.MaxPool1d(kernel_size=5, stride=5),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            #nn.Dropout(p=dropout_prob),
            nn.MaxPool1d(kernel_size=5, stride=5),
        )

        # Calculate CNN output size for LSTM input_size
        dummy_input = torch.randn(1, input_channels, 900)  # Example input with 900 for window_size
        cnn_output = self.encoder(dummy_input)
        cnn_output_size = cnn_output.size(1) * cnn_output.size(2)

        # LSTM layer (single time step with `window_size` as input)
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=hidden_size, batch_first=True)
        #self.dropout = nn.Dropout(p=dropout_prob)

        # Output layers
        self.time_regression = nn.Linear(hidden_size, 2)  # P and S time regression
        self.wave_existence = nn.Linear(hidden_size, 4)   # Multiclass classification for wave existence (no_PS, only_P, only_S, both_PS)

    def forward(self, x, verbose=False):
        """
        Forward pass for CNN-LSTM.
        Args:
            x: Tensor of shape [batch_size, num_windows, channels, 1, window_size]
            verbose: Whether to print intermediate shapes for debugging.
        """
        batch_size, num_windows, channels, _, window_size = x.size()

        if verbose:
            print(f"Initial input shape: {x.shape}")

        # Reshape for CNN: [batch_size * num_windows, channels, window_size]
        x = x.view(batch_size*num_windows, channels,window_size).to(device)
        if verbose:
            print(f"Shape after reshaping for CNN: {x.shape}")

        # Pass through CNN layers
        x = self.encoder(x)  # Shape: [batch_size * num_windows, cnn_output_channels, cnn_output_length]
        if verbose:
            print(f"Shape after CNN: {x.shape}")

        cnn_output_channels = x.size(1)  # Number of channels
        cnn_output_length = x.size(2)  # Length of the feature map
        flattened_size = cnn_output_channels * cnn_output_length

        # Reshape for LSTM
        x = x.view(batch_size, num_windows, flattened_size)

        if verbose:
            print(f"Shape after reshaping for LSTM: {x.shape}")

        # Pass through LSTM
        x, (hn, cn) = self.lstm(x)  # hn: [num_layers, batch_size, hidden_size]
        if verbose:
            print(f"Shape after LSTM output: {x.shape}")
            print(f"Shape of final hidden state (hn): {hn.shape}")

        # Use the final hidden state of the last LSTM layer
        x = hn[-1]  # Shape: [batch_size, hidden_size]
        if verbose:
            print(f"Shape of the last LSTM hidden state: {x.shape}")

        # Output layers
        times = self.time_regression(x)  # Regression output (p_idx, s_idx)
        existence = self.wave_existence(x)  # Classification output (no_PS, only_P, only_S, both_PS)

        if verbose:
            print(f"Shape after time regression: {times.shape}")
            print(f"Shape after wave existence: {existence.shape}")

        # Concatenate outputs: [batch_size, 6]
        output = torch.cat((times, existence), dim=1)
        if verbose:
            print(f"Final output shape: {output.shape}")

        return output