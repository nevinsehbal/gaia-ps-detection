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
        self.wave_existence = nn.Linear(hidden_size, 2)   # P and S existence classification

    def forward(self, x):
        # Expected input shape: [batch_size, num_windows, channels, 1, window_size]
        batch_size, num_windows, channels, _, window_size = x.size()
        print(f"Initial input shape: {x.shape}")  # Print input shape

        # Reshape for CNN input as [batch_size * num_windows, channels, window_size] ?? where to put num_windowss???
        x = x.view(batch_size, channels, num_windows).to(device)
        print(f"Shape after reshaping for CNN: {x.shape}")  # Print reshaped input

        # Pass through CNN layers
        x = self.encoder(x)  # Shape should now be [batch_size * num_windows, cnn_output_channels, cnn_output_length]
        print(f"Shape after CNN: {x.shape}")  # Print shape after CNN layers

        # Reshape for LSTM input as [batch_size, num_windows, cnn_output_channels * cnn_output_length]
        x = x.view(batch_size, num_windows, -1)
        print(f"Shape after reshaping for LSTM: {x.shape}")  # Print reshaped input for LSTM

        # Pass through the LSTM
        x, (hn, cn) = self.lstm(x)
        print(f"Shape after LSTM output: {x.shape}")  # Print LSTM output shape
        print(f"Shape of final hidden state (hn): {hn.shape}")  # Print shape of LSTM hidden state

        x = hn[-1]
        print(f"Shape of the last LSTM hidden state (x): {x.shape}")  # Print shape of last LSTM hidden state

        # Output layers
        times = self.time_regression(x)
        print(f"Shape after time regression: {times.shape}")  # Print shape of time regression output

        existence = self.wave_existence(x)
        existence = torch.sigmoid(existence)
        print(f"Shape after wave existence and sigmoid: {existence.shape}")  # Print shape of wave existence output

        # Concatenate outputs
        output = torch.cat((times, existence), dim=1)
        print(f"Final output shape: {output.shape}")  # Print final output shape
        return output