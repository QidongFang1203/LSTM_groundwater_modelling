import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator
from typing import Tuple


class GroundwaterDataset(Dataset):
    def __init__(self, input: np.ndarray, targets: np.ndarray, info: np.ndarray) -> None:
        self.input = torch.FloatTensor(input)
        self.targets = torch.FloatTensor(targets)
        self.station_nos = np.array([item[0] for item in info])

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return (self.input[idx], self.targets[idx], self.station_nos[idx])


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout_rate: float, output_size: int = 1,
                 initial_forget_bias: int = 5):
        """
        LSTM Network for sequence data, using the optimized nn.LSTM module
        with custom parameter initialization.

        Args:
            input_size: Number of features per timestep.
            hidden_size: Size of LSTM hidden state.
            num_layers: Number of stacked LSTM layers.
            dropout_rate: Dropout probability. Applied between LSTM layers
                          (if num_layers > 1) and before the final FC layer.
            output_size: Output size (default 1 for regression).
            initial_forget_bias: (New) Initial bias for the forget gate.
                                 A high value (e.g., 5) helps with long-term
                                 dependencies by initializing the forget
                                 gate to remember more.
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)

        # Dropout and output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

        # Call custom parameter initialization at the end of __init__
        self.reset_lstm_parameters(initial_forget_bias)

    def reset_lstm_parameters(self, initial_forget_bias: int):
        """Initialize all learnable parameters of the LSTM"""
        # Iterate over each layer (for stacked LSTMs)
        for layer_idx in range(self.num_layers):
            # --- 1. Initialize Weights ---
            weight_ih = getattr(self.lstm, f'weight_ih_l{layer_idx}')
            nn.init.orthogonal_(weight_ih.data)
            weight_hh = getattr(self.lstm, f'weight_hh_l{layer_idx}')
            weight_hh_data = torch.cat([torch.eye(self.hidden_size) for _ in range(4)], dim=0)
            weight_hh.data = weight_hh_data

            # --- 2. Initialize Biases ---
            bias_ih = getattr(self.lstm, f'bias_ih_l{layer_idx}')
            bias_hh = getattr(self.lstm, f'bias_hh_l{layer_idx}')
            nn.init.constant_(bias_ih.data, val=0)
            nn.init.constant_(bias_hh.data, val=0)

            n = self.hidden_size
            forget_gate_bias_segment = bias_ih.data[n: 2 * n]
            forget_gate_bias_segment.fill_(initial_forget_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Final prediction tensor of shape (batch_size, output_size)
        """
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)

        return out


class standardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, output_size=1):
        """
        LSTM Network for sequence data

        Args:
            input_size: Number of features per timestep
            hidden_size: Size of LSTM hidden state
            num_layers: Number of stacked LSTM layers
            dropout_rate: Dropout probability between LSTM layers
            output_size: Output size (default 1 for regression)
        """
        super(standardLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)

        # Dropout and output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)

        # Use output from last timestep
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out


class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, output_size=1):
        """
        Artificial Neural Network with configurable layers

        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layers
            num_layers: Number of hidden layers
            dropout_rate: Dropout probability (0-1)
            output_size: Size of output layer (default 1 for regression)
        """
        super(ANN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Input layer
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # Combine all layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_size)
        return self.model(x)
