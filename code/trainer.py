import torch
import torch.optim as optim
import torch.nn as nn
import logging
import traceback
import time
from typing import Dict, List, Tuple, Optional, Any


class CombinedWeightNSELoss(torch.nn.Module):
    def __init__(self, eps: float = 0.1):
        super(CombinedWeightNSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor,
                q_sample_weights: torch.Tensor = None):
        squared_error = (y_pred - y_true) ** 2
        # Variance weight
        weight_V = 1 / (q_stds + self.eps) ** 2

        # Combined weight
        if q_sample_weights is not None:
            # Combined variance weights and sample size weights
            total_weight = weight_V * q_sample_weights
        else:
            total_weight = weight_V

        weighted_error = total_weight * squared_error
        return torch.mean(weighted_error)


class VarianceWeightNSELoss(torch.nn.Module):
    def __init__(self, eps: float = 0.1):
        super(VarianceWeightNSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        squared_error = (y_pred - y_true) ** 2
        weights = 1 / (q_stds + self.eps) ** 2
        weighted_nse = weights * squared_error

        return torch.mean(weighted_nse)


class ModelTrainer:
    def __init__(self, station_std_dict: Dict[str, float], station_count_dict: Dict[str, int], device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        self.device = device
        self.station_std_dict = station_std_dict
        self.station_sample_weight_dict = {}

        for station, count in station_count_dict.items():
            if count > 0:
                self.station_sample_weight_dict[station] = 1.0 / count
            else:
                self.station_sample_weight_dict[station] = 0

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")


    def _get_q_stds(self, station_nos):
        """Convert the well number to the corresponding standard deviation"""
        return torch.tensor(
            [self.station_std_dict.get(no, 1.0) for no in station_nos],
            dtype=torch.float32,
            device=self.device
        )

    def _get_q_sample_weights(self, station_nos):
        """Get the sample count weight for each station in the batch"""
        return torch.tensor(
            [self.station_sample_weight_dict.get(no, 1.0) for no in station_nos],
            dtype=torch.float32,
            device=self.device
        )

    def train_nn(self, model: nn.Module, train_loader: Any, num_epochs: int = 50,
                 learning_rate: float = 0.001, lr_milestones: List[int] = None, 
                 lr_decay_rates: List[float] = None, lossfunction: str = 'CW_NSELoss') -> Tuple[nn.Module, List[float]]:
        """
        Train LSTM model with fixed epochs and scheduled learning rate decay.
        
        Args:
            model: LSTM model to train
            train_loader: Training data loader
            num_epochs: Number of epochs to train (fixed, no early stopping)
            learning_rate: Initial learning rate
            lr_milestones: Epochs at which to decay learning rate
            lr_decay_rates: Decay rates at each milestone
            lossfunction: Loss function name
            
        Returns:
            Tuple of (trained_model, train_losses)
        """
        start_time = time.time()
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if lr_milestones is None:
            lr_milestones = [15, 30]
        if lr_decay_rates is None:
            lr_decay_rates = [0.5, 0.5]

        if lossfunction == 'VW_NSELoss':
            criterion = VarianceWeightNSELoss()
        elif lossfunction == 'CW_NSELoss':
            criterion = CombinedWeightNSELoss()
        else:
            criterion = nn.MSELoss()
        self.logger.info(f"Loss function: {lossfunction}")
        self.logger.info(f"Starting training: {num_epochs} epochs, lr={learning_rate}, milestones={lr_milestones}")

        train_losses = []
        current_lr = learning_rate

        for epoch in range(num_epochs):
            # Apply learning rate decay at milestones
            if epoch in lr_milestones:
                milestone_idx = lr_milestones.index(epoch)
                current_lr = current_lr * lr_decay_rates[milestone_idx]
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                self.logger.info(f"Epoch {epoch + 1}: Learning rate decayed to {current_lr:.6f}")

            # Training stage
            model.train()
            train_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                # Unpacking batch data
                inputs, targets, station_nos = batch
                station_nos = station_nos.numpy()  # Convert to numpy array

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)

                if lossfunction == 'VW_NSELoss':
                    q_stds = self._get_q_stds(station_nos)
                    loss = criterion(outputs.squeeze(), targets.squeeze(), q_stds)
                elif lossfunction == 'CW_NSELoss':
                    q_stds = self._get_q_stds(station_nos)
                    q_sample_weights = self._get_q_sample_weights(station_nos)
                    loss = criterion(outputs.squeeze(), targets.squeeze(), q_stds, q_sample_weights)
                else:
                    loss = criterion(outputs.squeeze(), targets.squeeze())

                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            epoch_loss = train_loss / len(train_loader)
            train_losses.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.8f}")
            elif epoch == 0:
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.8f}")

        elapsed = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed / 60:.1f} minutes.")

        return model, train_losses

    def evaluate_model(self, model, data_loader, lossfunction):
        """Evaluation model"""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        if lossfunction == 'VW_NSELoss':
            criterion = VarianceWeightNSELoss()
        elif lossfunction == 'CW_NSELoss':
            criterion = CombinedWeightNSELoss()
        else:
            criterion = nn.MSELoss()

        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:
                    inputs, targets, station_nos = batch
                    station_nos = station_nos.numpy()
                else:
                    inputs, targets = batch
                    station_nos = None

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)

                if lossfunction == 'VW_NSELoss':
                    q_stds = self._get_q_stds(station_nos)
                    loss = criterion(outputs.squeeze(), targets.squeeze(), q_stds)
                elif lossfunction == 'CW_NSELoss':
                    q_stds = self._get_q_stds(station_nos)
                    q_sample_weights = self._get_q_sample_weights(station_nos)
                    loss = criterion(outputs.squeeze(), targets.squeeze(), q_stds, q_sample_weights)
                else:
                    loss = criterion(outputs.squeeze(), targets.squeeze())

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        return total_loss / total_samples
