"""
Configuration file for groundwater prediction project
"""
from dataclasses import dataclass
from typing import List


@dataclass
class DataProcessorConfig:
    """Data processor configuration - ALL parameters for data processing"""
    # Time ranges
    train_start: str = "1971-01-01"
    train_end: str = "2014-12-31"
    test_start: str = "2015-01-01"
    test_end: str = "2019-12-31"

    input_setting: str = "24"

    # IS (In-Sample) dataset - 636 known stations
    is_data_name: str = "LSTM_RND_636"
    model_name: str = f"{is_data_name}_{input_setting}"
    is_station_info_file: str = f"{is_data_name}_IS_TRAIN_input.csv"
    is_train_output_file: str = f"{model_name}_train_set.h5"
    is_test_output_file: str = f"{model_name}_IS_test_set.h5"
    is_scalers_file: str = f"{model_name}_scalers.pkl"
    
    # OOS (Out-of-Sample) dataset - 341 new unseen stations
    oos_data_name: str = "LSTM_RND_341"
    oos_station_info_file: str = f"{oos_data_name}_OOS_input.csv"
    oos_test_output_file: str = f"{oos_data_name}_{input_setting}_OOS_test_set.h5"
    
    # Sequence parameters
    seq_length_d: int = 3600  # Long sequence (days) d:decade
    seq_length_a: int = 360   # Short sequence (days) a:annual
    n_windows: int = 12       # Number of time windows

    dyn_features: List[str] = None
    data_dtype: str = "float32"  # Data type for arrays (float32 for memory efficiency, float64 for precision)
    
    # Optional seasonal reference feature
    add_pet_sin: bool = False  # Add PET-fitted sinusoidal seasonal reference
    
    def __post_init__(self):
        if self.dyn_features is None:
            self.dyn_features = ['P', 'PET', 'TAS', 'DTR', 'HUSS', 'WIND', 'PSURF', 'RLDS', 'RSDS']


@dataclass
class ModelConfig:
    """Model training configuration"""
    loss_function: str = "CW_NSELoss"

    # Model parameters
    hidden_size: int = 256
    num_layers: int = 1
    dropout_rate: float = 0.4
    learning_rate: float = 0.001
    batch_size: int = 512

    # Training parameters
    num_epochs: int = 50  # Fixed epochs (no early stopping)
    
    # Learning rate schedule - fixed milestones (0-indexed epochs)
    lr_milestones: List[int] = None  # Epochs where LR decays (0-indexed: 10 means epoch 11)
    lr_decay_rates: List[float] = None  # Corresponding decay rates
    
    def __post_init__(self):
        if self.lr_milestones is None:
            # Decay at epoch 11 (index 10) and epoch 21 (index 20)
            # Based on Kratzert et al., 2019
            self.lr_milestones = [10, 25]
        if self.lr_decay_rates is None:
            # 1e-3 -> 5e-4 (multiply by 0.5)
            # 5e-4 -> 1e-4 (multiply by 0.2)
            self.lr_decay_rates = [0.5, 0.2]


@dataclass
class EnsembleConfig:
    """Ensemble model configuration"""
    num_seeds: int = 10  # Number of models in ensemble
    seed_start: int = 42  # First seed value
    
    # Evaluation metrics
    metrics: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['NSE', 'KGE', 'Pearson_r', 'Spearman_r', 'Mean_Ratio', 'STD_Ratio']
    
    @property
    def seeds(self) -> List[int]:
        """Get list of seeds for ensemble"""
        return list(range(self.seed_start, self.seed_start + self.num_seeds))


@dataclass
class PathConfig:
    """Path configuration"""
    work_dir: str = "/user/work"
    log_subdir: str = "logs"
    data_subdir: str = "data"
    processor_subdir: str = "processor"
    model_subdir: str = "models"
    result_subdir: str = "results"


# Default configurations
default_model_config = ModelConfig()
default_ensemble_config = EnsembleConfig()
default_path_config = PathConfig()
default_processor_config = DataProcessorConfig()