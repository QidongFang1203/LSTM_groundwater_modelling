"""
Main script for training LSTM ensemble with multiple seeds and evaluation
"""
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
import warnings
import logging
import pickle
import h5py
import os
import pandas as pd
from datetime import datetime
import traceback
import sys

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import LSTM, GroundwaterDataset
from trainer import ModelTrainer
from metrics import EvaluationMetrics, EnsembleEvaluator, create_results_dataframe
from config import default_model_config, default_ensemble_config, default_path_config, default_processor_config


def setup_logging(model_no):
    """Setup logging configuration"""
    log_dir = os.path.join(default_path_config.work_dir, os.getenv('USER') or os.getenv('USERNAME') or 'default_user', default_path_config.log_subdir)
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_no}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_dir


def get_base_paths():
    """Get base directory paths"""
    username = os.getenv('USER') or os.getenv('USERNAME') or 'default_user'
    if not username:
        raise EnvironmentError("No USER or USERNAME environment variable found")

    work_dir = os.path.join(default_path_config.work_dir, username)
    data_dir = os.path.join(work_dir, default_path_config.data_subdir)
    model_dir = os.path.join(work_dir, default_path_config.model_subdir)
    result_dir = os.path.join(work_dir, default_path_config.result_subdir)

    for dir_path in [data_dir, model_dir, result_dir]:
        os.makedirs(dir_path, exist_ok=True)

    return {
        'work_dir': work_dir,
        'data_dir': data_dir,
        'model_dir': model_dir,
        'result_dir': result_dir
    }


warnings.filterwarnings('ignore')


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def calculate_station_stats(train_y_scaled: np.ndarray, train_info: np.ndarray) -> tuple:
    """Calculate standard deviation and sample count for each station"""
    station_nos = np.array([item[0] for item in train_info])
    unique_stations = np.unique(station_nos)

    station_std_dict = {}
    station_count_dict = {}

    for station in unique_stations:
        station_mask = (station_nos == station)
        station_std = np.std(train_y_scaled[station_mask])
        station_count = station_mask.sum()

        station_std_dict[station] = station_std
        station_count_dict[station] = station_count

    return station_std_dict, station_count_dict


def train_single_model(seed: int, 
                      model_name: str,
                      train_X: np.ndarray, train_y: np.ndarray,
                      test_X: np.ndarray, test_y: np.ndarray,
                      train_info: np.ndarray, test_info: np.ndarray,
                      paths: dict, loss_func: str = 'CW_NSELoss') -> tuple:
    """
    Train a single LSTM model with given seed
    
    Args:
        seed: Random seed for reproducibility
        model_name: Name for the model
        train_X, train_y: Training data
        test_X, test_y: Test data
        train_info, test_info: Metadata
        paths: Dictionary with directory paths
        loss_func: Loss function name
        
    Returns:
        Tuple of (train_pred, test_pred, model)
    """
    set_seed(seed)
    
    # Use configuration from config.py
    batch_size = default_model_config.batch_size
    num_epochs = default_model_config.num_epochs
    learning_rate = default_model_config.learning_rate
    
    best_params = {
        'hidden_size': default_model_config.hidden_size,
        'num_layers': default_model_config.num_layers,
        'dropout_rate': default_model_config.dropout_rate,
        'learning_rate': learning_rate
    }

    # Create model
    model = LSTM(
        input_size=train_X.shape[-1],
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout_rate=best_params['dropout_rate']
    )

    train_dataset = GroundwaterDataset(train_X, train_y, train_info)
    train_loader_shuffled = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    station_std_dict, station_count_dict = calculate_station_stats(train_y, train_info)
    trainer = ModelTrainer(station_std_dict=station_std_dict, station_count_dict=station_count_dict)

    # Train model with fixed epochs and learning rate decay
    try:
        model, train_losses = trainer.train_nn(
            model, train_loader_shuffled,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_milestones=default_model_config.lr_milestones,
            lr_decay_rates=default_model_config.lr_decay_rates,
            lossfunction=loss_func
        )
        logging.info(f"Model {model_name} (seed={seed}) trained successfully")
    except Exception as e:
        logging.error(f"Model training failed for seed {seed}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

    # Make predictions
    test_dataset = GroundwaterDataset(test_X, test_y, test_info)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader_inorder = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    train_predictions = []
    with torch.no_grad():
        for sequences, targets, info in train_loader_inorder:
            sequences = sequences.to(trainer.device)
            outputs = model(sequences)
            train_predictions.extend(outputs.cpu().numpy().flatten())
    
    test_predictions = []
    with torch.no_grad():
        for sequences, targets, info in test_loader:
            sequences = sequences.to(trainer.device)
            outputs = model(sequences)
            test_predictions.extend(outputs.cpu().numpy().flatten())

    # Save model
    model_path = os.path.join(paths['model_dir'], f'{model_name}_seed{seed}.pth')
    torch.save(model.state_dict(), model_path)

    return np.array(train_predictions), np.array(test_predictions), model


def read_group(group):
    """Recursively read HDF5 Group and its contents. Handles None gracefully."""
    if group is None:
        return {}
    
    data = {}
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            data[name] = read_group(item)
        elif isinstance(item, h5py.Dataset):
            data[name] = item[()]
    return data


def save_results_to_csv(results_df: pd.DataFrame, filepath: str) -> None:
    """
    Save results to CSV file
    
    Args:
        results_df: Results dataframe
        filepath: Output CSV file path
    """
    results_df.to_csv(filepath, index=False)
    logging.info(f"Results saved to: {filepath}")


def main():
    """Main execution function"""
    # Configuration from config.py
    loss_func = default_model_config.loss_function
    data_no = default_processor_config.model_name
    num_seeds = default_ensemble_config.num_seeds
    seeds = default_ensemble_config.seeds
    
    model_no = f'{data_no}_{loss_func}_ensemble{num_seeds}'
    
    # Setup
    paths = get_base_paths()
    setup_logging(model_no)

    logging.info(f"Starting ensemble model: {model_no}")
    logging.info(f"Number of seeds: {num_seeds}, Seeds: {seeds}")
    logging.info(f"Working directory: {paths['work_dir']}")

    # Load data from separate files
    try:
        # ========== TRAINING DATA (1971-2014, 637 known stations) ==========
        is_train_file = default_processor_config.is_train_output_file
        is_train_path = os.path.join(paths['data_dir'], is_train_file)
        
        with h5py.File(is_train_path, 'r') as hdf_train:
            train_X_scaled = hdf_train['train_X_scaled'][:].astype(default_processor_config.data_dtype)
            train_y_scaled = hdf_train['train_y_scaled'][:].astype(default_processor_config.data_dtype)
            train_y = hdf_train['train_y'][:].astype(default_processor_config.data_dtype)
            train_group = hdf_train.get('train_meta')
            if train_group is not None:
                meta_data = read_group(train_group)
                train_info = meta_data.get('info', np.arange(len(train_y_scaled)).reshape(-1, 1))
            else:
                # Fallback: create placeholder info if metadata doesn't exist
                train_info = np.arange(len(train_y_scaled)).reshape(-1, 1)

        # ========== IS TEST DATA (2015-2019, 637 known stations) ==========
        is_test_file = default_processor_config.is_test_output_file
        is_test_path = os.path.join(paths['data_dir'], is_test_file)
        
        with h5py.File(is_test_path, 'r') as hdf_is_test:
            is_test_X_scaled = hdf_is_test['test_X_scaled'][:].astype(default_processor_config.data_dtype)
            is_test_y_scaled = hdf_is_test['test_y_scaled'][:].astype(default_processor_config.data_dtype)
            is_test_y = hdf_is_test['test_y'][:].astype(default_processor_config.data_dtype)
            is_test_group = hdf_is_test.get('test_meta')
            if is_test_group is not None:
                meta_data = read_group(is_test_group)
                is_test_info = meta_data.get('info', np.arange(len(is_test_y_scaled)).reshape(-1, 1))
            else:
                # Fallback: create placeholder info if metadata doesn't exist
                is_test_info = np.arange(len(is_test_y_scaled)).reshape(-1, 1)

        # ========== OOS TEST DATA (2015-2019, 341 new unseen stations) ==========
        oos_test_file = default_processor_config.oos_test_output_file
        oos_test_path = os.path.join(paths['data_dir'], oos_test_file)
        
        with h5py.File(oos_test_path, 'r') as hdf_oos_test:
            oos_test_X = hdf_oos_test['test_X'][:].astype(default_processor_config.data_dtype)
            oos_test_y = hdf_oos_test['test_y'][:].astype(default_processor_config.data_dtype)
            oos_test_group = hdf_oos_test.get('test_meta')
            if oos_test_group is not None:
                meta_data = read_group(oos_test_group)
                oos_test_info = meta_data.get('info', np.arange(len(oos_test_y)).reshape(-1, 1))
            else:
                # Fallback: create placeholder info if metadata doesn't exist
                oos_test_info = np.arange(len(oos_test_y)).reshape(-1, 1)

        # Load scalers (from IS dataset)
        scaler_file = default_processor_config.is_scalers_file
        scaler_path = os.path.join(paths['data_dir'], scaler_file)
        
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
            scaler_target = scalers['target_scaler']
            scaler_static = scalers['static_scaler']
            
            # Check if using daily mode or multi-window mode
            use_daily_mode = 'dyn_scaler' in scalers
            if use_daily_mode:
                scaler_dyn = scalers['dyn_scaler']
            else:
                scaler_long = scalers['long_dyn_scaler']
                scaler_short = scalers['short_dyn_scaler']

        # Scale OOS test data using IS scalers (domain adaptation)
        base_dyn_features = [f for f in default_processor_config.dyn_features if f != 'PET_sin']
        DYN_FEAT_COUNT = len(base_dyn_features) + (1 if default_processor_config.add_pet_sin else 0)
        
        if use_daily_mode:
            # Daily mode: scale all dynamic features together
            X_oos_dyn = oos_test_X[:, :, :DYN_FEAT_COUNT]
            X_oos_static = oos_test_X[:, :, DYN_FEAT_COUNT:]
            
            X_oos_dyn_scaled = scaler_dyn.transform(
                X_oos_dyn.reshape(-1, DYN_FEAT_COUNT)).reshape(X_oos_dyn.shape)
            X_oos_static_scaled = scaler_static.transform(
                X_oos_static.reshape(-1, X_oos_static.shape[-1])).reshape(X_oos_static.shape)
            
            oos_test_X_scaled = np.concatenate([X_oos_dyn_scaled, X_oos_static_scaled], axis=2)
        else:
            # Multi-window mode: separate long and short sequences
            LONG_STEPS = default_processor_config.n_windows
            X_oos_long = oos_test_X[:, :LONG_STEPS, :DYN_FEAT_COUNT]
            X_oos_short = oos_test_X[:, LONG_STEPS:, :DYN_FEAT_COUNT]
            X_oos_static = oos_test_X[:, :, DYN_FEAT_COUNT:]
            
            X_oos_long_scaled = scaler_long.transform(
                X_oos_long.reshape(-1, DYN_FEAT_COUNT)).reshape(X_oos_long.shape)
            X_oos_short_scaled = scaler_short.transform(
                X_oos_short.reshape(-1, DYN_FEAT_COUNT)).reshape(X_oos_short.shape)
            X_oos_static_scaled = scaler_static.transform(
                X_oos_static.reshape(-1, X_oos_static.shape[-1])).reshape(X_oos_static.shape)
            
            oos_dyn_combined = np.concatenate([X_oos_long_scaled, X_oos_short_scaled], axis=1)
            oos_test_X_scaled = np.concatenate([oos_dyn_combined, X_oos_static_scaled], axis=2)
        
        oos_test_y_scaled = scaler_target.transform(oos_test_y.reshape(-1, 1)).flatten()

        logging.info("Data loaded and prepared successfully")
        logging.info(f"  Training: {len(train_y)} samples from {len(np.unique([item[0] for item in train_info]))} stations")
        logging.info(f"  IS test: {len(is_test_y)} samples from {len(np.unique([item[0] for item in is_test_info]))} stations")
        logging.info(f"  OOS test: {len(oos_test_y)} samples from {len(np.unique([item[0] for item in oos_test_info]))} unseen stations")
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

    # Train ensemble models
    logging.info(f"Training {num_seeds} models with different seeds...")
    ensemble_train_predictions = []
    ensemble_is_test_predictions = []
    ensemble_oos_test_predictions = []

    for i, seed in enumerate(seeds):
        logging.info(f"Training model {i+1}/{num_seeds} with seed {seed}")
        try:
            # Note: For simplicity, train only on IS data (636 known stations)
            # Then predict on both IS and OOS test data
            train_pred_scaled, is_test_pred_scaled, model = train_single_model(
                seed=seed,
                model_name=model_no,
                train_X=train_X_scaled, train_y=train_y_scaled,
                test_X=is_test_X_scaled, test_y=is_test_y_scaled,
                train_info=train_info, test_info=is_test_info,
                paths=paths, loss_func=loss_func
            )

            # Inverse transform predictions
            train_pred = scaler_target.inverse_transform(
                np.array(train_pred_scaled).reshape(-1, 1)
            ).flatten()
            is_test_pred = scaler_target.inverse_transform(
                np.array(is_test_pred_scaled).reshape(-1, 1)
            ).flatten()

            ensemble_train_predictions.append(train_pred)
            ensemble_is_test_predictions.append(is_test_pred)

            # Predict on OOS test data
            model.eval()
            oos_test_dataset = GroundwaterDataset(oos_test_X_scaled, oos_test_y_scaled, oos_test_info)
            oos_test_loader = TorchDataLoader(oos_test_dataset, batch_size=default_model_config.batch_size, shuffle=False)
            
            oos_test_pred_scaled = []
            with torch.no_grad():
                for sequences, targets, info in oos_test_loader:
                    sequences = sequences.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    outputs = model(sequences)
                    oos_test_pred_scaled.extend(outputs.cpu().numpy().flatten())
            
            oos_test_pred = scaler_target.inverse_transform(
                np.array(oos_test_pred_scaled).reshape(-1, 1)
            ).flatten()
            
            ensemble_oos_test_predictions.append(oos_test_pred)

        except Exception as e:
            logging.error(f"Failed to train model with seed {seed}: {str(e)}")
            logging.error(traceback.format_exc())
            sys.exit(1)

    logging.info(f"All {num_seeds} models trained successfully")

    # Evaluate ensemble predictions
    logging.info("Evaluating ensemble predictions...")
    evaluator = EnsembleEvaluator(num_seeds=num_seeds)

    # Extract station IDs
    train_station_ids = np.array([item[0] for item in train_info])
    is_test_station_ids = np.array([item[0] for item in is_test_info])
    oos_test_station_ids = np.array([item[0] for item in oos_test_info])

    logging.info(f"\nData Summary:")
    logging.info(f"  Train: {len(train_y)} samples from {len(np.unique(train_station_ids))} stations")
    logging.info(f"  IS test (In-Sample): {len(is_test_y)} samples from {len(np.unique(is_test_station_ids))} stations")
    logging.info(f"  OOS test (Out-of-Sample): {len(oos_test_y)} samples from {len(np.unique(oos_test_station_ids))} unseen stations")

    # ========== TRAIN PERIOD EVALUATION (1971-2014) ==========
    logging.info("\n" + "="*80)
    logging.info("Evaluating TRAIN period (1971-2014) on training data...")
    logging.info("="*80)
    
    train_period_station_metrics = evaluator.evaluate_ensemble_by_station(
        train_y, ensemble_train_predictions, train_station_ids
    )
    train_period_overall_metrics = evaluator.evaluate_overall(train_y, ensemble_train_predictions)
    train_period_results_df = create_results_dataframe(
        train_period_station_metrics, train_period_overall_metrics, mode='Train'
    )
    logging.info(f"Train period evaluation complete: {len(train_period_station_metrics)} stations")

    # ========== IS TEST EVALUATION (2015-2019, KNOWN STATIONS) ==========
    logging.info("\n" + "="*80)
    logging.info("Evaluating IS test period (2015-2019) on known stations...")
    logging.info("="*80)
    
    is_station_metrics = evaluator.evaluate_ensemble_by_station(
        is_test_y, ensemble_is_test_predictions, is_test_station_ids
    )
    is_overall_metrics = evaluator.evaluate_overall(is_test_y, ensemble_is_test_predictions)
    is_results_df = create_results_dataframe(
        is_station_metrics, is_overall_metrics, mode='IS'
    )
    logging.info(f"IS test evaluation complete: {len(is_station_metrics)} stations")

    # ========== OOS TEST EVALUATION (2015-2019, UNSEEN STATIONS) ==========
    logging.info("\n" + "="*80)
    logging.info("Evaluating OOS test period (2015-2019) on unseen stations...")
    logging.info("="*80)
    
    oos_station_metrics = evaluator.evaluate_ensemble_by_station(
        oos_test_y, ensemble_oos_test_predictions, oos_test_station_ids
    )
    oos_overall_metrics = evaluator.evaluate_overall(oos_test_y, ensemble_oos_test_predictions)
    oos_results_df = create_results_dataframe(
        oos_station_metrics, oos_overall_metrics, mode='OOS'
    )
    logging.info(f"OOS test evaluation complete: {len(oos_station_metrics)} stations")

    # Combine Train and IS results into one file
    train_is_combined_df = pd.concat([train_period_results_df, is_results_df], ignore_index=True)

    # Save results to CSV
    try:
        train_is_csv_path = os.path.join(paths['result_dir'], f'{model_no}_Train_IS_results.csv')
        oos_csv_path = os.path.join(paths['result_dir'], f'{model_no}_OOS_results.csv')

        save_results_to_csv(train_is_combined_df, train_is_csv_path)
        save_results_to_csv(oos_results_df, oos_csv_path)

        logging.info(f"\nResults saved successfully:")
        logging.info(f"  Train+IS results: {train_is_csv_path}")
        logging.info(f"  OOS results: {oos_csv_path}")

    except Exception as e:
        logging.error(f"Failed to save results: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

    # Write summary to log
    logging.info("\n" + "="*80)
    logging.info("ENSEMBLE EVALUATION SUMMARY")
    logging.info("="*80)
    
    logging.info("\nTRAIN Period (1971-2014) - Overall Metrics:")
    for metric, value in train_period_overall_metrics.items():
        if metric != 'Total_Samples':
            logging.info(f"  {metric}: {value:.4f}")

    logging.info("\nIS test Period (2015-2019, Known Stations) - Overall Metrics:")
    for metric, value in is_overall_metrics.items():
        if metric != 'Total_Samples':
            logging.info(f"  {metric}: {value:.4f}")

    logging.info("\nOOS test Period (2015-2019, Unseen Stations) - Overall Metrics:")
    for metric, value in oos_overall_metrics.items():
        if metric != 'Total_Samples':
            logging.info(f"  {metric}: {value:.4f}")

    logging.info("\n" + "="*80)
    logging.info("Groundwater ensemble modeling completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Main execution failed: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
