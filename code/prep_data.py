#!/usr/bin/env python
"""
Data preprocessing script - Calls GroundwaterDataProcessor to generate HDF5 datasets.
This is a one-time preprocessing step that generates:
  - train_X.h5, train_y.h5 (training data, 1071-2014)
  - test_X.h5, test_y.h5 (IS test, 2015-2019)
"""
import os
import sys
import logging
from sklearn.preprocessing import StandardScaler
import h5py
import pickle
import numpy as np

from data_processor import GroundwaterDataProcessor
from config import default_processor_config, default_path_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_is_dataset():
    """Preprocess IS (In-Sample) dataset - 637 known stations"""
    logger.info("=" * 80)
    logger.info("PREPROCESSING IS (In-Sample) DATASET")
    logger.info("=" * 80)
    
    # Setup paths
    username = os.getenv('USER', 'default_user')
    work_dir = os.path.join(default_path_config.work_dir, username)
    processor_dir = os.path.join(work_dir, default_path_config.processor_subdir)
    output_dir = os.path.expanduser(f'~/.cache/groundwater_data_{default_processor_config.is_data_name}')
    
    # Initialize processor
    processor = GroundwaterDataProcessor(
        os.path.join(processor_dir, 'GD_timeseries'),
        os.path.join(processor_dir, 'CHESS_timeseries'),
        os.path.join(processor_dir, default_processor_config.is_station_info_file),
        default_processor_config.dyn_features,
        default_processor_config.add_pet_sin
    )
    
    station_nos = processor.station_info['No'].tolist()
    logger.info(f"Processing {len(station_nos)} IS stations...")
    
    # Call HDF5 streaming prepare_data
    result = processor.prepare_data(
        station_nos,
        train_start=default_processor_config.train_start,
        train_end=default_processor_config.train_end,
        test_start=default_processor_config.test_start,
        test_end=default_processor_config.test_end,
        seq_length_d=default_processor_config.seq_length_d,
        seq_length_a=default_processor_config.seq_length_a,
        n_windows=default_processor_config.n_windows,
        output_dir=output_dir
    )
    
    # Load and normalize training data
    logger.info("Normalizing data...")
    with h5py.File(result['train_X_file'], 'r') as f:
        train_X = f['X'][:]  # Load for scaling (necessary once)
    with h5py.File(result['train_y_file'], 'r') as f:
        train_y = f['y'][:]
    
    # Fit scalers on training data
    base_dyn_features = [f for f in default_processor_config.dyn_features if f != 'PET_sin']
    n_dyn = len(base_dyn_features) + (1 if default_processor_config.add_pet_sin else 0)
    
    # Determine if using daily mode or multi-window mode
    use_daily_mode = (default_processor_config.seq_length_d == default_processor_config.seq_length_a)
    
    if use_daily_mode:
        # Daily mode: all steps are continuous, no separation
        logger.info("Mode: DAILY (no long/short separation)")
        X_dyn = train_X[:, :, :n_dyn]
        X_static = train_X[:, :, n_dyn:]
        
        scaler_dyn = StandardScaler()
        scaler_static = StandardScaler()
        scaler_target = StandardScaler()
        
        scaler_dyn.fit(X_dyn.reshape(-1, n_dyn))
        scaler_static.fit(X_static.reshape(-1, X_static.shape[-1]))
        scaler_target.fit(train_y.reshape(-1, 1))
        
        # Apply scaling to training data
        X_dyn_scaled = scaler_dyn.transform(X_dyn.reshape(-1, n_dyn)).reshape(X_dyn.shape)
        X_static_scaled = scaler_static.transform(X_static.reshape(-1, X_static.shape[-1])).reshape(X_static.shape)
        
        train_X_scaled = np.concatenate([X_dyn_scaled, X_static_scaled], axis=2)
        train_y_scaled = scaler_target.transform(train_y.reshape(-1, 1)).flatten()
        
        # Apply scaling to test data
        with h5py.File(result['test_X_file'], 'r') as f:
            test_X = f['X'][:]
        with h5py.File(result['test_y_file'], 'r') as f:
            test_y = f['y'][:]
        
        X_dyn = test_X[:, :, :n_dyn]
        X_static = test_X[:, :, n_dyn:]
        
        X_dyn_scaled = scaler_dyn.transform(X_dyn.reshape(-1, n_dyn)).reshape(X_dyn.shape)
        X_static_scaled = scaler_static.transform(X_static.reshape(-1, X_static.shape[-1])).reshape(X_static.shape)
        
        test_X_scaled = np.concatenate([X_dyn_scaled, X_static_scaled], axis=2)
        test_y_scaled = scaler_target.transform(test_y.reshape(-1, 1)).flatten()
        
        scalers_dict = {
            'dyn_scaler': scaler_dyn,
            'static_scaler': scaler_static,
            'target_scaler': scaler_target
        }
    else:
        # Multi-window mode: separate long and short sequences
        logger.info(f"Mode: MULTI-WINDOW (long={default_processor_config.seq_length_d - default_processor_config.seq_length_a}days + short={default_processor_config.seq_length_a}days)")
        LONG_STEPS = default_processor_config.n_windows
        
        X_long = train_X[:, :LONG_STEPS, :n_dyn]
        X_short = train_X[:, LONG_STEPS:, :n_dyn]
        X_static = train_X[:, :, n_dyn:]
        
        scaler_long = StandardScaler()
        scaler_short = StandardScaler()
        scaler_static = StandardScaler()
        scaler_target = StandardScaler()
        
        scaler_long.fit(X_long.reshape(-1, n_dyn))
        scaler_short.fit(X_short.reshape(-1, n_dyn))
        scaler_static.fit(X_static.reshape(-1, X_static.shape[-1]))
        scaler_target.fit(train_y.reshape(-1, 1))
        
        # Apply scaling to training data
        X_long_scaled = scaler_long.transform(X_long.reshape(-1, n_dyn)).reshape(X_long.shape)
        X_short_scaled = scaler_short.transform(X_short.reshape(-1, n_dyn)).reshape(X_short.shape)
        X_static_scaled = scaler_static.transform(X_static.reshape(-1, X_static.shape[-1])).reshape(X_static.shape)

        X_dyn_scaled = np.concatenate([X_long_scaled, X_short_scaled], axis=1)
        train_X_scaled = np.concatenate([X_dyn_scaled, X_static_scaled], axis=2)

        train_y_scaled = scaler_target.transform(train_y.reshape(-1, 1)).flatten()
        
        # Apply scaling to test data
        with h5py.File(result['test_X_file'], 'r') as f:
            test_X = f['X'][:]
        with h5py.File(result['test_y_file'], 'r') as f:
            test_y = f['y'][:]
        
        X_long = test_X[:, :LONG_STEPS, :n_dyn]
        X_short = test_X[:, LONG_STEPS:, :n_dyn]
        X_static = test_X[:, :, n_dyn:]
        
        X_long_scaled = scaler_long.transform(X_long.reshape(-1, n_dyn)).reshape(X_long.shape)
        X_short_scaled = scaler_short.transform(X_short.reshape(-1, n_dyn)).reshape(X_short.shape)
        X_static_scaled = scaler_static.transform(X_static.reshape(-1, X_static.shape[-1])).reshape(X_static.shape)

        X_dyn_scaled_test = np.concatenate([X_long_scaled, X_short_scaled], axis=1)
        test_X_scaled = np.concatenate([X_dyn_scaled_test, X_static_scaled], axis=2)
        test_y_scaled = scaler_target.transform(test_y.reshape(-1, 1)).flatten()
        
        scalers_dict = {
            'long_dyn_scaler': scaler_long,
            'short_dyn_scaler': scaler_short,
            'static_scaler': scaler_static,
            'target_scaler': scaler_target
        }
    
    # Save final HDF5 in main_ensemble.py format
    data_dir = os.path.join(work_dir, default_path_config.data_subdir)
    os.makedirs(data_dir, exist_ok=True)
    
    train_output = os.path.join(data_dir, default_processor_config.is_train_output_file)
    test_output = os.path.join(data_dir, default_processor_config.is_test_output_file)
    scalers_output = os.path.join(data_dir, default_processor_config.is_scalers_file)
    
    with h5py.File(train_output, 'w') as f:
        f.create_dataset('train_X', data=train_X, compression='gzip')
        f.create_dataset('train_X_scaled', data=train_X_scaled, compression='gzip')
        f.create_dataset('train_y', data=train_y, compression='gzip')
        f.create_dataset('train_y_scaled', data=train_y_scaled, compression='gzip')
        # Copy metadata from source file if it exists
        if result['train_X_file'] and os.path.exists(result['train_X_file']):
            try:
                with h5py.File(result['train_X_file'], 'r') as src:
                    if 'train_meta' in src:
                        train_meta = f.create_group('train_meta')
                        for key in src['train_meta'].keys():
                            train_meta.create_dataset(key, data=src['train_meta'][key][()])
            except Exception as e:
                logger.warning(f"Could not copy train_meta: {e}")
    
    with h5py.File(test_output, 'w') as f:
        f.create_dataset('test_X', data=test_X, compression='gzip')
        f.create_dataset('test_X_scaled', data=test_X_scaled, compression='gzip')
        f.create_dataset('test_y', data=test_y, compression='gzip')
        f.create_dataset('test_y_scaled', data=test_y_scaled, compression='gzip')
        # Copy metadata from source file if it exists
        if result['test_X_file'] and os.path.exists(result['test_X_file']):
            try:
                with h5py.File(result['test_X_file'], 'r') as src:
                    if 'test_meta' in src:
                        test_meta = f.create_group('test_meta')
                        for key in src['test_meta'].keys():
                            test_meta.create_dataset(key, data=src['test_meta'][key][()])
            except Exception as e:
                logger.warning(f"Could not copy test_meta: {e}")
    
    with open(scalers_output, 'wb') as f:
        pickle.dump(scalers_dict, f)
    
    logger.info(f"✓ Saved to {data_dir}")
    return result['train_samples'], result['test_samples']


def prepare_oos_dataset():
    """Preprocess OOS (Out-of-Sample) dataset - 341 unseen stations"""
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING OOS (Out-of-Sample) DATASET")
    logger.info("=" * 80)
    
    username = os.getenv('USER', 'default_user')
    work_dir = os.path.join(default_path_config.work_dir, username)
    processor_dir = os.path.join(work_dir, default_path_config.processor_subdir)
    output_dir = os.path.expanduser('~/.cache/groundwater_data')
    
    processor = GroundwaterDataProcessor(
        os.path.join(processor_dir, 'GD_timeseries'),
        os.path.join(processor_dir, 'CHESS_timeseries'),
        os.path.join(processor_dir, default_processor_config.oos_station_info_file),
        default_processor_config.dyn_features,
        default_processor_config.add_pet_sin
    )
    
    station_nos = processor.station_info['No'].tolist()
    logger.info(f"Processing {len(station_nos)} OOS stations...")
    
    result = processor.prepare_data(
        station_nos,
        train_start=None,
        train_end=None,
        test_start=default_processor_config.test_start,
        test_end=default_processor_config.test_end,
        seq_length_d=default_processor_config.seq_length_d,
        seq_length_a=default_processor_config.seq_length_a,
        n_windows=default_processor_config.n_windows,
        output_dir=output_dir
    )
    
    # Check if we have test data
    if result['test_samples'] == 0:
        logger.error("No valid OOS test samples found!")
        return 0
    
    # Save raw OOS data (no scaling needed)
    with h5py.File(result['test_X_file'], 'r') as src:
        test_X = src['X'][:]
    with h5py.File(result['test_y_file'], 'r') as src:
        test_y = src['y'][:]
    
    data_dir = os.path.join(work_dir, default_path_config.data_subdir)
    oos_output = os.path.join(data_dir, default_processor_config.oos_test_output_file)
    
    with h5py.File(oos_output, 'w') as f:
        f.create_dataset('test_X', data=test_X, compression='gzip')
        f.create_dataset('test_y', data=test_y, compression='gzip')
        # Copy metadata from source file if it exists
        if result['test_X_file'] and os.path.exists(result['test_X_file']):
            try:
                with h5py.File(result['test_X_file'], 'r') as src:
                    if 'test_meta' in src:
                        test_meta = f.create_group('test_meta')
                        for key in src['test_meta'].keys():
                            test_meta.create_dataset(key, data=src['test_meta'][key][()])
            except Exception as e:
                logger.warning(f"Could not copy OOS test_meta: {e}")
    
    logger.info(f"✓ Saved to {data_dir}")
    return result['test_samples']


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("GROUNDWATER DATA PREPROCESSING")
    logger.info("=" * 80)
    
    try:
        train_n, test_n = prepare_is_dataset()
        oos_n = prepare_oos_dataset()
        
        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"IS Train: {train_n} samples")
        logger.info(f"IS Test: {test_n} samples")
        logger.info(f"OOS Test: {oos_n} samples")
        logger.info("\nReady for main_ensemble.py")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
