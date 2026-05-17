#!/usr/bin/env python
"""
Unified groundwater data processor with HDF5 streaming.
Handles 3600-day daily sequences without memory overflow.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import h5py
import os
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import curve_fit

from config import default_processor_config

np.random.seed(42)
warnings.filterwarnings('ignore')


def sinusoidal_func(x, amplitude, phase, offset):
    """Sinusoidal function for fitting: amplitude * sin(2*pi*x + phase) + offset"""
    return amplitude * np.sin(2 * np.pi * x + phase) + offset


class GroundwaterDataProcessor:
    """Groundwater data processor - creates sequences and handles both IS/OOS datasets"""
    
    def __init__(self, gw_folder: str, met_folder: str, station_info_path: str, dyn_features: List[str], add_pet_sin: bool = False):
        self.gw_folder = Path(gw_folder)
        self.met_folder = Path(met_folder)
        self.station_info = pd.read_csv(station_info_path)
        self.dyn_features = dyn_features
        self.add_pet_sin = add_pet_sin
        self.pet_sin_params = {}  # Cache fitted parameters per station

    def fit_pet_seasonal_sin(self, station_no: str) -> Tuple[float, float, float]:
        """Fit sinusoidal function to PET data for seasonal reference"""
        if station_no in self.pet_sin_params:
            return self.pet_sin_params[station_no]
        
        _, met_df = self.load_station_data(station_no)
        if met_df is None or 'PET' not in met_df.columns:
            # Default parameters if no data
            return 1.0, 0.0, 0.0
        
        # Use available PET data
        pet_data = met_df['PET'].dropna()
        if len(pet_data) < 365:  # Need at least a year
            return np.std(pet_data) if len(pet_data) > 0 else 1.0, 0.0, np.mean(pet_data) if len(pet_data) > 0 else 0.0
        
        # Create day of year as fraction
        dates = met_df.index
        day_of_year = dates.dayofyear / 365.25
        
        try:
            # Fit sinusoidal function
            popt, _ = curve_fit(sinusoidal_func, day_of_year, pet_data, 
                              p0=[np.std(pet_data), 0, np.mean(pet_data)],
                              bounds=([0, -np.pi, -np.inf], [np.inf, np.pi, np.inf]))
            amplitude, phase, offset = popt
        except:
            # Fallback to simple statistics
            amplitude = np.std(pet_data)
            phase = 0.0
            offset = np.mean(pet_data)
        
        self.pet_sin_params[station_no] = (amplitude, phase, offset)
        return amplitude, phase, offset

    def load_station_data(self, station_no: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load GW and meteorological data for a station"""
        gw_file = self.gw_folder / f"{station_no}_timeseries.csv"
        met_file = self.met_folder / f"{station_no}.csv"
        
        if not gw_file.exists() or not met_file.exists():
            return None, None
        
        gw_df = pd.read_csv(gw_file)
        gw_df['date'] = pd.to_datetime(gw_df['date'])
        gw_df = gw_df.set_index('date').sort_index()
        
        met_df = pd.read_csv(met_file)
        met_df['Date'] = pd.to_datetime(met_df['Date'])
        met_df = met_df.set_index('Date').sort_index()
        
        return gw_df, met_df

    def create_sequences(self, station_no: str, seq_length_d: int, seq_length_a: int, 
                        n_windows: int) -> Optional[Dict[str, Any]]:
        """Create sequences (daily or multi-window mode)"""
        gw_df, met_df = self.load_station_data(station_no)
        if gw_df is None or met_df is None:
            return None

        # Get original dyn_features without PET_sin
        base_dyn_features = [f for f in self.dyn_features if f != 'PET_sin']
        
        # Fit PET seasonal sin if needed and create as Series for proper indexing
        pet_sin_series = None
        if self.add_pet_sin and 'PET' in met_df.columns:
            amplitude, phase, offset = self.fit_pet_seasonal_sin(station_no)
            day_of_year = met_df.index.dayofyear / 365.25
            pet_sin_values = sinusoidal_func(day_of_year, amplitude, phase, offset)
            # Create a Series with the same index as met_df for proper alignment
            pet_sin_series = pd.Series(pet_sin_values, index=met_df.index)

        sequences = []
        targets = []
        dates = []
        use_daily_mode = (seq_length_a == seq_length_d)

        for gw_date, gw_value in gw_df.iterrows():
            if use_daily_mode:
                # Daily mode: 3600 consecutive days
                seq_start = gw_date - timedelta(days=seq_length_d - 1)
                met_seq = met_df[seq_start:gw_date]
                
                if len(met_seq) == seq_length_d and not met_seq.isnull().any().any() and not pd.isna(gw_value['GD']):
                    met_data = met_seq[base_dyn_features].values
                    if self.add_pet_sin and pet_sin_series is not None:
                        pet_sin_seq = pet_sin_series[seq_start:gw_date].values.reshape(-1, 1)
                        met_data = np.concatenate([met_data, pet_sin_seq], axis=1)
                    sequences.append(met_data)
                    targets.append(gw_value['GD'])
                    dates.append(gw_date)
            else:
                # Multi-window mode: aggregate windows
                seq_d_start = gw_date - timedelta(days=seq_length_d - 1)
                seq_d_end = gw_date - timedelta(days=seq_length_a)
                seq_a_start = gw_date - timedelta(days=seq_length_a - 1)

                met_d_seq = met_df[seq_d_start:seq_d_end]
                met_a_seq = met_df[seq_a_start:gw_date]

                if len(met_d_seq) == (seq_length_d - seq_length_a) and len(met_a_seq) == seq_length_a and \
                   not met_d_seq.isnull().any().any() and not met_a_seq.isnull().any().any() and not pd.isna(gw_value['GD']):
                    
                    met_d_data = met_d_seq[base_dyn_features].values
                    met_a_data = met_a_seq[base_dyn_features].values
                    
                    if self.add_pet_sin and pet_sin_series is not None:
                        # Extract pet_sin values aligned with the date indices
                        pet_sin_d = pet_sin_series[seq_d_start:seq_d_end].values.reshape(-1, 1)
                        pet_sin_a = pet_sin_series[seq_a_start:gw_date].values.reshape(-1, 1)
                        met_d_data = np.concatenate([met_d_data, pet_sin_d], axis=1)
                        met_a_data = np.concatenate([met_a_data, pet_sin_a], axis=1)
                    
                    n_feat = len(base_dyn_features) + (1 if self.add_pet_sin else 0)
                    met_d_month = met_d_data.reshape(n_windows, -1, n_feat).sum(axis=1)
                    met_a_month = met_a_data.reshape(n_windows, -1, n_feat).sum(axis=1)
                    met_month = np.concatenate([met_d_month, met_a_month], axis=0)
                    
                    sequences.append(met_month)
                    targets.append(gw_value['GD'])
                    dates.append(gw_date)

        if len(sequences) > 0:
            return {
                'sequences': np.array(sequences, dtype=default_processor_config.data_dtype),
                'targets': np.array(targets, dtype=default_processor_config.data_dtype),
                'dates': dates,
                'station_no': station_no
            }
        return None

    def prepare_data(self, station_nos: List[int], train_start: Optional[str], train_end: Optional[str],
                    test_start: str, test_end: str, seq_length_d: int, seq_length_a: int, 
                    n_windows: int, output_dir: str = None) -> Dict[str, Any]:
        """
        Prepare data with HDF5 streaming - writes directly to disk without 
        loading entire array to memory. Solves 37.5GB overflow problem.
        """
        if output_dir is None:
            output_dir = os.path.expanduser('~/.cache/groundwater_data')
        os.makedirs(output_dir, exist_ok=True)
        
        use_daily_mode = (seq_length_a == seq_length_d)
        seq_steps = seq_length_d if use_daily_mode else 2 * n_windows
        
        logging.info(f"Mode: {'DAILY' if use_daily_mode else 'MULTI-WINDOW'} (steps={seq_steps})")
        logging.info(f"HDF5 STREAMING with compression...")
        
        # Get base dyn features (without PET_sin)
        base_dyn_features = [f for f in self.dyn_features if f != 'PET_sin']
        n_dyn_features = len(base_dyn_features) + (1 if self.add_pet_sin else 0)
        
        # Count samples
        logging.info("Counting samples...")
        train_count = test_count = 0
        valid_stations = []
        
        for station_no in tqdm(station_nos, desc="Count"):
            station_row = self.station_info[self.station_info['No'] == station_no]
            if station_row.empty:
                continue
            
            full_data = self.create_sequences(station_no, seq_length_d, seq_length_a, n_windows)
            if full_data is None:
                continue
            
            dates = pd.to_datetime(full_data['dates'])
            test_mask = (dates <= pd.to_datetime(test_end)) & (dates >= pd.to_datetime(test_start))
            test_count += test_mask.sum()
            
            if train_start and train_end:
                train_mask = (dates <= pd.to_datetime(train_end)) & (dates >= pd.to_datetime(train_start))
                if train_mask.sum() > 0 and test_mask.sum() > 0:
                    train_count += train_mask.sum()
                    valid_stations.append(station_no)
            else:
                if test_mask.sum() > 0:
                    valid_stations.append(station_no)
        
        logging.info(f"Train: {train_count}, Test: {test_count}")
        
        # Count static features
        sample_station = self.station_info[self.station_info['No'] == valid_stations[0]]
        n_static_features = len([col for col in sample_station.columns if col not in ['No']])
        total_features = n_dyn_features + n_static_features
        
        # Create HDF5 files
        train_file = os.path.join(output_dir, 'train_X.h5')
        test_file = os.path.join(output_dir, 'test_X.h5')
        train_y_file = os.path.join(output_dir, 'train_y.h5')
        test_y_file = os.path.join(output_dir, 'test_y.h5')
        
        logging.info("Writing to HDF5...")
        
        # Pre-allocate info arrays to track station numbers
        if train_count > 0:
            train_info_list = []
        else:
            train_info_list = None
        test_info_list = []
        
        with h5py.File(train_file, 'w') as f_train_X, \
             h5py.File(test_file, 'w') as f_test_X, \
             h5py.File(train_y_file, 'w') as f_train_y, \
             h5py.File(test_y_file, 'w') as f_test_y:
            
            # Calculate appropriate chunk sizes (must not exceed data shape)
            train_chunk_size_0 = 1
            test_chunk_size_0 = 1
            train_y_chunk_size = min(1024, max(1, train_count)) if train_count > 0 else 1
            test_y_chunk_size = min(1024, max(1, test_count)) if test_count > 0 else 1
            
            # Create datasets only if there are samples
            if train_count > 0:
                train_X_ds = f_train_X.create_dataset('X', shape=(train_count, seq_steps, total_features),
                                                       dtype=default_processor_config.data_dtype,
                                                       chunks=(train_chunk_size_0, seq_steps, total_features),
                                                       compression='gzip', compression_opts=4)
                train_y_ds = f_train_y.create_dataset('y', shape=(train_count,),
                                                      dtype=default_processor_config.data_dtype,
                                                      chunks=(train_y_chunk_size,),
                                                      compression='gzip', compression_opts=4)
            else:
                logging.warning("No training samples, skipping train dataset creation")
                train_X_ds = None
                train_y_ds = None
            
            if test_count > 0:
                test_X_ds = f_test_X.create_dataset('X', shape=(test_count, seq_steps, total_features),
                                                    dtype=default_processor_config.data_dtype,
                                                    chunks=(test_chunk_size_0, seq_steps, total_features),
                                                    compression='gzip', compression_opts=4)
                test_y_ds = f_test_y.create_dataset('y', shape=(test_count,),
                                                   dtype=default_processor_config.data_dtype,
                                                   chunks=(test_y_chunk_size,),
                                                   compression='gzip', compression_opts=4)
            else:
                logging.error("No test samples found!")
                return None
            
            train_idx = test_idx = 0
            
            # Write streaming
            for station_no in tqdm(valid_stations, desc="Write"):
                station_row = self.station_info[self.station_info['No'] == station_no]
                static_cols = [col for col in station_row.columns if col not in ['No']]
                static_input = station_row[static_cols].values[0]
                
                full_data = self.create_sequences(station_no, seq_length_d, seq_length_a, n_windows)
                if full_data is None:
                    continue
                
                dates = pd.to_datetime(full_data['dates'])
                seqs = full_data['sequences'].astype(default_processor_config.data_dtype)
                targets = full_data['targets'].astype(default_processor_config.data_dtype)
                
                test_mask = (dates <= pd.to_datetime(test_end)) & (dates >= pd.to_datetime(test_start))
                
               # Write test
                if test_mask.sum() > 0:
                    test_seqs = seqs[test_mask]
                    test_targ = targets[test_mask]
                    static_exp = np.tile(static_input[np.newaxis, np.newaxis, :], (len(test_seqs), seq_steps, 1))
                    combined = np.concatenate([test_seqs, static_exp], axis=2)
                    
                    test_X_ds[test_idx:test_idx + len(test_seqs)] = combined
                    test_y_ds[test_idx:test_idx + len(test_seqs)] = test_targ
                    # Store station info
                    for _ in range(len(test_seqs)):
                        test_info_list.append(station_no)
                    test_idx += len(test_seqs)
                
                # Write train
                if train_start and train_end and train_X_ds is not None:
                    train_mask = (dates <= pd.to_datetime(train_end)) & (dates >= pd.to_datetime(train_start))
                    if train_mask.sum() > 0:
                        train_seqs = seqs[train_mask]
                        train_targ = targets[train_mask]
                        static_exp = np.tile(static_input[np.newaxis, np.newaxis, :], (len(train_seqs), seq_steps, 1))
                        combined = np.concatenate([train_seqs, static_exp], axis=2)
                        
                        train_X_ds[train_idx:train_idx + len(train_seqs)] = combined
                        train_y_ds[train_idx:train_idx + len(train_seqs)] = train_targ
                        # Store station info
                        for _ in range(len(train_seqs)):
                            train_info_list.append(station_no)
                        train_idx += len(train_seqs)
            
            # Save metadata (train_meta and test_meta groups)
            if train_info_list is not None and len(train_info_list) > 0:
                train_meta_group = f_train_X.create_group('train_meta')
                train_meta_group.create_dataset('info', data=np.array([[s] for s in train_info_list]))
            
            if len(test_info_list) > 0:
                test_meta_group = f_test_X.create_group('test_meta')
                test_meta_group.create_dataset('info', data=np.array([[s] for s in test_info_list]))
        
        logging.info(f"HDF5 created: {train_idx} train, {test_idx} test")
        
        return {
            'train_X_file': train_file if train_idx > 0 else None,
            'test_X_file': test_file if test_idx > 0 else None,
            'train_y_file': train_y_file if train_idx > 0 else None,
            'test_y_file': test_y_file if test_idx > 0 else None,
            'train_samples': train_idx,
            'test_samples': test_idx,
            'valid_stations': valid_stations,
            'seq_steps': seq_steps,
            'n_features': total_features
        }
