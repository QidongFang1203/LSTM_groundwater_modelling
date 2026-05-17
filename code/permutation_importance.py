"""
Permutation Feature Importance (PFI) Analysis for LSTM Ensemble Model

Evaluates feature importance by permuting each feature across samples
and measuring the degradation of ensemble model performance on OOS data.

Two permutation methods:
  a) shuffle:    randomly permute feature values across all samples (24 timesteps move together)
  b) half_swap:  randomly split samples in half and swap feature values between halves

Output: one CSV per metric.
        Rows = stations, Columns = [No, Baseline, feat_1, feat_2, ..., feat_26].
        Values = M-repetition averaged metric.
        Baseline (unpermuted) metric is included as the second column for easy ratio computation.
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import LSTM, GroundwaterDataset
from metrics import EvaluationMetrics
from config import (default_model_config, default_ensemble_config,
                    default_path_config, default_processor_config)
from main_ensemble import get_base_paths, setup_logging, read_group

warnings.filterwarnings('ignore')

# ======================== USER CONFIGURATION ========================
M = 10                      # Number of permutation repetitions
RANDOM_SEED_BASE = 2026    # Base seed for reproducibility
DATA_MODE = 'TRAIN'          # 'OOS' or 'IS' or 'TRAIN'
PERMUTE_METHOD = 'full_shuffle'  # 'shuffle' or 'half_swap' or 'full_shuffle'
PERMUTE_MODE = 'individual'  # 'individual' or 'group'
#   individual : permute one feature at a time (original behaviour)
#   group      : permute all features within a category simultaneously
#                using the same permutation index, so that intra-group
#                multivariate structure is preserved while inter-sample
#                correspondence is broken for the whole group at once.

# Feature groups for group-wise permutation (0-based indices into feature_names)
FEATURE_GROUPS = {
    'Meteorology':      list(range(0, 9)),
    'Topography':       list(range(9, 12)),
    'Soil':             list(range(12, 20)),
    'Hydrogeology':     list(range(20, 23)),
    'Water management': list(range(23, 26)),
}
# ====================================================================


def get_feature_names(n_total_features, n_dyn_features, static_col_names=None):
    """Construct human-readable feature name list."""
    base_dyn = [f for f in default_processor_config.dyn_features if f != 'PET_sin']
    dyn_names = list(base_dyn)
    if default_processor_config.add_pet_sin:
        dyn_names.append('PET_sin')

    n_static = n_total_features - n_dyn_features
    if static_col_names is not None and len(static_col_names) == n_static:
        static_names = list(static_col_names)
    else:
        static_names = [f'static_{i}' for i in range(n_static)]

    return dyn_names + static_names


def load_data_and_scale(paths, mode='OOS'):
    """
    Load test data and apply IS scalers.

    For OOS: loads raw test_X, scales with IS scalers.
    For IS:  loads pre-scaled test_X_scaled directly.

    Returns:
        X_scaled, y_raw, y_scaled, info, scaler_target, DYN_FEAT_COUNT
    """
    scaler_path = os.path.join(paths['data_dir'], default_processor_config.is_scalers_file)
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    scaler_target = scalers['target_scaler']
    scaler_static = scalers['static_scaler']
    use_daily_mode = 'dyn_scaler' in scalers

    base_dyn = [f for f in default_processor_config.dyn_features if f != 'PET_sin']
    DYN_FEAT_COUNT = len(base_dyn) + (1 if default_processor_config.add_pet_sin else 0)

    if mode == 'TRAIN':
        # train data already has scaled arrays
        train_path = os.path.join(paths['data_dir'], default_processor_config.is_train_output_file)
        with h5py.File(train_path, 'r') as f:
            X_scaled = f['train_X_scaled'][:].astype(default_processor_config.data_dtype)
            y_raw = f['train_y'][:].astype(default_processor_config.data_dtype)
            y_scaled = f['train_y_scaled'][:].astype(default_processor_config.data_dtype)
            grp = f.get('train_meta')
            info = (read_group(grp).get('info', np.arange(len(y_raw)).reshape(-1, 1))
                    if grp else np.arange(len(y_raw)).reshape(-1, 1))
        return X_scaled, y_raw, y_scaled, info, scaler_target, DYN_FEAT_COUNT

    if mode == 'IS':
        # IS test data already has scaled arrays
        is_test_path = os.path.join(paths['data_dir'], default_processor_config.is_test_output_file)
        with h5py.File(is_test_path, 'r') as f:
            X_scaled = f['test_X_scaled'][:].astype(default_processor_config.data_dtype)
            y_raw = f['test_y'][:].astype(default_processor_config.data_dtype)
            y_scaled = f['test_y_scaled'][:].astype(default_processor_config.data_dtype)
            grp = f.get('test_meta')
            info = (read_group(grp).get('info', np.arange(len(y_raw)).reshape(-1, 1))
                    if grp else np.arange(len(y_raw)).reshape(-1, 1))
        return X_scaled, y_raw, y_scaled, info, scaler_target, DYN_FEAT_COUNT

    # ---------- OOS ----------
    oos_path = os.path.join(paths['data_dir'], default_processor_config.oos_test_output_file)
    with h5py.File(oos_path, 'r') as f:
        X_raw = f['test_X'][:].astype(default_processor_config.data_dtype)
        y_raw = f['test_y'][:].astype(default_processor_config.data_dtype)
        grp = f.get('test_meta')
        info = (read_group(grp).get('info', np.arange(len(y_raw)).reshape(-1, 1))
                if grp else np.arange(len(y_raw)).reshape(-1, 1))

    if use_daily_mode:
        scaler_dyn = scalers['dyn_scaler']
        X_dyn = X_raw[:, :, :DYN_FEAT_COUNT]
        X_static = X_raw[:, :, DYN_FEAT_COUNT:]
        X_dyn_s = scaler_dyn.transform(X_dyn.reshape(-1, DYN_FEAT_COUNT)).reshape(X_dyn.shape)
        X_static_s = scaler_static.transform(
            X_static.reshape(-1, X_static.shape[-1])).reshape(X_static.shape)
        X_scaled = np.concatenate([X_dyn_s, X_static_s], axis=2)
    else:
        scaler_long = scalers['long_dyn_scaler']
        scaler_short = scalers['short_dyn_scaler']
        LONG_STEPS = default_processor_config.n_windows
        X_long = X_raw[:, :LONG_STEPS, :DYN_FEAT_COUNT]
        X_short = X_raw[:, LONG_STEPS:, :DYN_FEAT_COUNT]
        X_static = X_raw[:, :, DYN_FEAT_COUNT:]
        X_long_s = scaler_long.transform(X_long.reshape(-1, DYN_FEAT_COUNT)).reshape(X_long.shape)
        X_short_s = scaler_short.transform(X_short.reshape(-1, DYN_FEAT_COUNT)).reshape(X_short.shape)
        X_static_s = scaler_static.transform(
            X_static.reshape(-1, X_static.shape[-1])).reshape(X_static.shape)
        dyn_combined = np.concatenate([X_long_s, X_short_s], axis=1)
        X_scaled = np.concatenate([dyn_combined, X_static_s], axis=2)

    y_scaled = scaler_target.transform(y_raw.reshape(-1, 1)).flatten()
    return X_scaled, y_raw, y_scaled, info, scaler_target, DYN_FEAT_COUNT


def load_ensemble_models(paths, input_size):
    """Load all pre-trained ensemble models."""
    loss_func = default_model_config.loss_function
    data_no = default_processor_config.model_name
    num_seeds = default_ensemble_config.num_seeds
    model_no = f'{data_no}_{loss_func}_ensemble{num_seeds}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    for seed in default_ensemble_config.seeds:
        model = LSTM(
            input_size=input_size,
            hidden_size=default_model_config.hidden_size,
            num_layers=default_model_config.num_layers,
            dropout_rate=default_model_config.dropout_rate
        )
        model_path = os.path.join(paths['model_dir'], f'{model_no}_seed{seed}.pth')
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        models.append(model)
        logging.info(f"Loaded model: seed {seed}")
    return models, device


def predict_ensemble(models, X_scaled, y_scaled, info, device, scaler_target, batch_size):
    """
    Run all ensemble members, return inverse-transformed ensemble-average prediction.
    """
    ensemble_preds = []
    dataset = GroundwaterDataset(X_scaled, y_scaled, info)
    loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=False)

    for model in models:
        preds = []
        with torch.no_grad():
            for sequences, _, _ in loader:
                sequences = sequences.to(device)
                outputs = model(sequences)
                preds.extend(outputs.cpu().numpy().flatten())
        pred_orig = scaler_target.inverse_transform(
            np.array(preds).reshape(-1, 1)).flatten()
        ensemble_preds.append(pred_orig)

    return np.mean(ensemble_preds, axis=0)


def compute_station_metrics(observations, predictions, station_ids):
    """Compute 6 performance metrics for each station."""
    calc = EvaluationMetrics()
    unique_stations = np.unique(station_ids)
    station_metrics = {}
    for sid in unique_stations:
        mask = (station_ids == sid)
        obs = observations[mask]
        pred = predictions[mask]
        if len(obs) > 1:
            station_metrics[sid] = calc.calculate_all_metrics(obs, pred)
    return station_metrics


# ======================== Permutation methods ========================

def permute_shuffle(X_scaled, feature_idx, rng):
    """Method a: Randomly shuffle feature across all samples (in-place safe)."""
    n = X_scaled.shape[0]
    perm = rng.permutation(n)
    original_col = X_scaled[:, :, feature_idx].copy()
    X_scaled[:, :, feature_idx] = original_col[perm]
    return original_col  # return backup for restoration


def permute_half_swap(X_scaled, feature_idx, rng):
    """Method b: Randomly split samples in half and swap feature values."""
    n = X_scaled.shape[0]
    indices = rng.permutation(n)
    half = n // 2
    idx_a = indices[:half]
    idx_b = indices[half:2 * half]
    original_col = X_scaled[:, :, feature_idx].copy()
    X_scaled[idx_a, :, feature_idx] = original_col[idx_b]
    X_scaled[idx_b, :, feature_idx] = original_col[idx_a]
    return original_col


def permute_full_shuffle(X_scaled, feature_idx, rng):
    """Method c: Shuffle feature across ALL samples × ALL timesteps.

    Unlike permute_shuffle (which keeps the 24-step sequence of each sample
    intact and only re-assigns sequences between samples), this method pools
    all n_samples × n_steps values for the feature, shuffles them globally,
    and writes them back.  This destroys both the cross-sample relationship
    and the within-sample temporal structure simultaneously.

    For static features (constant across timesteps) the result is equivalent
    to permute_shuffle.
    """
    original_col = X_scaled[:, :, feature_idx].copy()   # (n_samples, n_steps)
    flat = original_col.flatten()
    rng.shuffle(flat)
    X_scaled[:, :, feature_idx] = flat.reshape(original_col.shape)
    return original_col


def restore_feature(X_scaled, feature_idx, original_col):
    """Restore original feature column after permutation."""
    X_scaled[:, :, feature_idx] = original_col


# ======================== Group permutation methods ========================

def permute_group_shuffle(X_scaled, feature_indices, rng):
    """Shuffle a group of features simultaneously using the SAME sample permutation.

    All features in the group share one permutation vector, so the multivariate
    structure within each sample is preserved while the cross-sample correspondence
    is broken for the whole group at once.
    """
    n = X_scaled.shape[0]
    perm = rng.permutation(n)
    originals = {}
    for idx in feature_indices:
        originals[idx] = X_scaled[:, :, idx].copy()
        X_scaled[:, :, idx] = originals[idx][perm]
    return originals


def permute_group_half_swap(X_scaled, feature_indices, rng):
    """Half-swap a group of features simultaneously using the SAME split."""
    n = X_scaled.shape[0]
    indices = rng.permutation(n)
    half = n // 2
    idx_a = indices[:half]
    idx_b = indices[half:2 * half]
    originals = {}
    for idx in feature_indices:
        originals[idx] = X_scaled[:, :, idx].copy()
        X_scaled[idx_a, :, idx] = originals[idx][idx_b]
        X_scaled[idx_b, :, idx] = originals[idx][idx_a]
    return originals


def permute_group_full_shuffle(X_scaled, feature_indices, rng):
    """Full-shuffle a group of features simultaneously using the SAME flat permutation.

    All features share one permutation of the flattened (n_samples × n_steps) space,
    breaking both cross-sample and within-sample temporal structure uniformly.
    """
    n_total = X_scaled.shape[0] * X_scaled.shape[1]
    perm = rng.permutation(n_total)
    originals = {}
    for idx in feature_indices:
        originals[idx] = X_scaled[:, :, idx].copy()
        X_scaled[:, :, idx] = originals[idx].flatten()[perm].reshape(originals[idx].shape)
    return originals


def restore_group(X_scaled, originals):
    """Restore all features in a group after group permutation."""
    for idx, col in originals.items():
        X_scaled[:, :, idx] = col


# ======================== Main ========================

def main():
    # ---- Config ----
    loss_func = default_model_config.loss_function
    data_no = default_processor_config.model_name
    num_seeds = default_ensemble_config.num_seeds
    model_no = f'{data_no}_{loss_func}_ensemble{num_seeds}'
    job_name = f'PFI_{DATA_MODE}_{model_no}'

    paths = get_base_paths()
    setup_logging(job_name)

    logging.info("=" * 80)
    logging.info("PERMUTATION FEATURE IMPORTANCE (PFI) ANALYSIS")
    logging.info(f"Model:       {model_no}")
    logging.info(f"Data mode:   {DATA_MODE}")
    logging.info(f"Method:      {PERMUTE_METHOD}")
    logging.info(f"M reps:      {M}")
    logging.info(f"Seed base:   {RANDOM_SEED_BASE}")
    logging.info("=" * 80)

    # ---- 1. Load data ----
    logging.info("Loading data and scalers ...")
    X_scaled, y_raw, y_scaled, info, scaler_target, DYN_FEAT_COUNT = \
        load_data_and_scale(paths, mode=DATA_MODE)

    n_samples, n_steps, n_features = X_scaled.shape
    station_ids = np.array([item[0] for item in info])
    unique_stations = np.unique(station_ids)
    logging.info(f"Data shape: {X_scaled.shape}  |  Stations: {len(unique_stations)}")
    logging.info(f"Permute mode: {PERMUTE_MODE}")

    # ---- 2. Feature names ----
    static_col_names = None
    try:
        processor_dir = os.path.join(paths['work_dir'], default_path_config.processor_subdir)
        oos_info_file = default_processor_config.oos_station_info_file if DATA_MODE == 'OOS' \
            else default_processor_config.is_station_info_file
        info_csv = os.path.join(processor_dir, oos_info_file)
        if os.path.exists(info_csv):
            cols = pd.read_csv(info_csv, nrows=0).columns.tolist()
            static_col_names = [c for c in cols if c != 'No']
    except Exception:
        pass
    feature_names = get_feature_names(n_features, DYN_FEAT_COUNT, static_col_names)
    logging.info(f"Features ({n_features}): {feature_names}")

    # ---- 3. Load ensemble models ----
    logging.info("Loading ensemble models ...")
    models, device = load_ensemble_models(paths, input_size=n_features)
    logging.info(f"Loaded {len(models)} models on {device}")

    batch_size = default_model_config.batch_size
    metric_names = default_ensemble_config.metrics

    # ---- 4. Output directory ----
    pfi_dir = os.path.join(paths['result_dir'], f'PFI_{DATA_MODE}_{model_no}')
    os.makedirs(pfi_dir, exist_ok=True)

    # ---- 5. Select permutation method ----
    permute_func_map = {
        'shuffle':      permute_shuffle,
        'half_swap':    permute_half_swap,
        'full_shuffle': permute_full_shuffle,
    }
    permute_group_func_map = {
        'shuffle':      permute_group_shuffle,
        'half_swap':    permute_group_half_swap,
        'full_shuffle': permute_group_full_shuffle,
    }
    if PERMUTE_METHOD not in permute_func_map:
        raise ValueError(f"Unknown PERMUTE_METHOD '{PERMUTE_METHOD}'. "
                         f"Use 'shuffle', 'half_swap', or 'full_shuffle'.")
    if PERMUTE_MODE not in ('individual', 'group'):
        raise ValueError(f"Unknown PERMUTE_MODE '{PERMUTE_MODE}'. "
                         f"Use 'individual' or 'group'.")
    permute_func = permute_func_map[PERMUTE_METHOD]
    permute_group_func = permute_group_func_map[PERMUTE_METHOD]

    # ---- 6. Baseline (no permutation) ----
    logging.info("Computing baseline metrics (no permutation) ...")
    baseline_pred = predict_ensemble(
        models, X_scaled, y_scaled, info, device, scaler_target, batch_size)
    baseline_station = compute_station_metrics(y_raw, baseline_pred, station_ids)
    logging.info("Baseline complete.")

    # ---- 7. PFI: permute each feature (or group) M times ----
    logging.info(f"\nRunning PFI with method='{PERMUTE_METHOD}', mode='{PERMUTE_MODE}' ...")

    if PERMUTE_MODE == 'individual':
        # ------------------------------------------------------------------
        # Individual mode: permute one feature at a time (original behaviour)
        # ------------------------------------------------------------------
        col_names = feature_names   # output column names = individual feature names

        # Storage: metric -> feature_name -> station -> list of M values
        all_results = {
            metric: {f_name: {sid: [] for sid in unique_stations}
                     for f_name in col_names}
            for metric in metric_names
        }

        for f_idx in range(n_features):
            f_name = feature_names[f_idx]
            logging.info(f"  Feature {f_idx + 1}/{n_features}: {f_name}")

            for m in range(M):
                rng = np.random.default_rng(RANDOM_SEED_BASE + f_idx * M + m)

                original_col = permute_func(X_scaled, f_idx, rng)
                ensemble_avg = predict_ensemble(
                    models, X_scaled, y_scaled, info,
                    device, scaler_target, batch_size)
                restore_feature(X_scaled, f_idx, original_col)

                st_metrics = compute_station_metrics(y_raw, ensemble_avg, station_ids)
                for sid, mdict in st_metrics.items():
                    for metric in metric_names:
                        all_results[metric][f_name][sid].append(mdict[metric])

                if (m + 1) % 10 == 0:
                    logging.info(f"    rep {m + 1}/{M}")

    else:
        # ------------------------------------------------------------------
        # Group mode: permute all features in a category simultaneously.
        # All features in the group share the same permutation index so that
        # intra-group multivariate structure is preserved.  Output columns
        # are group names (not individual feature names).
        # ------------------------------------------------------------------
        col_names = list(FEATURE_GROUPS.keys())   # one column per group

        # Storage: metric -> group_name -> station -> list of M values
        all_results = {
            metric: {g_name: {sid: [] for sid in unique_stations}
                     for g_name in col_names}
            for metric in metric_names
        }

        for g_idx, (g_name, g_indices) in enumerate(FEATURE_GROUPS.items()):
            logging.info(f"  Group {g_idx + 1}/{len(FEATURE_GROUPS)}: '{g_name}' "
                         f"(features {g_indices[0]}–{g_indices[-1]})")

            for m in range(M):
                rng = np.random.default_rng(RANDOM_SEED_BASE + g_idx * M + m)

                originals = permute_group_func(X_scaled, g_indices, rng)
                ensemble_avg = predict_ensemble(
                    models, X_scaled, y_scaled, info,
                    device, scaler_target, batch_size)
                restore_group(X_scaled, originals)

                st_metrics = compute_station_metrics(y_raw, ensemble_avg, station_ids)
                for sid, mdict in st_metrics.items():
                    for metric in metric_names:
                        all_results[metric][g_name][sid].append(mdict[metric])

                if (m + 1) % 10 == 0:
                    logging.info(f"    rep {m + 1}/{M}")

    # ---- 8. Save: one CSV per metric ----
    # Individual mode: columns = [No, Baseline, feat_0, ..., feat_25]
    # Group mode:      columns = [No, Baseline, Meteorology, Topography, ...]
    for metric in metric_names:
        rows = []
        for sid in unique_stations:
            row = {'No': sid}
            row['Baseline'] = baseline_station[sid][metric] if sid in baseline_station else np.nan
            for col in col_names:
                vals = all_results[metric][col][sid]
                row[col] = np.nanmean(vals) if vals else np.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        suffix = f'group_{PERMUTE_METHOD}' if PERMUTE_MODE == 'group' else PERMUTE_METHOD
        out_path = os.path.join(pfi_dir, f'PFI_{suffix}_{metric}.csv')
        df.to_csv(out_path, index=False)
        logging.info(f"Saved: {out_path}")

    # ---- Done ----
    logging.info("\n" + "=" * 80)
    logging.info("PFI ANALYSIS COMPLETED")
    logging.info(f"Results: {pfi_dir}")
    logging.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"PFI failed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
