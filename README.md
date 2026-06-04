# Code and results for paper "Exploring the Generalisation Ability and Interpretability of an LSTM Model for Large-Sample Groundwater Predictions"

# LSTM Groundwater Modelling

Code and tabular outputs for modelling groundwater-level dynamics with Long Short-Term Memory (LSTM) neural networks.

This repository contains the modelling scripts, station metadata/static attributes, and performance summary tables used to evaluate LSTM groundwater predictions for in-sample stations and out-of-sample stations. The workflow trains an ensemble of PyTorch LSTM models on known stations and evaluates model transfer to unseen groundwater stations.

## What is included

```text
.
|-- code/
|   |-- config.py                  # Main configuration: dates, model settings, paths, features
|   |-- data_processor.py          # Builds groundwater/met sequence samples with HDF5 streaming
|   |-- prep_data.py               # Preprocesses IS and OOS datasets
|   |-- model.py                   # PyTorch LSTM, standard LSTM, ANN, and dataset wrappers
|   |-- trainer.py                 # Training loop and weighted NSE-style losses
|   |-- metrics.py                 # NSE, KGE, Pearson r, Spearman r, mean ratio, std ratio
|   |-- main_ensemble.py           # Trains and evaluates the LSTM ensemble
|   |-- permutation_importance.py  # Permutation feature importance analysis
|-- metadata/
|   |-- 636_IS_TRAIN_metadata.csv
|   |-- 341_OOS_metadata.csv
|   |-- 977_dynamic_mean_1961-2019.csv
|-- processor/
|   |-- LSTM_RND_636_IS_TRAIN_input.csv
|   |-- LSTM_RND_341_OOS_input.csv
|   |-- LSTM_ENV_636_IS_TRAIN_input.csv
|   |-- LSTM_ENV_341_OOS_input.csv
|-- results/
|   |-- *_Performance.csv
|   |-- DECIPHeR-GW_Performance.csv
```

The repository includes performance tables, but not the raw per-station groundwater and meteorological time-series files, generated HDF5 datasets, trained model weights, or run logs.

## Study design

- Training period: 1971-01-01 to 2014-12-31.
- Test period: 2015-01-01 to 2019-12-31.
- In-sample dataset: 636 stations used for model training and in-sample testing.
- Out-of-sample dataset: 341 unseen stations used for transfer evaluation.
- Dynamic meteorological inputs: `P`, `PET`, `TAS`, `DTR`, `HUSS`, `WIND`, `PSURF`, `RLDS`, `RSDS`.
- Static catchment/station inputs: elevation, slope, HAND, soil properties, transmissivity/storage proxies, population density, irrigation intensity, groundwater abstraction, aquifer class, and soil classes.
- Main model: PyTorch LSTM with 256 hidden units, dropout 0.4, batch size 512, 50 epochs, and a 10-member seed ensemble.
- Evaluation metrics: NSE, KGE, Pearson correlation, Spearman correlation, mean ratio, and standard deviation ratio.

The default sequence setting is `24`: the code aggregates meteorological information from a 3600-day look-back period into 24 time steps, with 12 windows for the earlier long-term period and 12 windows for the most recent annual period.

## Model variants and output files

The result filenames encode the dataset, model-input type, and ensemble setting:

- `636_IS_*`: in-sample station results.
- `341_OOS_*`: out-of-sample station results.
- `RND`: ranked/coded static input version.
- `ENV`: environmental/static descriptor input version.
- `seed42`: single model trained with seed 42.
- `ensemble10`: average prediction from 10 seeds, 42 to 51.
- `DECIPHeR-GW_Performance.csv`: benchmark comparison table against DECIPHeR-GW simulations.

For example, `results/341_OOS_ENV_ensemble10_Performance.csv` contains out-of-sample performance for the 10-member LSTM ensemble using the environmental/static descriptor input set.

## Python environment

Python 3.9 or later is recommended. The main dependencies are:

```bash
pip install numpy pandas scipy scikit-learn h5py tqdm torch
```

Install the PyTorch build appropriate for your CPU/GPU environment from the official PyTorch instructions if CUDA support is required.

## Reproducing the workflow

The scripts currently assume a Linux/HPC-style working directory defined in `code/config.py`:

```python
work_dir = "/user/work"
```

At run time, outputs are written under:

```text
/user/work/$USER/data/
/user/work/$USER/models/
/user/work/$USER/results/
/user/work/$USER/logs/
```

Edit `PathConfig.work_dir` in `code/config.py` if running on another machine.

### 1. Prepare required raw input folders

To rerun preprocessing, arrange the raw files as:

```text
/user/work/$USER/processor/
|-- GD_timeseries/
|   |-- <station_no>_timeseries.csv   # columns include date and GD
|-- CHESS_timeseries/
|   |-- <station_no>.csv              # columns include Date and meteorological variables
|-- LSTM_RND_636_IS_TRAIN_input.csv
|-- LSTM_RND_341_OOS_input.csv
```

If using the `ENV` input version, update `DataProcessorConfig` in `code/config.py` so the station-info filenames point to the `LSTM_ENV_*` files.

### 2. Preprocess data

```bash
python code/prep_data.py
```

This creates HDF5 train/test datasets and scalers under `/user/work/$USER/data/`.

### 3. Train and evaluate the ensemble

```bash
python code/main_ensemble.py
```

This trains the LSTM ensemble, saves model checkpoints under `/user/work/$USER/models/`, and writes Train/IS and OOS result CSVs under `/user/work/$USER/results/`.

### 4. Run permutation feature importance

```bash
python code/permutation_importance.py
```

Adjust the user configuration block at the top of `code/permutation_importance.py` to choose `TRAIN`, `IS`, or `OOS`, the permutation method, and individual or grouped feature permutations.

## Reading the included results

The included `results/*.csv` files can be inspected directly without rerunning the model. Key columns include:

- `No`: station identifier.
- `Wavelet transform`: groundwater dynamic class used in the analysis.
- `NSE_*`, `KGE_*`, `Pearson_r_*`, `Spearman_r_*`, `Mean_ratio_*`, `STD_ratio_*`: model performance metrics.
- `Sample Count`: number of observations used for station-level evaluation.
- environmental and hydrogeological attributes used for interpretation.

## Notes on reproducibility

The repository is intended to make the model code and tabular outputs transparent. Full reruns require the raw groundwater observations and CHESS meteorological time series, which are not included here. The current scripts are configured for the author's HPC file layout, so paths may need to be edited before reuse.

## Citation

If you use this code or results, please cite the associated manuscript or thesis chapter when available.
If you have questions, please send an email to: qidong.fang@bristol.ac.uk

