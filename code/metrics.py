"""
Evaluation metrics for model performance assessment
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple, List


class EvaluationMetrics:
    """Calculate various performance metrics for hydrological predictions"""

    @staticmethod
    def nse(observations: np.ndarray, predictions: np.ndarray) -> float:
        """
        Nash-Sutcliffe Efficiency (NSE)
        
        NSE = 1 - (sum((obs - pred)^2) / sum((obs - mean(obs))^2))
        Range: (-∞, 1], 1 = perfect prediction
        
        Args:
            observations: Observed values
            predictions: Predicted values
            
        Returns:
            NSE value
        """
        obs_mean = np.mean(observations)
        numerator = np.sum((observations - predictions) ** 2)
        denominator = np.sum((observations - obs_mean) ** 2)
        
        if denominator == 0:
            return np.nan
        
        return 1 - (numerator / denominator)

    @staticmethod
    def kge(observations: np.ndarray, predictions: np.ndarray, eps: float = 1e-6) -> float:
        """
        Kling-Gupta Efficiency (KGE)
        
        KGE = 1 - sqrt((r-1)^2 + (α-1)^2 + (β-1)^2)
        where:
            r = Pearson correlation coefficient
            α = ratio of standard deviations (std(pred) / std(obs))
            β = ratio of means (mean(pred) / mean(obs))
        
        Range: (-∞, 1], 1 = perfect prediction
        
        Args:
            observations: Observed values
            predictions: Predicted values
            eps: Small value to avoid division by zero
            
        Returns:
            KGE value
        """
        # Avoid division by zero
        if len(observations) < 2:
            return np.nan
        
        # Pearson correlation
        r, _ = pearsonr(observations, predictions)
        
        # Standard deviation ratio
        obs_std = np.std(observations)
        pred_std = np.std(predictions)
        
        if obs_std == 0:
            alpha = np.nan
        else:
            alpha = pred_std / (obs_std + eps)
        
        # Mean ratio
        obs_mean = np.mean(observations)
        pred_mean = np.mean(predictions)
        
        if obs_mean == 0:
            beta = np.nan
        else:
            beta = pred_mean / (obs_mean + eps)
        
        # Calculate KGE
        kge_value = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        
        return kge_value

    @staticmethod
    def pearson_r(observations: np.ndarray, predictions: np.ndarray) -> float:
        """
        Pearson correlation coefficient
        
        Range: [-1, 1], 1 = perfect positive correlation
        
        Args:
            observations: Observed values
            predictions: Predicted values
            
        Returns:
            Correlation coefficient
        """
        if len(observations) < 2:
            return np.nan
        
        r, _ = pearsonr(observations, predictions)
        return r

    @staticmethod
    def spearman_r(observations: np.ndarray, predictions: np.ndarray) -> float:
        """
        Spearman rank correlation coefficient
        
        Range: [-1, 1], 1 = perfect rank correlation
        
        Args:
            observations: Observed values
            predictions: Predicted values
            
        Returns:
            Spearman correlation coefficient
        """
        if len(observations) < 2:
            return np.nan
        
        r, _ = spearmanr(observations, predictions)
        return r

    @staticmethod
    def mean_ratio(observations: np.ndarray, predictions: np.ndarray) -> float:
        """
        Ratio of predicted mean to observed mean
        
        Mean Ratio = mean(predictions) / mean(observations)
        Ideal value: 1.0
        
        Args:
            observations: Observed values
            predictions: Predicted values
            
        Returns:
            Mean ratio
        """
        obs_mean = np.mean(observations)
        pred_mean = np.mean(predictions)
        
        if obs_mean == 0:
            return np.nan
        
        return pred_mean / obs_mean

    @staticmethod
    def std_ratio(observations: np.ndarray, predictions: np.ndarray) -> float:
        """
        Ratio of predicted std to observed std
        
        STD Ratio = std(predictions) / std(observations)
        Ideal value: 1.0
        
        Args:
            observations: Observed values
            predictions: Predicted values
            
        Returns:
            Standard deviation ratio
        """
        obs_std = np.std(observations)
        pred_std = np.std(predictions)
        
        if obs_std == 0:
            return np.nan
        
        return pred_std / obs_std

    @staticmethod
    def calculate_all_metrics(observations: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate all metrics at once
        
        Args:
            observations: Observed values
            predictions: Predicted values
            
        Returns:
            Dictionary with all metric values
        """
        return {
            'NSE': EvaluationMetrics.nse(observations, predictions),
            'KGE': EvaluationMetrics.kge(observations, predictions),
            'Pearson_r': EvaluationMetrics.pearson_r(observations, predictions),
            'Spearman_r': EvaluationMetrics.spearman_r(observations, predictions),
            'Mean_Ratio': EvaluationMetrics.mean_ratio(observations, predictions),
            'STD_Ratio': EvaluationMetrics.std_ratio(observations, predictions)
        }


class EnsembleEvaluator:
    """Evaluate ensemble predictions across multiple stations"""

    def __init__(self, num_seeds: int = 8):
        """
        Initialize ensemble evaluator
        
        Args:
            num_seeds: Number of model seeds in the ensemble
        """
        self.num_seeds = num_seeds
        self.metrics_calculator = EvaluationMetrics()

    def evaluate_ensemble_by_station(self, 
                                    observations: np.ndarray,
                                    ensemble_predictions: List[np.ndarray],
                                    station_ids: np.ndarray) -> pd.DataFrame:
        """
        Evaluate ensemble predictions for each station separately
        
        Args:
            observations: Observed values (N,)
            ensemble_predictions: List of N predictions from N seeds, each (N,)
            station_ids: Station ID for each sample (N,)
            
        Returns:
            DataFrame with metrics for each station
        """
        if len(ensemble_predictions) != self.num_seeds:
            raise ValueError(f"Expected {self.num_seeds} predictions, got {len(ensemble_predictions)}")

        # Ensemble average prediction
        ensemble_avg = np.mean(ensemble_predictions, axis=0)
        
        unique_stations = np.unique(station_ids)
        results = []

        for station_id in unique_stations:
            mask = (station_ids == station_id)
            obs_station = observations[mask]
            pred_station = ensemble_avg[mask]

            if len(obs_station) > 0:
                metrics = self.metrics_calculator.calculate_all_metrics(obs_station, pred_station)
                metrics['Station_ID'] = station_id
                metrics['Sample_Count'] = len(obs_station)
                results.append(metrics)

        return pd.DataFrame(results)

    def evaluate_overall(self, 
                        observations: np.ndarray,
                        ensemble_predictions: List[np.ndarray]) -> Dict[str, float]:
        """
        Evaluate overall ensemble performance across all data
        
        Args:
            observations: Observed values (N,)
            ensemble_predictions: List of N predictions from N seeds, each (N,)
            
        Returns:
            Dictionary with overall metrics
        """
        if len(ensemble_predictions) != self.num_seeds:
            raise ValueError(f"Expected {self.num_seeds} predictions, got {len(ensemble_predictions)}")

        # Ensemble average prediction
        ensemble_avg = np.mean(ensemble_predictions, axis=0)
        
        metrics = self.metrics_calculator.calculate_all_metrics(observations, ensemble_avg)
        metrics['Total_Samples'] = len(observations)
        
        return metrics

    def evaluate_ensemble_uncertainty(self,
                                     ensemble_predictions: List[np.ndarray]) -> np.ndarray:
        """
        Calculate prediction uncertainty (standard deviation across seeds)
        
        Args:
            ensemble_predictions: List of N predictions from N seeds, each (N,)
            
        Returns:
            Uncertainty for each prediction (N,)
        """
        if len(ensemble_predictions) != self.num_seeds:
            raise ValueError(f"Expected {self.num_seeds} predictions, got {len(ensemble_predictions)}")
        
        ensemble_array = np.array(ensemble_predictions)
        return np.std(ensemble_array, axis=0)


def create_results_dataframe(station_metrics: pd.DataFrame, 
                            overall_metrics: Dict[str, float],
                            mode: str = 'IS') -> pd.DataFrame:
    """
    Create comprehensive results dataframe with station-by-station and overall metrics
    
    Args:
        station_metrics: DataFrame with per-station metrics
        overall_metrics: Dictionary with overall metrics
        mode: 'IS' for in-sample or 'OOS' for out-of-sample
        
    Returns:
        Comprehensive results dataframe
    """
    # Add overall row with 'Overall' station_id
    overall_row = overall_metrics.copy()
    overall_row['Station_ID'] = 'Overall'
    overall_row.pop('Total_Samples', None)
    overall_row['Sample_Count'] = overall_metrics.get('Total_Samples', 'All')
    
    # Combine station and overall results
    results_df = pd.concat([
        station_metrics,
        pd.DataFrame([overall_row])
    ], ignore_index=True, sort=False)
    
    # Add mode column
    results_df['Mode'] = mode
    
    # Reorder columns
    metric_cols = ['NSE', 'KGE', 'Pearson_r', 'Spearman_r', 'Mean_Ratio', 'STD_Ratio']
    col_order = ['Mode', 'Station_ID', 'Sample_Count'] + metric_cols
    
    return results_df[col_order]
