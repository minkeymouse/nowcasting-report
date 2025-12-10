"""Plot creation code for forecasting report visualizations.

This module generates forecasting-related plots:
- plot_horizon_trend: Performance trend by forecast horizon (1-22 months)
- plot_forecast_vs_actual: Forecast vs actual time series (per target)

All plots are saved to nowcasting-report/images/ directory as *.png files.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['font.family'] = 'DejaVu Sans'

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = PROJECT_ROOT / "nowcasting-report" / "images"


def _setup_import_paths() -> Path:
    """Set up Python import paths for src and dfm-python modules."""
    import sys
    project_root = Path(__file__).parent.parent.parent
    src_path = project_root / "src"
    dfm_path = project_root / "dfm-python" / "src"
    for path in [str(project_root), str(src_path), str(dfm_path)]:
        if path not in sys.path:
            sys.path.insert(0, path)
    return project_root


def _create_placeholder_plot(message: str, figsize: Tuple[int, int] = (14, 6), 
                            save_path: Optional[Path] = None, n_subplots: int = 1) -> None:
    """Create a placeholder plot with a message."""
    if n_subplots == 1:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, n_subplots, figsize=figsize)
    
    for ax in axes:
        ax.text(0.5, 0.5, message, ha='center', va='center', 
               fontsize=14 if n_subplots > 1 else 16, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated placeholder: {save_path.name}")


def _load_comparison_results(outputs_dir: Path) -> Dict[str, List[Dict]]:
    """Load all comparison results from outputs/comparisons/."""
    comparisons_dir = outputs_dir / "comparisons"
    if not comparisons_dir.exists():
        return {}
    
    all_results = {}
    for comparison_dir in comparisons_dir.iterdir():
        if not comparison_dir.is_dir():
            continue
        results_file = comparison_dir / "comparison_results.json"
        if not results_file.exists():
            continue
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            target_series = data.get('target_series')
            if target_series:
                if target_series not in all_results:
                    all_results[target_series] = []
                all_results[target_series].append(data)
        except Exception:
            continue
    return all_results


def _inverse_transform_chg(transformed_values: np.ndarray, base_value: float) -> np.ndarray:
    """Apply inverse transformation for 'chg' (difference) transformation."""
    if len(transformed_values) == 0:
        return transformed_values
    transformed_clean = np.where(np.isnan(transformed_values), 0.0, transformed_values)
    original_values = base_value + np.cumsum(transformed_clean)
    original_values = np.where(np.isnan(transformed_values), np.nan, original_values)
    return original_values


def _get_transformation_type(target: str, metadata_file: Optional[Path] = None) -> str:
    """Get transformation type for a target series from metadata."""
    if metadata_file is None:
        project_root = Path(__file__).parent.parent.parent
        metadata_file = project_root / "data" / "metadata.csv"
    
    # Check actual data scale first
    try:
        data_file = project_root / "data" / "data.csv"
        if data_file.exists():
            data = pd.read_csv(data_file, index_col=0, parse_dates=True)
            if target in data.columns:
                data_recent = data[(data.index >= '2023-01-01') & (data.index <= '2024-12-31')]
                if len(data_recent) > 0:
                    data_monthly = data_recent.resample('ME').last()
                    values = data_monthly[target].dropna().values
                    if len(values) > 0 and np.all((values >= 80) & (values <= 120)):
                        return 'lin'
    except Exception:
        pass
    
    if not metadata_file.exists():
        return 'lin'
    
    try:
        metadata = pd.read_csv(metadata_file)
        row = metadata[metadata['SeriesID'] == target]
        if len(row) > 0:
            trans = row.iloc[0]['Transformation']
            return str(trans).lower() if pd.notna(trans) else 'lin'
    except Exception:
        pass
    return 'lin'


def _get_base_value(target: str, forecast_start: pd.Timestamp = pd.Timestamp('2024-01-01'), 
                   data_file: Optional[Path] = None) -> float:
    """Get base value for inverse transformation."""
    if data_file is None:
        project_root = Path(__file__).parent.parent.parent
        data_file = project_root / "data" / "data.csv"
    
    if not data_file.exists():
        return 0.0
    
    try:
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        if target not in data.columns:
            return 0.0
        
        trans_type = _get_transformation_type(target)
        base_data = data[(data.index < forecast_start)]
        if len(base_data) == 0:
            return 0.0
        
        base_monthly = base_data.resample('ME').last()
        if len(base_monthly) == 0:
            return 0.0
        
        target_values = base_monthly[target].dropna()
        if len(target_values) == 0:
            return 0.0
        
        if trans_type == 'chg':
            cumulative = np.cumsum(target_values.values)
            return float(cumulative[-1])
        else:
            return float(target_values.iloc[-1])
    except Exception:
        return 0.0


def extract_metrics_from_results(all_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Extract metrics from comparison results into a DataFrame."""
    rows = []
    model_mapping = {'arima': 'ARIMA', 'var': 'VAR', 'dfm': 'DFM', 'ddfm': 'DDFM'}
    targets = ['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
    horizons = list(range(1, 23))
    metrics = ['sMSE', 'sMAE', 'sRMSE']
    
    for target in targets:
        if target not in all_results:
            continue
        result_data = all_results[target][-1] if all_results[target] else None
        if not result_data:
            continue
        results = result_data.get('results', {})
        
        for model_key, model_data in results.items():
            if not isinstance(model_data, dict):
                continue
            model_name = model_mapping.get(model_key.lower(), model_key.upper())
            if model_key.lower() not in ['arima', 'var', 'dfm', 'ddfm']:
                continue
            
            model_metrics = model_data.get('metrics', {})
            if not isinstance(model_metrics, dict):
                continue
            forecast_metrics = model_metrics.get('forecast_metrics', {})
            if not isinstance(forecast_metrics, dict):
                continue
            
            for horizon in horizons:
                horizon_str = str(horizon)
                if horizon_str not in forecast_metrics:
                    continue
                horizon_metrics = forecast_metrics[horizon_str]
                if not isinstance(horizon_metrics, dict):
                    continue
                n_valid = horizon_metrics.get('n_valid', 0)
                if n_valid and n_valid > 0:
                    for metric in metrics:
                        value = horizon_metrics.get(metric)
                        if value == "NaN" or value == "nan" or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                            value = None
                        elif value is not None:
                            rows.append({
                                'model': model_name, 'target': target, 'horizon': horizon,
                                'metric': metric, 'value': value
                            })
    return pd.DataFrame(rows)


def plot_accuracy_heatmap(save_path: Optional[Path] = None):
    """Create accuracy heatmap showing standardized RMSE for 4 models × 3 targets."""
    # Try loading from aggregated_results.csv first
    aggregated_file = OUTPUTS_DIR / "experiments" / "aggregated_results.csv"
    heatmap_data = None
    
    if aggregated_file.exists():
        try:
            df_agg = pd.read_csv(aggregated_file)
            # Calculate average sRMSE per model-target combination
            df_agg_valid = df_agg[
                (df_agg['sRMSE'].notna()) & 
                (df_agg['n_valid'] > 0) &
                (df_agg['sRMSE'] < 1e10)  # Filter extreme values
            ].copy()
            
            if len(df_agg_valid) > 0:
                heatmap_data = df_agg_valid.groupby(['model', 'target'])['sRMSE'].mean().reset_index()
        except Exception as e:
            print(f"Warning: Failed to load aggregated_results.csv: {e}")
    
    # Fallback to comparison results
    if heatmap_data is None or len(heatmap_data) == 0:
        all_results = _load_comparison_results(OUTPUTS_DIR)
        df = extract_metrics_from_results(all_results)
        
        if df.empty or 'value' not in df.columns:
            if save_path is None:
                save_path = IMAGES_DIR / "accuracy_heatmap.png"
            _create_placeholder_plot('Placeholder: No data available', figsize=(8, 6), save_path=save_path)
            return
        
        df_valid = df[(df['metric'] == 'sRMSE') & (df['value'].notna())].copy()
        if len(df_valid) == 0:
            if save_path is None:
                save_path = IMAGES_DIR / "accuracy_heatmap.png"
            _create_placeholder_plot('Placeholder: No sRMSE data available', figsize=(8, 6), save_path=save_path)
            return
        
        heatmap_data = df_valid.groupby(['model', 'target'])['value'].mean().reset_index()
    
    if heatmap_data is None or len(heatmap_data) == 0:
        if save_path is None:
            save_path = IMAGES_DIR / "accuracy_heatmap.png"
        _create_placeholder_plot('Placeholder: No data available', figsize=(8, 6), save_path=save_path)
        return
    
    # Create pivot table
    models = ['ARIMA', 'VAR', 'DFM', 'DDFM']
    targets = ['KOIPALL.G', 'KOEQUIPTE', 'KOWRCCNSE']
    
    pivot_data = heatmap_data.pivot(index='model', columns='target', values='sRMSE' if 'sRMSE' in heatmap_data.columns else 'value')
    
    # Ensure all models and targets are present
    for model in models:
        if model not in pivot_data.index:
            pivot_data.loc[model] = np.nan
    for target in targets:
        if target not in pivot_data.columns:
            pivot_data[target] = np.nan
    
    # Reorder rows and columns
    pivot_data = pivot_data.reindex(models)
    pivot_data = pivot_data[targets]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a colormap where lower values are better (darker = better)
    cmap = sns.cm.rocket_r  # Reversed rocket colormap (dark = low, light = high)
    
    # Mask NaN values
    mask = pivot_data.isna()
    
    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        mask=mask,
        cbar_kws={'label': 'Standardized RMSE'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0,  # Set minimum to 0 for better visualization
        vmax=None  # Auto-scale maximum
    )
    
    ax.set_xlabel('Target Series', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Heatmap: Standardized RMSE by Model and Target', fontsize=13, fontweight='bold', pad=15)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
    
    plt.tight_layout()
    if save_path is None:
        save_path = IMAGES_DIR / "accuracy_heatmap.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {save_path.name}")


def plot_horizon_trend(save_path: Optional[Path] = None):
    """Create horizon trend plot showing sMSE values for all horizons (1-22 months)."""
    all_results = _load_comparison_results(OUTPUTS_DIR)
    df = extract_metrics_from_results(all_results)
    
    if df.empty or 'value' not in df.columns or df['value'].isna().all():
        if save_path is None:
            save_path = IMAGES_DIR / "horizon_trend.png"
        _create_placeholder_plot('Placeholder: No data available', figsize=(10, 6), save_path=save_path)
        return
    
    df_valid = df[df['value'].notna()].copy()
    if len(df_valid) == 0:
        return
    
    horizon_avg = df_valid[df_valid['metric'] == 'sMSE'].groupby(['model', 'horizon'])['value'].mean().reset_index()
    horizon_avg_pivot = horizon_avg.pivot(index='model', columns='horizon', values='value')
    
    if horizon_avg_pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    horizons = list(range(1, 23))
    models = ['ARIMA', 'VAR', 'DFM', 'DDFM']
    colors = {'ARIMA': '#1f77b4', 'VAR': '#ff7f0e', 'DFM': '#2ca02c', 'DDFM': '#d62728'}
    
    for model in models:
        if model not in horizon_avg_pivot.index:
            continue
        values = []
        for h in horizons:
            if h in horizon_avg_pivot.columns:
                val = horizon_avg_pivot.loc[model, h]
                values.append(val if pd.notna(val) and not np.isinf(val) else None)
            else:
                values.append(None)
        
        valid_values = [v for v in values if v is not None and not np.isnan(v)]
        if valid_values:
            plot_values, plot_horizons = [], []
            for h, v in zip(horizons, values):
                if v is not None and not np.isnan(v) and abs(v) < 1e10:
                    plot_values.append(v)
                    plot_horizons.append(h)
            if plot_values:
                ax.plot(plot_horizons, plot_values, marker='o', label=model, 
                       linewidth=2, markersize=4, color=colors.get(model, 'gray'), alpha=0.8)
    
    ax.set_xlabel('Forecast Horizon (months)', fontsize=11)
    ax.set_ylabel('Standardized MSE (sMSE)', fontsize=11)
    ax.set_title('Performance Trend by Forecast Horizon (Standardized MSE)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, 23, 3))
    
    plt.tight_layout()
    if save_path is None:
        save_path = IMAGES_DIR / "horizon_trend.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {save_path.name}")


def plot_forecast_vs_actual(target: str, save_path: Optional[Path] = None):
    """Create forecast vs actual time series plot for a specific target.
    
    Shows:
    - Historical actual values (extended period, e.g., 2019-01 to 2023-12)
    - Forecast period actual values (2024-01 to 2025-10)
    - Model forecasts (ARIMA, VAR, and optionally DFM, DDFM) in original scale with different colors
    """
    project_root = _setup_import_paths()
    
    # Use the local function instead of importing from src.evaluation
    all_results = _load_comparison_results(OUTPUTS_DIR)
    
    if target not in all_results or not all_results[target]:
        if save_path is None:
            save_path = IMAGES_DIR / f"forecast_vs_actual_{target.lower().replace('.', '_')}.png"
        _create_placeholder_plot(f'Placeholder: No data available for {target}', save_path=save_path)
        return
    
    result_data = all_results[target][-1]
    results = result_data.get('results', {})
    
    data_file = project_root / "data" / "data.csv"
    if not data_file.exists():
        print(f"Warning: Data file not found at {data_file}")
        return
    
    try:
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        if target not in data.columns:
            print(f"Warning: Target {target} not found in data columns")
            return
        
        y_full = data[[target]].dropna()
        
        # Extended historical period (e.g., 2019-01 to 2023-12)
        hist_start = pd.Timestamp('2019-01-01')
        hist_end = pd.Timestamp('2023-12-31')
        forecast_start = pd.Timestamp('2024-01-01')
        forecast_end = pd.Timestamp('2025-10-31')
        
        # Historical data (extended)
        y_historical = y_full[(y_full.index >= hist_start) & (y_full.index <= hist_end)]
        y_historical_monthly = y_historical.resample('ME').last()
        
        # Actual forecast period data
        y_actual_forecast = y_full[(y_full.index >= forecast_start) & (y_full.index <= forecast_end)]
        y_actual_forecast_monthly = y_actual_forecast.resample('ME').last()
        
        trans_type = _get_transformation_type(target)
        
        # Determine data scale: check if recent data is transformed
        # Recent data (2010+) is typically in transformed scale for this target
        recent_check_data = y_full[(y_full.index >= '2010-01-01') & (y_full.index <= '2019-12-31')]
        recent_check_monthly = recent_check_data.resample('ME').last()
        if target in recent_check_monthly.columns:
            recent_check_values = recent_check_monthly[target].dropna().values
            data_is_transformed = len(recent_check_values) > 0 and np.abs(recent_check_values).mean() < 10
        else:
            data_is_transformed = False
        
        # Get base value for inverse transformation
        # The data has mixed scales: training period (1985-2019) has mixed original/transformed values
        # Recent period (2010+) is in transformed scale
        # Model predicts in original scale (matching the scale of training data)
        
        # Method: Use the model's training data to determine the correct base value
        # The model was trained on data ending at 2019-12, so we use that as reference
        train_data = y_full[(y_full.index >= '1985-01-01') & (y_full.index <= '2019-12-31')]
        train_monthly = train_data.resample('ME').last()
        
        hist_base_value = 0.0
        if target in train_monthly.columns:
            train_values = train_monthly[target].dropna()
            if len(train_values) > 0:
                train_values_array = train_values.values
                train_dates = train_values.index
                
                # Find the last original-scale value in training period (used by model)
                # Original-scale values are typically > 50 in absolute value
                original_scale_mask = np.abs(train_values_array) > 50
                
                if np.any(original_scale_mask):
                    # Get the last original-scale value and its date
                    last_original_idx = np.where(original_scale_mask)[0][-1]
                    last_original_date = train_dates[last_original_idx]
                    hist_base_value = float(train_values_array[last_original_idx])
                    
                    # Accumulate transformed values from that point to 2019-12
                    if last_original_idx < len(train_values_array) - 1:
                        remaining_values = train_values_array[last_original_idx + 1:]
                        if trans_type == 'chg':
                            # For 'chg' (difference), accumulate to get level
                            cumulative_remaining = np.cumsum(remaining_values)
                            hist_base_value = hist_base_value + float(cumulative_remaining[-1]) if len(cumulative_remaining) > 0 else hist_base_value
                        else:
                            # For other transformations, use the last value directly
                            hist_base_value = float(remaining_values[-1]) if len(remaining_values) > 0 else hist_base_value
                else:
                    # No original-scale values found - all data is transformed
                    # Use cumulative sum from beginning
                    if trans_type == 'chg':
                        cumulative_all = np.cumsum(train_values_array)
                        hist_base_value = float(cumulative_all[-1]) if len(cumulative_all) > 0 else 0.0
                    else:
                        hist_base_value = float(train_values_array[-1]) if len(train_values_array) > 0 else 0.0
        
        # Convert historical data to original scale
        hist_values_transformed = y_historical_monthly[target].values
        if data_is_transformed and trans_type == 'chg' and len(hist_values_transformed) > 0:
            hist_values_original = _inverse_transform_chg(hist_values_transformed, hist_base_value)
            base_value = float(hist_values_original[-1]) if len(hist_values_original) > 0 else hist_base_value
        else:
            hist_values_original = hist_values_transformed
            base_value = float(hist_values_original[-1]) if len(hist_values_original) > 0 else hist_base_value
        
        # Convert actual forecast period to original scale
        actual_transformed = y_actual_forecast_monthly[target].values
        if data_is_transformed and trans_type == 'chg':
            actual_original = _inverse_transform_chg(actual_transformed, base_value)
        else:
            actual_original = actual_transformed
        
        # Get forecast dates (needed for model forecasts)
        forecast_dates = y_actual_forecast_monthly.index
        horizon = len(forecast_dates)
        
        # Load model forecasts from checkpoints
        checkpoint_dir = project_root / "checkpoints"
        forecast_data = {}
        model_colors = {
            'ARIMA': '#1f77b4',  # Blue
            'VAR': '#ff7f0e',     # Orange
            'DFM': '#2ca02c',     # Green
            'DDFM': '#d62728'     # Red
        }
        
        # Models to try (in order of preference)
        models_to_load = ['arima', 'var', 'dfm', 'ddfm']
        
        for model_name in models_to_load:
            checkpoint_path = checkpoint_dir / f"{target}_{model_name}" / "model.pkl"
            if not checkpoint_path.exists():
                continue
            
            try:
                import pickle
                from src.models import forecast_arima, forecast_var, forecast_dfm, forecast_ddfm
                from src.preprocessing import resample_to_monthly
                
                # Load checkpoint
                with open(checkpoint_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                forecaster = model_data.get('forecaster')
                if forecaster is None:
                    continue
                
                # Generate forecast for all horizons (22 months: 2024-01 to 2025-10)
                # For ARIMA/VAR: recursive forecasting from training end (2019-12)
                # Training ends: 2019-12-31, Forecast starts: 2024-01-01
                # Since forecast_arima/var uses MS (Month Start) frequency,
                # 2019-12-31 + 1 month (MS) = 2020-02-01
                # To reach 2024-01-01: 2020-02-01 to 2024-01-01 = 48 steps
                if horizon == 0:
                    continue
                
                training_end = pd.Timestamp('2019-12-31')
                
                # Calculate steps: 2020-02-01 (first forecast) to 2024-01-01 (forecast start) = 48 steps
                months_to_forecast_start = 48
                total_horizon = months_to_forecast_start + horizon
                
                # Generate forecast
                if model_name.lower() == 'arima':
                    forecast_df = forecast_arima(forecaster, total_horizon, training_end)
                elif model_name.lower() == 'var':
                    forecast_df = forecast_var(forecaster, total_horizon, training_end)
                elif model_name.lower() == 'dfm':
                    forecast_df = forecast_dfm(forecaster, total_horizon, training_end)
                elif model_name.lower() == 'ddfm':
                    forecast_df = forecast_ddfm(forecaster, total_horizon, training_end)
                else:
                    continue
                
                # Extract the last 'horizon' values (corresponding to 2024-01 to 2025-10)
                # The forecast index uses MS (Month Start), but forecast_dates uses ME (Month End)
                # So we extract by position and reassign index
                if isinstance(forecast_df, pd.DataFrame):
                    forecast_df = forecast_df.iloc[-horizon:].copy()
                elif isinstance(forecast_df, pd.Series):
                    forecast_df = forecast_df.iloc[-horizon:].copy()
                
                # Reassign index to match forecast_dates (ME frequency)
                forecast_df.index = forecast_dates[:len(forecast_df)]
                
                # Extract target series from forecast
                if isinstance(forecast_df, pd.DataFrame):
                    if target in forecast_df.columns:
                        forecast_series = forecast_df[target]
                    else:
                        forecast_series = forecast_df.iloc[:, 0]
                else:
                    forecast_series = forecast_df
                
                # Convert forecast to original scale
                forecast_values = forecast_series.values
                if len(forecast_values) > horizon:
                    forecast_values = forecast_values[:horizon]
                
                # Model forecasts are typically in original scale (175-177 for this target)
                # Actual data is in transformed scale, so we've already converted it to original scale above
                # Therefore, forecast is already in the correct scale (original) and can be used as-is
                forecast_original = forecast_values
                
                # Store forecast
                model_display_name = model_name.upper()
                forecast_data[model_display_name] = forecast_original
                
            except Exception as e:
                print(f"Warning: Failed to load forecast for {model_name}: {e}")
                continue
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot historical actual values
        hist_dates = y_historical_monthly.index
        ax.plot(hist_dates, hist_values_original, color='gray', linestyle='-', 
               linewidth=2, label='Historical Actual', alpha=0.8)
        
        # Plot forecast period actual values
        forecast_dates = y_actual_forecast_monthly.index
        ax.plot(forecast_dates, actual_original, 'k-', linewidth=2.5, 
               label='Actual (2024-2025)', alpha=0.9)
        
        # Plot model forecasts with different colors
        for model_name in ['ARIMA', 'VAR', 'DFM', 'DDFM']:
            if model_name in forecast_data:
                forecast_values = forecast_data[model_name]
                if len(forecast_values) == len(forecast_dates):
                    color = model_colors.get(model_name, 'gray')
                    ax.plot(forecast_dates, forecast_values, color=color, linestyle='--', 
                           linewidth=2, label=f'{model_name} Forecast', alpha=0.8, marker='o', markersize=3)
        
        # Add vertical line at forecast start
        ax.axvline(x=forecast_start, color='red', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel(f'{target} Value (Original Scale)', fontsize=11)
        ax.set_title(f'Forecast vs Actual: {target} (Original Scale)', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        
        plt.tight_layout()
        if save_path is None:
            save_path = IMAGES_DIR / f"forecast_vs_actual_{target.lower().replace('.', '_')}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated: {save_path.name}")
        
    except Exception as e:
        print(f"Error generating forecast plot for {target}: {e}")
        import traceback
        traceback.print_exc()
        if save_path is None:
            save_path = IMAGES_DIR / f"forecast_vs_actual_{target.lower().replace('.', '_')}.png"
        _create_placeholder_plot(f'Error generating plot for {target}\n{str(e)}', save_path=save_path)


def generate_forecast_plots():
    """Generate all forecasting-related plots."""
    print("=" * 70)
    print("Generating Forecast Plots")
    print("=" * 70)
    
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    targets = ['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
    
    print("\n1. Forecast vs Actual (3 plots)")
    for target in targets:
        print(f"   - forecast_vs_actual_{target.lower().replace('.', '_')}.png")
        plot_forecast_vs_actual(target)
    
    print("\n2. Accuracy Heatmap")
    print("   - accuracy_heatmap.png")
    plot_accuracy_heatmap()
    
    print("\n3. Horizon Performance Trend")
    print("   - horizon_trend.png")
    plot_horizon_trend()
    
    print("\n" + "=" * 70)
    print("Forecast plot generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    generate_forecast_plots()

