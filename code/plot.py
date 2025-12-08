"""Plot creation code for nowcasting report visualizations.

This module generates all plots required by RESULTS_NEEDED.md section 6 (Visualization).
All plots are saved to nowcasting-report/images/ directory as *.png files (no subdirectories).
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

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = PROJECT_ROOT / "nowcasting-report" / "images"


def _setup_import_paths() -> Path:
    """Set up Python import paths for src and dfm-python modules.
    
    Returns
    -------
    Path
        Project root directory
    """
    import sys
    
    project_root = Path(__file__).parent.parent.parent
    src_path = project_root / "src"
    dfm_path = project_root / "dfm-python" / "src"
    
    paths_to_add = [
        str(project_root),
        str(src_path),
        str(dfm_path)
    ]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return project_root


def _create_placeholder_plot(message: str, figsize: Tuple[int, int] = (14, 6), 
                            save_path: Optional[Path] = None, n_subplots: int = 1) -> None:
    """Create a placeholder plot with a message.
    
    Parameters
    ----------
    message : str
        Message to display
    figsize : Tuple[int, int]
        Figure size (width, height)
    save_path : Path, optional
        Output path for the plot
    n_subplots : int
        Number of subplots (1 for single plot, 2 for side-by-side)
    """
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
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated placeholder: {save_path.name}")


def _load_comparison_results(outputs_dir: Path) -> Dict[str, List[Dict]]:
    """Load all comparison results from outputs/comparisons/.
    
    This is a local helper that matches the interface of collect_all_comparison_results
    but doesn't require importing from src.evaluation (which may have dependencies).
    """
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
    """Apply inverse transformation for 'chg' (difference) transformation.
    
    For difference transformation: X_diff[t] = X[t] - X[t-1]
    Inverse: X[t] = X[t-1] + X_diff[t] = base_value + cumulative_sum(X_diff)
    
    Parameters
    ----------
    transformed_values : np.ndarray
        Transformed values (differences)
    base_value : float
        Base value (last value before forecast period) to start cumulative sum
        
    Returns
    -------
    np.ndarray
        Original level values
    """
    if len(transformed_values) == 0:
        return transformed_values
    
    # Handle NaN values: replace with 0 for cumulative sum calculation
    transformed_clean = np.where(np.isnan(transformed_values), 0.0, transformed_values)
    
    # Calculate cumulative sum starting from base_value
    # Use numpy.cumsum for efficiency
    original_values = base_value + np.cumsum(transformed_clean)
    
    # Restore NaN where original values were NaN
    original_values = np.where(np.isnan(transformed_values), np.nan, original_values)
    
    return original_values


def _get_transformation_type(target: str, metadata_file: Optional[Path] = None) -> str:
    """Get transformation type for a target series from metadata.
    
    Parameters
    ----------
    target : str
        Target series name
    metadata_file : Path, optional
        Path to metadata.csv file. If None, uses default location.
        
    Returns
    -------
    str
        Transformation type ('chg', 'pch', 'log', 'lin', etc.)
    """
    if metadata_file is None:
        project_root = Path(__file__).parent.parent.parent
        metadata_file = project_root / "data" / "metadata.csv"
    
    if not metadata_file.exists():
        # Default to 'chg' if metadata not found (most common)
        return 'chg'
    
    try:
        metadata = pd.read_csv(metadata_file)
        row = metadata[metadata['SeriesID'] == target]
        if len(row) > 0:
            trans = row.iloc[0]['Transformation']
            return str(trans).lower() if pd.notna(trans) else 'chg'
    except Exception:
        pass
    
    # Default to 'chg' if not found
    return 'chg'


def _get_base_value(target: str, forecast_start: pd.Timestamp = pd.Timestamp('2024-01-01'), 
                   data_file: Optional[Path] = None) -> float:
    """Get base value for inverse transformation (last value before forecast period in ORIGINAL SCALE).
    
    IMPORTANT: data.csv contains values in TRANSFORMED SPACE (chg = differences).
    To get the original scale base_value, we need to:
    1. Get all transformed values before forecast_start
    2. Calculate cumulative sum from a starting point
    3. Use the last cumulative sum value as base_value
    
    However, we need a reference point. We use the earliest available data point
    as the starting point (assuming it represents the level at that time, or use 0 as default).
    
    Parameters
    ----------
    target : str
        Target series name
    forecast_start : pd.Timestamp
        Start of forecast period (default: 2024-01-01)
    data_file : Path, optional
        Path to data.csv file. If None, uses default location.
        
    Returns
    -------
    float
        Base value in ORIGINAL SCALE (last value before forecast period), or 0.0 if not found
    """
    if data_file is None:
        project_root = Path(__file__).parent.parent.parent
        data_file = project_root / "data" / "data.csv"
    
    if not data_file.exists():
        return 0.0
    
    try:
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        if target not in data.columns:
            return 0.0
        
        # Get transformation type
        trans_type = _get_transformation_type(target)
        
        # Get all data before forecast_start
        base_data = data[(data.index < forecast_start)]
        if len(base_data) == 0:
            return 0.0
        
        # Aggregate to monthly (consistent with forecast period aggregation)
        base_monthly = base_data.resample('ME').last()
        if len(base_monthly) == 0:
            return 0.0
        
        target_values = base_monthly[target].dropna()
        if len(target_values) == 0:
            return 0.0
        
        if trans_type == 'chg':
            # Data is in transformed space (differences)
            # To get original scale, we need to calculate cumulative sum
            # We need a starting point - use the first non-NaN value as reference level
            # Or use 0 as default (which means first value becomes the starting level)
            # For index series like KOIPALL.G, we might need to use a known reference point
            # But for now, we'll use cumulative sum from 0, which gives us relative levels
            # The actual absolute level doesn't matter for plotting - what matters is continuity
            
            # Calculate cumulative sum
            # Start from 0 (relative levels) - the absolute starting point doesn't matter
            # because we're just ensuring continuity between historical and forecast periods
            cumulative = np.cumsum(target_values.values)
            base_value = float(cumulative[-1])  # Last cumulative value before forecast
        else:
            # For non-chg transformations, use last value directly
            base_value = float(target_values.iloc[-1])
        
        return base_value
    except Exception as e:
        print(f"Warning: Failed to get base_value for {target}: {e}")
        return 0.0


def _get_target_display_name(target: str) -> str:
    """Get display name for a target series.
    
    Parameters
    ----------
    target : str
        Target series name
        
    Returns
    -------
    str
        Display name for the target
    """
    target_names = {
        'KOEQUIPTE': 'Equipment Investment',
        'KOWRCCNSE': 'Wholesale/Retail Sales',
        'KOIPALL.G': 'Industrial Production'
    }
    return target_names.get(target, target)


def _load_backtest_results(target: str, outputs_dir: Path, model_filter: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load all backtest JSON files for a target.
    
    Parameters
    ----------
    target : str
        Target series name
    outputs_dir : Path
        Outputs directory
    model_filter : str, optional
        Filter by model ('dfm' or 'ddfm'). If None, loads both.
    
    Returns:
        Dictionary with structure: {timepoint: {month: {forecasts: [], actual: value}}}
    """
    backtest_dir = outputs_dir / "backtest"
    if not backtest_dir.exists():
        return {}
    
    # Filter by model if specified
    if model_filter:
        model_filter_lower = model_filter.lower()
        if model_filter_lower == 'dfm':
            models = ['dfm']
        elif model_filter_lower == 'ddfm':
            models = ['ddfm']
        else:
            models = []
    else:
        # Only DFM and DDFM support nowcasting (ARIMA/VAR cannot handle missing data from release date masking)
        models = ['dfm', 'ddfm']
    
    timepoints = ['4weeks', '1weeks']
    data_by_timepoint = {tp: {} for tp in timepoints}
    
    for model in models:
        backtest_file = backtest_dir / f"{target}_{model}_backtest.json"
        if not backtest_file.exists():
            continue
        
        try:
            with open(backtest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results_by_timepoint = data.get('results_by_timepoint', {})
            
            for tp in timepoints:
                if tp not in results_by_timepoint:
                    continue
                
                tp_results = results_by_timepoint[tp]
                monthly_results = tp_results.get('monthly_results', [])
                
                for month_data in monthly_results:
                    month_str = month_data.get('month')
                    if month_str is None:
                        continue
                    
                    if month_str not in data_by_timepoint[tp]:
                        data_by_timepoint[tp][month_str] = {'forecasts': [], 'actual': None}
                    
                    forecast_value = month_data.get('forecast_value')
                    if forecast_value is not None and not np.isnan(forecast_value):
                        data_by_timepoint[tp][month_str]['forecasts'].append(forecast_value)
                    
                    # Store actual value (same for all models)
                    if data_by_timepoint[tp][month_str]['actual'] is None:
                        data_by_timepoint[tp][month_str]['actual'] = month_data.get('actual_value')
        except Exception as e:
            print(f"Warning: Failed to load {backtest_file}: {e}")
            continue
    
    return data_by_timepoint


def extract_metrics_from_results(all_results: Dict[str, List[Dict]]) -> pd.DataFrame:
    """Extract metrics from comparison results into a DataFrame.
    
    Used by plot_horizon_trend to extract sMSE values for all horizons.
    """
    rows = []
    
    # Model name mapping (lowercase in JSON)
    model_mapping = {
        'arima': 'ARIMA',
        'var': 'VAR',
        'dfm': 'DFM',
        'ddfm': 'DDFM'
    }
    
    # Current targets: KOEQUIPTE, KOWRCCNSE, KOIPALL.G
    targets = ['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
    horizons = list(range(1, 23))  # 1-22 months (2024-01 to 2025-10) as per WORKFLOW.md
    metrics = ['sMSE', 'sMAE', 'sRMSE']
    
    for target in targets:
        if target not in all_results:
            continue
        
        # Use the latest result for each target
        result_data = all_results[target][-1] if all_results[target] else None
        if not result_data:
            continue
        
        results = result_data.get('results', {})
        
        for model_key, model_data in results.items():
            if not isinstance(model_data, dict) or model_data is None:
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
                
                # Only include if we have valid predictions
                if n_valid and n_valid > 0:
                    for metric in metrics:
                        value = horizon_metrics.get(metric)
                        
                        # Handle NaN values (stored as string "NaN" in JSON or actual NaN)
                        if value == "NaN" or value == "nan" or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                            value = None
                        elif value is not None:
                            rows.append({
                                'model': model_name,
                                'target': target,
                                'horizon': horizon,
                                'metric': metric,
                                'value': value
                            })
    
    return pd.DataFrame(rows)


def plot_horizon_trend(save_path: Optional[Path] = None):
    """Create horizon trend plot (Plot3: fig:horizon_performance_trend).
    
    Shows sMSE values for all horizons from 1 to 22 months (2024-01 to 2025-10).
    X-axis: forecast horizon (1-22 months), Y-axis: sMSE value.
    Four lines representing four models (ARIMA, VAR, DFM, DDFM).
    """
    # Load data
    all_results = _load_comparison_results(OUTPUTS_DIR)
    df = extract_metrics_from_results(all_results)
    
    # Check if we have any valid data
    if df.empty or 'value' not in df.columns or df['value'].isna().all():
        if save_path is None:
            save_path = IMAGES_DIR / "horizon_trend.png"
        _create_placeholder_plot('Placeholder: No data available', figsize=(10, 6), save_path=save_path)
        return
    
    # Aggregate by model and horizon (average across targets)
    # Exclude None values
    df_valid = df[df['value'].notna()].copy()
    if len(df_valid) == 0:
        return
    
    # Filter for sMSE metric only (as per WORKFLOW.md Plot3)
    horizon_avg = df_valid[df_valid['metric'] == 'sMSE'].groupby(['model', 'horizon'])['value'].mean().reset_index()
    horizon_avg_pivot = horizon_avg.pivot(
        index='model', columns='horizon', values='value'
    )
    
    if horizon_avg_pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # All horizons from 1 to 22 (monthly: 2024-01 to 2025-10)
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
        
        # Only plot if we have at least one valid value
        valid_values = [v for v in values if v is not None and not np.isnan(v)]
        if valid_values:
            # Filter out extreme values for plotting (numerical instability)
            plot_values = []
            plot_horizons = []
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
    ax.set_xticks(range(1, 23, 3))  # Show every 3 months for readability
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = IMAGES_DIR / "horizon_trend.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {save_path.name}")


def plot_forecast_vs_actual(target: str, save_path: Optional[Path] = None):
    """Create forecast vs actual time series plot for a specific target.
    
    Shows 22 months of forecasts (2024-01 to 2025-10) with actual values and 4 model predictions.
    Each plot consists of original series line, ARIMA, VAR, DFM, DDFM lines (5 lines total).
    
    Parameters
    ----------
    target : str
        Target series name (KOEQUIPTE, KOWRCCNSE, or KOIPALL.G)
    save_path : Path, optional
        Output path for the plot
    """
    # Set up import paths
    project_root = _setup_import_paths()
    
    # Load comparison results to get model paths
    from src.evaluation import collect_all_comparison_results
    all_results = collect_all_comparison_results(OUTPUTS_DIR)
    
    if target not in all_results or not all_results[target]:
        if save_path is None:
            save_path = IMAGES_DIR / f"forecast_vs_actual_{target.lower().replace('.', '_')}.png"
        _create_placeholder_plot(f'Placeholder: No data available for {target}', save_path=save_path)
        return
    
    # Use latest result for this target
    result_data = all_results[target][-1]
    results = result_data.get('results', {})
    
    # Load original data
    data_file = project_root / "data" / "data.csv"
    if not data_file.exists():
        print(f"Warning: Data file not found at {data_file}")
        return
    
    try:
        # Data file has first column as dates (no column name), use index_col=0
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        if target not in data.columns:
            print(f"Warning: Target {target} not found in data columns")
            return
        
        # Extract target series
        y_full = data[[target]].dropna()
        
        # Define date ranges
        # Historical period: 2023-01 to 2023-12 (12 months)
        hist_start = pd.Timestamp('2023-01-01')
        hist_end = pd.Timestamp('2023-12-31')
        # Forecast period: 2024-01 to 2025-10 (22 months)
        forecast_start = pd.Timestamp('2024-01-01')
        forecast_end = pd.Timestamp('2025-10-31')
        
        # Filter data for historical period (2023)
        y_historical = y_full[(y_full.index >= hist_start) & (y_full.index <= hist_end)]
        # Aggregate to monthly (take last value of each month)
        y_historical_monthly = y_historical.resample('ME').last()
        
        # Filter data for forecast period (2024-2025)
        y_actual_forecast = y_full[(y_full.index >= forecast_start) & (y_full.index <= forecast_end)]
        # Aggregate to monthly (take last value of each month)
        y_actual_forecast_monthly = y_actual_forecast.resample('ME').last()
        
        # Get training data for model fitting (1985-2019)
        train_start = pd.Timestamp('1985-01-01')
        train_end = pd.Timestamp('2019-12-31')
        y_train_data = y_full[(y_full.index >= train_start) & (y_full.index <= train_end)]
        
        # Get transformation type for this target
        trans_type = _get_transformation_type(target)
        
        # Get base value for inverse transformation
        # IMPORTANT: data.csv contains values in TRANSFORMED SPACE (chg = differences).
        # 
        # Strategy for continuity:
        # 1. First, inverse transform historical data (2023) to get original scale
        # 2. Use the last value of inverse-transformed historical data (2023-12) as base_value
        #    for forecast period (2024-2025)
        # 
        # This ensures perfect continuity: historical[2023-12] == forecast_base_value
        
        # Step 1: Calculate base_value for historical data (2023)
        # Get all data before 2023-01 (from earliest available)
        hist_base_data = y_full[(y_full.index < hist_start)]
        hist_base_value = 0.0
        if len(hist_base_data) > 0:
            hist_base_monthly = hist_base_data.resample('ME').last()
            hist_base_values = hist_base_monthly[target].dropna()
            
            if len(hist_base_values) > 0:
                # Calculate cumulative sum to get base_value before 2023
                cumulative_before_2023 = np.cumsum(hist_base_values.values)
                hist_base_value = float(cumulative_before_2023[-1]) if len(cumulative_before_2023) > 0 else 0.0
        
        # Step 2: Inverse transform historical data (2023) to get original scale
        hist_values_transformed = y_historical_monthly[target].values
        hist_values_original = None  # Initialize for use in plotting section
        if trans_type == 'chg' and len(hist_values_transformed) > 0:
            hist_values_original = _inverse_transform_chg(hist_values_transformed, hist_base_value)
            # Step 3: Use the last value of inverse-transformed historical data as base_value for forecast
            base_value = float(hist_values_original[-1])  # This is 2023-12 in original scale
        else:
            # For non-chg transformations, use last historical value directly
            if len(hist_values_transformed) > 0:
                base_value = float(hist_values_transformed[-1])
            else:
                base_value = _get_base_value(target, forecast_start)
        
        # Prepare forecast data
        forecast_data = {}
        # Apply inverse transformation to actual values if needed
        actual_transformed = y_actual_forecast_monthly[target].values
        if trans_type == 'chg':
            # Convert differences to levels using base_value (last value of historical period)
            forecast_data['Actual'] = _inverse_transform_chg(actual_transformed, base_value)
        else:
            # For other transformations, use as-is (or add more inverse transforms if needed)
            forecast_data['Actual'] = actual_transformed
        
        # Load models and generate forecasts
        models_to_load = ['arima', 'var', 'dfm', 'ddfm']
        for model_key in models_to_load:
            if model_key not in results:
                continue
            
            model_info = results[model_key]
            model_dir_str = model_info.get('model_dir', '')
            if not model_dir_str:
                continue
            
            # Convert relative path to absolute path if needed
            model_dir = Path(model_dir_str)
            if not model_dir.is_absolute():
                # If relative path, assume it's relative to project root
                model_dir = project_root / model_dir_str
            model_file = model_dir / "model.pkl"
            
            if not model_file.exists():
                print(f"Warning: Model file not found: {model_file}")
                continue
            
            try:
                # Set up paths for imports (needed for unpickling)
                import os
                import pickle
                project_root = _setup_import_paths()
                src_path = project_root / "src"
                dfm_path = project_root / "dfm-python" / "src"
                
                # Change to project root directory to help with relative imports in pickled models
                original_cwd = os.getcwd()
                try:
                    os.chdir(str(project_root))
                    
                    # Also set PYTHONPATH environment variable for subprocess imports
                    env = os.environ.copy()
                    env['PYTHONPATH'] = os.pathsep.join([str(project_root), str(src_path), str(dfm_path), env.get('PYTHONPATH', '')])
                    os.environ.update(env)
                    
                    # Pre-import modules that might be needed for unpickling
                    # This helps Python find modules when unpickling objects that reference them
                    try:
                        import importlib
                        # Import modules that pickled models might reference
                        if model_key in ['dfm', 'ddfm']:
                            importlib.import_module('src.model.dfm_models')
                            importlib.import_module('src.model.sktime_forecaster')
                            importlib.import_module('src.train')
                            importlib.import_module('src.utils.config_parser')
                    except ImportError as import_err:
                        # If imports fail, continue anyway - pickle might still work
                        print(f"Warning: Pre-import failed (may be OK): {import_err}")
                    
                    # Custom unpickler to handle wrong module paths in pickled objects
                    class CustomUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            # Handle wrong module paths that might be in pickled objects
                            if module == 'src.model.dfm' and name in ['DFM', 'DFMForecaster']:
                                # Redirect to correct module
                                module = 'src.model.dfm_models'
                            elif module == 'src.model.ddfm' and name in ['DDFM', 'DDFMForecaster']:
                                # Redirect to correct module
                                module = 'src.model.dfm_models'
                            # Call parent method with potentially corrected module
                            return super().find_class(module, name)
                    
                    # Load model with custom unpickler
                    with open(model_file, 'rb') as f:
                        unpickler = CustomUnpickler(f)
                        model_data = unpickler.load()
                    
                    forecaster = model_data.get('forecaster')
                    if forecaster is None:
                        continue
                    
                    # Prepare training data based on model type
                    if model_key == 'arima':
                        # ARIMA uses univariate data (target series only)
                        y_train_model = y_train_data[[target]]
                    else:
                        # VAR, DFM, DDFM use multivariate data (all series)
                        # Use all numeric columns from full data
                        y_train_model = data.select_dtypes(include=[np.number]).dropna()
                        # Ensure target is included
                        if target not in y_train_model.columns:
                            y_train_model[target] = data[target]
                        y_train_model = y_train_model.dropna()
                    
                    # Generate forecasts for test period
                    # Fit on training data (if not already fitted)
                    try:
                        # Check if model is already fitted
                        if hasattr(forecaster, '_is_fitted') and forecaster._is_fitted:
                            # Model already fitted, just predict
                            pass
                        else:
                            # Fit on training data
                            forecaster.fit(y_train_model)
                        
                        # Predict for forecast period (2024-01 to 2025-10, 22 months)
                        # CRITICAL: Predict all horizons at once to get proper sequential forecasts
                        # Each horizon should use the previous predictions, not independent predictions from the same base
                        n_forecast = 22
                        
                        try:
                            # Predict all horizons at once - this gives proper sequential forecasts
                            # where each horizon uses the previous predictions
                            # For DFM/DDFM, use history parameter to update factor state with recent data (2023)
                            # This ensures predictions reflect the most recent information, not just training period
                            if model_key in ['dfm', 'ddfm']:
                                # Use 12 months of history (2023 data) to update factor state
                                # This is critical for getting wiggly, realistic forecasts instead of flat lines
                                pred_all = forecaster.predict(fh=list(range(1, n_forecast + 1)), history=12)
                            else:
                                pred_all = forecaster.predict(fh=list(range(1, n_forecast + 1)))
                            
                            # Extract predictions for target series
                            if isinstance(pred_all, pd.DataFrame):
                                if target in pred_all.columns:
                                    forecast_values = pred_all[target].values
                                else:
                                    # Try case-insensitive match
                                    target_lower = target.lower()
                                    matching_cols = [c for c in pred_all.columns if c.lower() == target_lower]
                                    if matching_cols:
                                        forecast_values = pred_all[matching_cols[0]].values
                                    else:
                                        # Use first column as fallback
                                        forecast_values = pred_all.iloc[:, 0].values
                            elif isinstance(pred_all, pd.Series):
                                forecast_values = pred_all.values
                            else:
                                # Convert to array
                                pred_array = np.asarray(pred_all)
                                if pred_array.ndim == 1:
                                    forecast_values = pred_array
                                elif pred_array.ndim == 2:
                                    # If 2D, use first column (assuming univariate or first series)
                                    forecast_values = pred_array[:, 0] if pred_array.shape[1] > 0 else np.full(n_forecast, np.nan)
                                else:
                                    forecast_values = np.full(n_forecast, np.nan)
                            
                            # Ensure we have exactly n_forecast values
                            if len(forecast_values) > n_forecast:
                                forecast_values = forecast_values[:n_forecast]
                            elif len(forecast_values) < n_forecast:
                                # Pad with NaN if needed
                                padded = np.full(n_forecast, np.nan)
                                padded[:len(forecast_values)] = forecast_values
                                forecast_values = padded
                            
                            # Convert to float array and handle invalid values
                            forecast_values = np.array([float(v) if not (pd.isna(v) or np.isinf(v)) else np.nan for v in forecast_values])
                            
                        except Exception as e:
                            print(f"Warning: Failed to predict all horizons at once for {model_key}, trying individual predictions: {e}")
                            # Fallback: try individual predictions (less ideal but better than nothing)
                            forecast_values_list = []
                            for h in range(1, n_forecast + 1):
                                try:
                                    pred = forecaster.predict(fh=[h])
                                    if isinstance(pred, pd.DataFrame):
                                        if target in pred.columns:
                                            pred_value = pred[target].iloc[-1]
                                        else:
                                            pred_value = pred.iloc[-1, 0]
                                    elif isinstance(pred, pd.Series):
                                        pred_value = pred.iloc[-1] if len(pred) > 0 else np.nan
                                    else:
                                        pred_array = np.asarray(pred)
                                        pred_value = pred_array.flatten()[-1] if pred_array.size > 0 else np.nan
                                    
                                    if pd.isna(pred_value) or np.isinf(pred_value):
                                        forecast_values_list.append(np.nan)
                                    else:
                                        forecast_values_list.append(float(pred_value))
                                except Exception as e2:
                                    print(f"Warning: Failed to predict horizon {h} for {model_key}: {e2}")
                                    forecast_values_list.append(np.nan)
                            
                            forecast_values = np.array(forecast_values_list)
                        
                        # NOTE: DFM/DDFM predict() returns values in transformed space (after unstandardization)
                        # The formula is: X_forecast = X_forecast_std * Wx + Mx
                        # This unstandardizes (reverses mean/scale) but does NOT reverse the transformation (log, pch, etc.)
                        # Both the data.csv and predictions are in transformed scale (e.g., percent change, difference)
                        # Apply inverse transformation to convert back to original levels for plotting
                        
                        # Apply inverse transformation based on transformation type
                        if trans_type == 'chg':
                            # Convert differences to levels using cumulative sum
                            forecast_values_original = _inverse_transform_chg(forecast_values, base_value)
                        else:
                            # For other transformations, use as-is (or add more inverse transforms if needed)
                            forecast_values_original = forecast_values
                        
                        forecast_data[model_key.upper()] = forecast_values_original
                        
                    except Exception as e:
                        print(f"Warning: Forecast generation failed for {model_key}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                finally:
                    # Restore original working directory
                    os.chdir(original_cwd)
                    
            except Exception as e:
                print(f"Warning: Failed to load/generate forecasts for {model_key}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot historical data (2023-01 to 2023-12) - single line, actual values only
        # Use the same inverse transformation as calculated above for base_value
        hist_dates = y_historical_monthly.index
        # hist_values_original was already calculated above for base_value calculation
        if trans_type == 'chg' and hist_values_original is not None:
            hist_values = hist_values_original  # Reuse the already-calculated inverse-transformed values
        else:
            hist_values = hist_values_transformed
        
        ax.plot(hist_dates, hist_values, 'k-', linewidth=2, 
               label='Historical (2023)', alpha=0.7)
        
        # Plot forecast period (2024-01 to 2025-10) - actual and predictions
        forecast_dates = y_actual_forecast_monthly.index
        n_forecast_periods = len(forecast_dates)
        
        # Ensure all forecast arrays have the same length as forecast_dates
        for model_name in forecast_data.keys():
            if len(forecast_data[model_name]) > n_forecast_periods:
                forecast_data[model_name] = forecast_data[model_name][:n_forecast_periods]
            elif len(forecast_data[model_name]) < n_forecast_periods:
                # Pad with NaN if needed
                padded = np.full(n_forecast_periods, np.nan)
                padded[:len(forecast_data[model_name])] = forecast_data[model_name]
                forecast_data[model_name] = padded
        
        # Plot actual values in forecast period
        ax.plot(forecast_dates, forecast_data['Actual'], 'k-', linewidth=2, 
               label='Actual (2024-2025)', alpha=0.7)
        
        # Plot model forecasts
        colors = {'ARIMA': '#1f77b4', 'VAR': '#ff7f0e', 'DFM': '#2ca02c', 'DDFM': '#d62728'}
        for model_name in ['ARIMA', 'VAR', 'DFM', 'DDFM']:
            if model_name in forecast_data:
                # Only plot non-NaN values
                valid_mask = ~np.isnan(forecast_data[model_name])
                if np.any(valid_mask):
                    valid_dates = forecast_dates[valid_mask]
                    valid_values = forecast_data[model_name][valid_mask]
                    ax.plot(valid_dates, valid_values, '--', linewidth=1.5, 
                           label=model_name, color=colors.get(model_name, 'gray'), alpha=0.8)
        
        # Add vertical line at start of forecast period (2024-01-01)
        ax.axvline(x=forecast_start, color='red', linestyle=':', linewidth=1, alpha=0.5, 
                  label='Forecast Start (2024-01)')
        
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel(f'{target} Value (Original Scale)', fontsize=11)
        ax.set_title(f'Forecast vs Actual: {target} (Original Scale)', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        
        # Format x-axis dates
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
        # Create placeholder on error
        if save_path is None:
            save_path = IMAGES_DIR / f"forecast_vs_actual_{target.lower().replace('.', '_')}.png"
        _create_placeholder_plot(f'Error generating plot for {target}\n{str(e)}', 
                                 save_path=save_path)
        print(f"Generated error placeholder: {save_path.name}")


def plot_nowcasting_comparison(target: str, save_path: Optional[Path] = None):
    """Create nowcasting comparison plot (Plot4: fig:nowcasting_comparison).
    
    For each target, create side-by-side plots comparing "4 weeks before" vs "1 week before" nowcasting.
    Each plot shows 22 months (2024-01 to 2025-10) of predictions and actual values.
    
    Parameters
    ----------
    target : str
        Target series name (KOEQUIPTE, KOWRCCNSE, or KOIPALL.G)
    save_path : Path, optional
        Output path for the plot
    """
    # Get transformation type and base value for inverse transformation
    trans_type = _get_transformation_type(target)
    base_value = _get_base_value(target)
    
    # Load all backtest JSON files for this target
    data_by_timepoint = _load_backtest_results(target, OUTPUTS_DIR)
    
    # Check if we have any data
    if not any(data_by_timepoint.get(tp, {}) for tp in ['4weeks', '1weeks']):
        if save_path is None:
            save_path = IMAGES_DIR / f"nowcasting_comparison_{target.lower().replace('.', '_')}.png"
        _create_placeholder_plot(f'Placeholder: No backtest data for {target}', 
                                 figsize=(14, 5), save_path=save_path, n_subplots=2)
        return
    
    # Convert to format needed for this plot: {timepoint: {model: {month: [predictions]}}}
    # Separate predictions by model (DFM and DDFM)
    predictions_by_timepoint_model = {'4weeks': {'DFM': {}, 'DDFM': {}}, '1weeks': {'DFM': {}, 'DDFM': {}}}
    actual_values = {}
    
    # Load backtest results separately for each model
    for model in ['dfm', 'ddfm']:
        model_upper = model.upper()
        model_data = _load_backtest_results(target, OUTPUTS_DIR, model_filter=model)
        
        for tp in ['4weeks', '1weeks']:
            for month_str, month_data in model_data.get(tp, {}).items():
                if month_str not in predictions_by_timepoint_model[tp][model_upper]:
                    predictions_by_timepoint_model[tp][model_upper][month_str] = []
                forecasts = month_data.get('forecasts', [])
                if forecasts:
                    predictions_by_timepoint_model[tp][model_upper][month_str].extend(forecasts)
                if month_str not in actual_values:
                    actual_values[month_str] = month_data.get('actual')
    
    # Check if we have any data
    has_data = False
    for tp in ['4weeks', '1weeks']:
        for model in ['DFM', 'DDFM']:
            if predictions_by_timepoint_model[tp][model]:
                has_data = True
                break
        if has_data:
            break
    
    if not has_data:
        if save_path is None:
            save_path = IMAGES_DIR / f"nowcasting_comparison_{target.lower().replace('.', '_')}.png"
        _create_placeholder_plot(f'Placeholder: No backtest data for {target}', 
                                 figsize=(14, 5), save_path=save_path, n_subplots=2)
        return
    
    # Prepare data for plotting
    # Sort months chronologically
    months = sorted(actual_values.keys())
    
    # Calculate model-specific predictions for each timepoint
    dfm_predictions_4weeks = []
    dfm_predictions_1weeks = []
    ddfm_predictions_4weeks = []
    ddfm_predictions_1weeks = []
    actual_vals = []
    
    for month in months:
        # DFM - 4 weeks before
        preds_dfm_4w = predictions_by_timepoint_model['4weeks']['DFM'].get(month, [])
        if preds_dfm_4w:
            avg_dfm_4w = np.mean([p for p in preds_dfm_4w if p is not None and not np.isnan(p)])
            dfm_predictions_4weeks.append(avg_dfm_4w)
        else:
            dfm_predictions_4weeks.append(np.nan)
        
        # DFM - 1 week before
        preds_dfm_1w = predictions_by_timepoint_model['1weeks']['DFM'].get(month, [])
        if preds_dfm_1w:
            avg_dfm_1w = np.mean([p for p in preds_dfm_1w if p is not None and not np.isnan(p)])
            dfm_predictions_1weeks.append(avg_dfm_1w)
        else:
            dfm_predictions_1weeks.append(np.nan)
        
        # DDFM - 4 weeks before
        preds_ddfm_4w = predictions_by_timepoint_model['4weeks']['DDFM'].get(month, [])
        if preds_ddfm_4w:
            avg_ddfm_4w = np.mean([p for p in preds_ddfm_4w if p is not None and not np.isnan(p)])
            ddfm_predictions_4weeks.append(avg_ddfm_4w)
        else:
            ddfm_predictions_4weeks.append(np.nan)
        
        # DDFM - 1 week before
        preds_ddfm_1w = predictions_by_timepoint_model['1weeks']['DDFM'].get(month, [])
        if preds_ddfm_1w:
            avg_ddfm_1w = np.mean([p for p in preds_ddfm_1w if p is not None and not np.isnan(p)])
            ddfm_predictions_1weeks.append(avg_ddfm_1w)
        else:
            ddfm_predictions_1weeks.append(np.nan)
        
        # Actual
        actual_val = actual_values.get(month)
        actual_vals.append(actual_val)
    
    # Apply inverse transformation to all values if needed
    if trans_type == 'chg':
        # Convert differences to levels
        if len(dfm_predictions_4weeks) > 0:
            dfm_predictions_4weeks = _inverse_transform_chg(np.array(dfm_predictions_4weeks), base_value).tolist()
        if len(dfm_predictions_1weeks) > 0:
            dfm_predictions_1weeks = _inverse_transform_chg(np.array(dfm_predictions_1weeks), base_value).tolist()
        if len(ddfm_predictions_4weeks) > 0:
            ddfm_predictions_4weeks = _inverse_transform_chg(np.array(ddfm_predictions_4weeks), base_value).tolist()
        if len(ddfm_predictions_1weeks) > 0:
            ddfm_predictions_1weeks = _inverse_transform_chg(np.array(ddfm_predictions_1weeks), base_value).tolist()
        if len(actual_vals) > 0:
            actual_vals_clean = [v if v is not None and not np.isnan(v) else 0.0 for v in actual_vals]
            actual_vals = _inverse_transform_chg(np.array(actual_vals_clean), base_value).tolist()
    
    # Convert month strings to datetime for x-axis
    month_dates = []
    for month_str in months:
        try:
            # Assume format like "2024-01" or "2024-01-31"
            if len(month_str) == 7:  # "2024-01"
                dt = pd.to_datetime(month_str + "-01")
            else:
                dt = pd.to_datetime(month_str)
            month_dates.append(dt)
        except:
            # Fallback: use index
            month_dates.append(pd.Timestamp('2024-01-01') + pd.DateOffset(months=len(month_dates)))
    
    # Create side-by-side plots (similar to attached image)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Get target display name
    target_name = _get_target_display_name(target)
    
    # Calculate y-axis limits (include all series)
    all_values = [v for v in actual_vals if v is not None and not np.isnan(v)]
    all_values.extend([v for v in dfm_predictions_4weeks if v is not None and not np.isnan(v)])
    all_values.extend([v for v in dfm_predictions_1weeks if v is not None and not np.isnan(v)])
    all_values.extend([v for v in ddfm_predictions_4weeks if v is not None and not np.isnan(v)])
    all_values.extend([v for v in ddfm_predictions_1weeks if v is not None and not np.isnan(v)])
    y_min = min(all_values) - 1 if all_values else -2
    y_max = max(all_values) + 1 if all_values else 2
    
    # Plot 1: 4 weeks before
    ax1 = axes[0]
    # Actual value: blue solid line
    ax1.plot(month_dates, actual_vals, 'b-', linewidth=2, label=f'{target_name} (Actual)', alpha=0.9)
    # DFM nowcast: orange dashed line with circle markers
    ax1.plot(month_dates, dfm_predictions_4weeks, '--', color='#FF8C00', marker='o', 
            linewidth=1.5, markersize=5, label='DFM', alpha=0.9, markeredgewidth=1)
    # DDFM nowcast: red dashed line with square markers
    ax1.plot(month_dates, ddfm_predictions_4weeks, '--', color='#d62728', marker='s', 
            linewidth=1.5, markersize=5, label='DDFM', alpha=0.9, markeredgewidth=1)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Value (%)', fontsize=11)
    ax1.set_title(f'{target_name} Nowcasting (4 weeks before)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim([y_min, y_max])
    # Format x-axis as YYYY.MM
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: pd.Timestamp(x).strftime('%Y.%m')))
    fig.autofmt_xdate()
    
    # Plot 2: 1 week before
    ax2 = axes[1]
    # Actual value: blue solid line
    ax2.plot(month_dates, actual_vals, 'b-', linewidth=2, label=f'{target_name} (Actual)', alpha=0.9)
    # DFM nowcast: orange dashed line with circle markers
    ax2.plot(month_dates, dfm_predictions_1weeks, '--', color='#FF8C00', marker='o', 
            linewidth=1.5, markersize=5, label='DFM', alpha=0.9, markeredgewidth=1)
    # DDFM nowcast: red dashed line with square markers
    ax2.plot(month_dates, ddfm_predictions_1weeks, '--', color='#d62728', marker='s', 
            linewidth=1.5, markersize=5, label='DDFM', alpha=0.9, markeredgewidth=1)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Value (%)', fontsize=11)
    ax2.set_title(f'{target_name} Nowcasting (1 week before)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_ylim([y_min, y_max])
    # Format x-axis as YYYY.MM
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: pd.Timestamp(x).strftime('%Y.%m')))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = IMAGES_DIR / f"nowcasting_comparison_{target.lower().replace('.', '_')}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {save_path.name}")


def plot_nowcasting_trend_and_error(target: str, save_path: Optional[Path] = None):
    """Create nowcasting trend and forecast error comparison plot.
    
    Left plot: Nowcasting trend over time (actual vs nowcast at 4 weeks and 1 week before)
    Right plot: Average forecast error by forecast point (4 weeks vs 1 week before)
    
    Parameters
    ----------
    target : str
        Target series name (KOEQUIPTE, KOWRCCNSE, or KOIPALL.G)
    save_path : Path, optional
        Output path for the plot
    """
    # Get transformation type and base value for inverse transformation
    trans_type = _get_transformation_type(target)
    base_value = _get_base_value(target)
    # Load all backtest JSON files for this target
    data_by_timepoint = _load_backtest_results(target, OUTPUTS_DIR)
    
    # Check if we have any data
    if not any(data_by_timepoint.get(tp, {}) for tp in ['4weeks', '1weeks']):
        # No backtest results, create placeholder
        if save_path is None:
            save_path = IMAGES_DIR / f"nowcasting_trend_error_{target.lower().replace('.', '_')}.png"
        _create_placeholder_plot(f'Placeholder: No backtest data for {target}', 
                                 figsize=(14, 5), save_path=save_path, n_subplots=2)
        return
    
    # Check if we have any data
    if not any(data_by_timepoint.get(tp, {}) for tp in ['4weeks', '1weeks']):
        if save_path is None:
            save_path = IMAGES_DIR / f"nowcasting_trend_error_{target.lower().replace('.', '_')}.png"
        _create_placeholder_plot(f'Placeholder: No backtest data for {target}', 
                                 figsize=(14, 5), save_path=save_path, n_subplots=2)
        return
    
    # Prepare data for plotting
    # Get all months (from both timepoints)
    all_months = set()
    for tp in ['4weeks', '1weeks']:
        all_months.update(data_by_timepoint.get(tp, {}).keys())
    months = sorted(all_months)
    
    # Convert month strings to datetime
    month_dates = []
    for month_str in months:
        try:
            if len(month_str) == 7:  # "2024-01"
                dt = pd.to_datetime(month_str + "-01")
            else:
                dt = pd.to_datetime(month_str)
            month_dates.append(dt)
        except:
            month_dates.append(pd.Timestamp('2024-01-01') + pd.DateOffset(months=len(month_dates)))
    
    # Prepare trend data (average forecasts and actuals)
    avg_forecasts_4w = []
    avg_forecasts_1w = []
    actual_vals = []
    
    for month in months:
        # 4 weeks before
        month_data_4w = data_by_timepoint.get('4weeks', {}).get(month, {})
        forecasts_4w = month_data_4w.get('forecasts', [])
        if forecasts_4w:
            avg_forecasts_4w.append(np.mean([f for f in forecasts_4w if not np.isnan(f)]))
        else:
            avg_forecasts_4w.append(np.nan)
        
        # 1 week before
        month_data_1w = data_by_timepoint.get('1weeks', {}).get(month, {})
        forecasts_1w = month_data_1w.get('forecasts', [])
        if forecasts_1w:
            avg_forecasts_1w.append(np.mean([f for f in forecasts_1w if not np.isnan(f)]))
        else:
            avg_forecasts_1w.append(np.nan)
        
        # Actual (use from either timepoint)
        actual = month_data_4w.get('actual') or month_data_1w.get('actual')
        actual_vals.append(actual)
    
    # Calculate forecast errors for error comparison plot
    errors_4w = []
    errors_1w = []
    
    for i, month in enumerate(months):
        actual = actual_vals[i]
        if actual is None or np.isnan(actual):
            errors_4w.append(np.nan)
            errors_1w.append(np.nan)
            continue
        
        # 4 weeks before error
        forecast_4w = avg_forecasts_4w[i]
        if forecast_4w is not None and not np.isnan(forecast_4w):
            errors_4w.append(abs(forecast_4w - actual))
        else:
            errors_4w.append(np.nan)
        
        # 1 week before error
        forecast_1w = avg_forecasts_1w[i]
        if forecast_1w is not None and not np.isnan(forecast_1w):
            errors_1w.append(abs(forecast_1w - actual))
        else:
            errors_1w.append(np.nan)
    
    # Calculate average errors for each timepoint
    # Apply inverse transformation to forecasts and actual values if needed
    if trans_type == 'chg':
        if len(avg_forecasts_4w) > 0:
            avg_forecasts_4w = _inverse_transform_chg(np.array(avg_forecasts_4w), base_value).tolist()
        if len(avg_forecasts_1w) > 0:
            avg_forecasts_1w = _inverse_transform_chg(np.array(avg_forecasts_1w), base_value).tolist()
        if len(actual_vals) > 0:
            actual_vals_clean = [v if v is not None and not np.isnan(v) else 0.0 for v in actual_vals]
            actual_vals = _inverse_transform_chg(np.array(actual_vals_clean), base_value).tolist()
        # Recalculate errors after inverse transformation
        errors_4w = [avg_forecasts_4w[i] - actual_vals[i] if i < len(actual_vals) and actual_vals[i] is not None and not np.isnan(actual_vals[i]) else np.nan for i in range(len(avg_forecasts_4w))]
        errors_1w = [avg_forecasts_1w[i] - actual_vals[i] if i < len(actual_vals) and actual_vals[i] is not None and not np.isnan(actual_vals[i]) else np.nan for i in range(len(avg_forecasts_1w))]
    
    avg_error_4w = np.nanmean(errors_4w) if any(not np.isnan(e) for e in errors_4w) else np.nan
    avg_error_1w = np.nanmean(errors_1w) if any(not np.isnan(e) for e in errors_1w) else np.nan
    
    # Target name mapping
    target_names = {
        'KOEQUIPTE': 'Equipment Investment',
        'KOWRCCNSE': 'Wholesale/Retail Sales',
        'KOIPALL.G': 'Industrial Production'
    }
    target_name = target_names.get(target, target)
    
    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Nowcasting trend (left)
    ax1 = axes[0]
    # Actual values: red squares (like in image)
    actual_valid = [(d, v) for d, v in zip(month_dates, actual_vals) if v is not None and not np.isnan(v)]
    if actual_valid:
        actual_dates, actual_values = zip(*actual_valid)
        ax1.plot(actual_dates, actual_values, 'rs', markersize=8, label='Actual', alpha=0.9, zorder=3)
    
    # 4 weeks before: gray dashed line (Bloomberg-like)
    ax1.plot(month_dates, avg_forecasts_4w, '--', color='gray', linewidth=1.5, 
            label='4 weeks before', alpha=0.8)
    
    # 1 week before: yellow solid line (DFM.i-like)
    ax1.plot(month_dates, avg_forecasts_1w, '-', color='#FFD700', linewidth=2, 
            label='1 week before (DFM.i)', alpha=0.9)
    
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Value (%)', fontsize=11)
    ax1.set_title(f'{target_name} Nowcasting Trend', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: pd.Timestamp(x).strftime('%Y.%m')))
    fig.autofmt_xdate()
    
    # Plot 2: Forecast error comparison (right)
    ax2 = axes[1]
    # Plot average errors for each timepoint
    weeks_labels = ['4 weeks', '1 week']
    avg_errors = [avg_error_4w, avg_error_1w]
    colors = ['gray', '#FFD700']
    linestyles = ['--', '-']
    
    ax2.plot(weeks_labels, avg_errors, '-', color='#FFD700', linewidth=2, 
            marker='o', markersize=8, label='DFM.i', alpha=0.9)
    
    ax2.set_xlabel('Forecast Timepoint', fontsize=11)
    ax2.set_ylabel('Average Forecast Error', fontsize=11)
    ax2.set_title('Nowcasting Average Forecast Error by Timepoint', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--', axis='y')
    ax2.set_ylim(bottom=0)
    
    # Add note
    fig.text(0.5, 0.02, f'Note: Average forecast error for 2024-01 to 2025-10', 
            ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    if save_path is None:
        save_path = IMAGES_DIR / f"nowcasting_trend_error_{target.lower().replace('.', '_')}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {save_path.name}")


def generate_all_plots():
    """Generate all plots required by WORKFLOW.md.
    
    Plot1: forecast_vs_actual (3 plots, one per target, 22 months)
    Plot3: horizon_trend (1-22 months, sMSE)
    Plot4: nowcasting_comparison (3 pairs, one per target, 22 months)
    Plot5: nowcasting_trend_and_error (3 plots, one per target)
    """
    print("=" * 70)
    print("Generating Report Images (WORKFLOW.md)")
    print("=" * 70)
    
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("\n1. Loading comparison results...")
    all_results = _load_comparison_results(OUTPUTS_DIR)
    print(f"   Found results for {len(all_results)} target series")
    
    # Generate images
    print("\n2. Generating images...")
    
    # Define targets once
    targets = ['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
    
    # Plot1: Forecast vs actual (one plot per target)
    print("\n   Plot1: Forecast vs Actual (3 plots)")
    for target in targets:
        print(f"   - forecast_vs_actual_{target.lower().replace('.', '_')}.png")
        plot_forecast_vs_actual(target)
    
    # Plot3: Horizon trend (1-22 months, sMSE)
    print("\n   Plot3: Horizon Performance Trend (1-22 months, sMSE)")
    print("   - horizon_trend.png")
    plot_horizon_trend()
    
    # Plot4: Nowcasting comparison (one pair per target)
    print("\n   Plot4: Nowcasting Comparison (3 pairs)")
    for target in targets:
        print(f"   - nowcasting_comparison_{target.lower().replace('.', '_')}.png")
        plot_nowcasting_comparison(target)
    
    # Plot5: Nowcasting trend and error comparison (one plot per target)
    print("\n   Plot5: Nowcasting Trend and Error Comparison (3 plots)")
    for target in targets:
        print(f"   - nowcasting_trend_error_{target.lower().replace('.', '_')}.png")
        plot_nowcasting_trend_and_error(target)
    
    print("\n" + "=" * 70)
    print("Image generation complete!")
    print("=" * 70)
    print(f"\nImages saved in: {IMAGES_DIR}")


if __name__ == "__main__":
    generate_all_plots()
