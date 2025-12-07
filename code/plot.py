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


def _load_backtest_results(target: str, outputs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all backtest JSON files for a target.
    
    Returns:
        Dictionary with structure: {timepoint: {month: {forecasts: [], actual: value}}}
    """
    backtest_dir = outputs_dir / "backtest"
    if not backtest_dir.exists():
        return {}
    
    models = ['arima', 'var', 'dfm', 'ddfm']
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
    """Extract metrics from comparison results into a DataFrame."""
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
        # No data, create placeholder text image
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Placeholder: No data available', 
                ha='center', va='center', fontsize=16, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        if save_path is None:
            save_path = IMAGES_DIR / "horizon_trend.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated placeholder: {save_path.name}")
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


def plot_accuracy_heatmap(save_path: Optional[Path] = None):
    """Create accuracy heatmap (fig:accuracy_heatmap)."""
    # Load data
    all_results = _load_comparison_results(OUTPUTS_DIR)
    df = extract_metrics_from_results(all_results)
    
    # Check if we have any valid data
    if df.empty or 'value' not in df.columns or df['value'].isna().all():
        # No data, create placeholder text image
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'Placeholder: No data available', 
                ha='center', va='center', fontsize=16, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        if save_path is None:
            save_path = IMAGES_DIR / "accuracy_heatmap.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated placeholder: {save_path.name}")
        return
    
    # Aggregate by model and target (average across horizons)
    # Exclude None values
    df_valid = df[df['value'].notna()].copy()
    if len(df_valid) == 0:
        # Already handled above, but keep for safety
        return
    
    target_avg = df_valid.groupby(['model', 'target', 'metric'])['value'].mean().reset_index()
    target_avg_pivot = target_avg[target_avg['metric'] == 'sRMSE'].pivot(
        index='model', columns='target', values='value'
    )
    
    if target_avg_pivot.empty:
        return
    
    # Rename targets for display (only rename existing columns)
    target_name_map = {
        'KOEQUIPTE': 'Equipment Investment',
        'KOWRCCNSE': 'Wholesale/Retail Sales',
        'KOIPALL.G': 'Industrial Production'
    }
    # Rename columns that exist
    new_columns = []
    for col in target_avg_pivot.columns:
        new_columns.append(target_name_map.get(col, col))
    target_avg_pivot.columns = new_columns
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(target_avg_pivot, annot=True, fmt='.3f', cmap='YlOrRd_r', 
                cbar_kws={'label': 'Standardized RMSE'}, ax=ax, linewidths=0.5, 
                mask=target_avg_pivot.isna())
    
    ax.set_xlabel('Target Variable', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)
    ax.set_title('Prediction Accuracy Heatmap by Target Variable (Standardized RMSE)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = IMAGES_DIR / "accuracy_heatmap.png"
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
    import pickle
    import sys
    from pathlib import Path as PathLib
    
    # Add project root, src and dfm-python to path for imports
    project_root = Path(__file__).parent.parent.parent
    src_path = project_root / "src"
    dfm_path = project_root / "dfm-python" / "src"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(dfm_path) not in sys.path:
        sys.path.insert(0, str(dfm_path))
    
    # Load comparison results to get model paths
    from src.evaluation import collect_all_comparison_results
    all_results = collect_all_comparison_results(OUTPUTS_DIR)
    
    if target not in all_results or not all_results[target]:
        # No data, create placeholder
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.text(0.5, 0.5, f'Placeholder: No data available for {target}', 
                ha='center', va='center', fontsize=16, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        if save_path is None:
            save_path = IMAGES_DIR / f"forecast_vs_actual_{target.lower().replace('.', '_')}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated placeholder: {save_path.name}")
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
        
        # Recreate train/test split (80/20 as in training.py)
        split_idx = int(len(y_full) * 0.8)
        y_train_data = y_full.iloc[:split_idx]
        y_test_data = y_full.iloc[split_idx:]
        
        # Aggregate to monthly for plotting (take last value of each month)
        y_train_monthly = y_train_data.resample('ME').last()
        y_test_monthly = y_test_data.resample('ME').last()
        
        # Get last 22 months of training data for historical plot (optional context)
        n_historical_months = min(22, len(y_train_monthly))
        y_historical = y_train_monthly.iloc[-n_historical_months:]
        
        # Get first 22 months of test data for forecast period (2024-01 to 2025-10)
        n_forecast_months = min(22, len(y_test_monthly))
        y_actual_forecast = y_test_monthly.iloc[:n_forecast_months]
        
        # Prepare forecast data
        forecast_data = {}
        forecast_data['Actual'] = y_actual_forecast[target].values
        
        # Load models and generate forecasts
        models_to_load = ['arima', 'var', 'dfm', 'ddfm']
        for model_key in models_to_load:
            if model_key not in results:
                continue
            
            model_info = results[model_key]
            model_dir_str = model_info.get('model_dir', '')
            if not model_dir_str:
                continue
            
            model_dir = PathLib(model_dir_str)
            model_file = model_dir / "model.pkl"
            
            if not model_file.exists():
                print(f"Warning: Model file not found: {model_file}")
                continue
            
            try:
                # Set up paths for imports (needed for unpickling)
                import sys
                import os
                project_root = Path(__file__).parent.parent.parent
                src_path = project_root / "src"
                dfm_path = project_root / "dfm-python" / "src"
                
                # Add paths to sys.path if not already there (at the beginning for priority)
                paths_to_add = [
                    str(project_root),  # For 'src' imports
                    str(src_path),      # Direct src path
                    str(dfm_path),      # dfm-python path
                ]
                for path in paths_to_add:
                    if path not in sys.path:
                        sys.path.insert(0, path)
                
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
                        
                        # Predict for entire test period at once
                        n_forecast = len(y_test_data)
                        pred = forecaster.predict(fh=list(range(1, n_forecast + 1)))
                        
                        # Extract predictions for target series
                        if isinstance(pred, pd.DataFrame):
                            if target in pred.columns:
                                forecast_series = pred[target]
                            else:
                                # Try to find target by case-insensitive match
                                target_lower = target.lower()
                                matching_cols = [c for c in pred.columns if c.lower() == target_lower]
                                if matching_cols:
                                    forecast_series = pred[matching_cols[0]]
                                else:
                                    # Use first column as fallback
                                    forecast_series = pred.iloc[:, 0]
                        elif isinstance(pred, pd.Series):
                            forecast_series = pred
                        else:
                            forecast_series = pd.Series(pred.flatten() if hasattr(pred, 'flatten') else pred)
                        
                        # Align index with test data
                        if len(forecast_series) == len(y_test_data):
                            forecast_series.index = y_test_data.index[:len(forecast_series)]
                        else:
                            # Create index if needed
                            forecast_series.index = y_test_data.index[:len(forecast_series)]
                        
                        # Aggregate to monthly (take last value of each month)
                        forecast_monthly = forecast_series.resample('ME').last()
                        forecast_data[model_key.upper()] = forecast_monthly.iloc[:n_forecast_months].values
                        
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
        
        # Plot historical data (single line)
        hist_dates = y_historical.index
        ax.plot(hist_dates, y_historical[target].values, 'k-', linewidth=2, label='Historical (Actual)', alpha=0.7)
        
        # Plot forecast period
        forecast_dates = y_actual_forecast.index
        
        # Plot actual values in forecast period
        ax.plot(forecast_dates, forecast_data['Actual'], 'k-', linewidth=2, label='Actual', alpha=0.7)
        
        # Plot model forecasts
        colors = {'ARIMA': '#1f77b4', 'VAR': '#ff7f0e', 'DFM': '#2ca02c', 'DDFM': '#d62728'}
        for model_name in ['ARIMA', 'VAR', 'DFM', 'DDFM']:
            if model_name in forecast_data:
                ax.plot(forecast_dates, forecast_data[model_name], '--', linewidth=1.5, 
                       label=model_name, color=colors.get(model_name, 'gray'), alpha=0.8)
        
        # Add vertical line at train/test split
        split_date = y_train_data.index[-1]
        ax.axvline(x=split_date, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Train/Test Split')
        
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel(f'{target} Value', fontsize=11)
        ax.set_title(f'Forecast vs Actual: {target}', fontsize=13, fontweight='bold')
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
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.text(0.5, 0.5, f'Error generating plot for {target}\n{str(e)}', 
                ha='center', va='center', fontsize=12, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        
        if save_path is None:
            save_path = IMAGES_DIR / f"forecast_vs_actual_{target.lower().replace('.', '_')}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
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
    # Load all backtest JSON files for this target
    data_by_timepoint = _load_backtest_results(target, OUTPUTS_DIR)
    
    # Check if we have any data
    if not any(data_by_timepoint.get(tp, {}) for tp in ['4weeks', '1weeks']):
        # No backtest results, create placeholder
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax in axes:
            ax.text(0.5, 0.5, f'Placeholder: No backtest data for {target}', 
                    ha='center', va='center', fontsize=14, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        if save_path is None:
            save_path = IMAGES_DIR / f"nowcasting_comparison_{target.lower().replace('.', '_')}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated placeholder: {save_path.name}")
        return
    
    # Convert to format needed for this plot: {timepoint: {month: [predictions]}}
    predictions_by_timepoint = {'4weeks': {}, '1weeks': {}}
    actual_values = {}
    
    for tp in ['4weeks', '1weeks']:
        for month_str, month_data in data_by_timepoint.get(tp, {}).items():
            predictions_by_timepoint[tp][month_str] = month_data.get('forecasts', [])
            if month_str not in actual_values:
                actual_values[month_str] = month_data.get('actual')
    
    # Check if we have any data
    if not any(predictions_by_timepoint.get(tp, {}) for tp in ['4weeks', '1weeks']):
        # No data, create placeholder
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax in axes:
            ax.text(0.5, 0.5, f'Placeholder: No backtest data for {target}', 
                    ha='center', va='center', fontsize=14, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        if save_path is None:
            save_path = IMAGES_DIR / f"nowcasting_comparison_{target.lower().replace('.', '_')}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated placeholder: {save_path.name}")
        return
    
    # Prepare data for plotting
    # Sort months chronologically
    months = sorted(actual_values.keys())
    
    # Calculate model average predictions for each timepoint
    avg_predictions_4weeks = []
    avg_predictions_1weeks = []
    actual_vals = []
    
    for month in months:
        # 4 weeks before
        preds_4w = predictions_by_timepoint['4weeks'].get(month, [])
        if preds_4w:
            avg_4w = np.mean([p for p in preds_4w if p is not None and not np.isnan(p)])
            avg_predictions_4weeks.append(avg_4w)
        else:
            avg_predictions_4weeks.append(np.nan)
        
        # 1 week before
        preds_1w = predictions_by_timepoint['1weeks'].get(month, [])
        if preds_1w:
            avg_1w = np.mean([p for p in preds_1w if p is not None and not np.isnan(p)])
            avg_predictions_1weeks.append(avg_1w)
        else:
            avg_predictions_1weeks.append(np.nan)
        
        # Actual
        actual_vals.append(actual_values.get(month))
    
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
    
    # Target name mapping for display
    target_names = {
        'KOEQUIPTE': 'Equipment Investment',
        'KOWRCCNSE': 'Wholesale/Retail Sales',
        'KOIPALL.G': 'Industrial Production'
    }
    target_name = target_names.get(target, target)
    
    # Plot 1: 4 weeks before
    ax1 = axes[0]
    # Actual value: blue solid line
    ax1.plot(month_dates, actual_vals, 'b-', linewidth=2, label=f'{target_name} (Actual)', alpha=0.9)
    # Nowcast: orange dashed line with circle markers (like DFM.w.i in image)
    ax1.plot(month_dates, avg_predictions_4weeks, '--', color='#FF8C00', marker='o', 
            linewidth=1.5, markersize=5, label='DFM.w.i', alpha=0.9, markeredgewidth=1)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Value (%)', fontsize=11)
    ax1.set_title(f'{target_name} Nowcasting (4 weeks before)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_ylim([min(min(actual_vals), min(avg_predictions_4weeks)) - 1, 
                  max(max(actual_vals), max(avg_predictions_4weeks)) + 1])
    # Format x-axis as YYYY.MM
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: pd.Timestamp(x).strftime('%Y.%m')))
    fig.autofmt_xdate()
    
    # Plot 2: 1 week before
    ax2 = axes[1]
    # Actual value: blue solid line
    ax2.plot(month_dates, actual_vals, 'b-', linewidth=2, label=f'm{target_name_kr}, %', alpha=0.9)
    # Nowcast: orange dashed line with circle markers
    ax2.plot(month_dates, avg_predictions_1weeks, '--', color='#FF8C00', marker='o', 
            linewidth=1.5, markersize=5, label='DFM.w.i', alpha=0.9, markeredgewidth=1)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Value (%)', fontsize=11)
    ax2.set_title(f'{target_name} Nowcasting (1 week before)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_ylim([min(min(actual_vals), min(avg_predictions_1weeks)) - 1, 
                  max(max(actual_vals), max(avg_predictions_1weeks)) + 1])
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
    # Load all backtest JSON files for this target
    data_by_timepoint = _load_backtest_results(target, OUTPUTS_DIR)
    
    # Check if we have any data
    if not any(data_by_timepoint.get(tp, {}) for tp in ['4weeks', '1weeks']):
        # No backtest results, create placeholder
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax in axes:
            ax.text(0.5, 0.5, f'Placeholder: No backtest data for {target}', 
                    ha='center', va='center', fontsize=14, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        if save_path is None:
            save_path = IMAGES_DIR / f"nowcasting_trend_error_{target.lower().replace('.', '_')}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated placeholder: {save_path.name}")
        return
    
    # Check if we have any data
    if not any(data_by_timepoint.get(tp, {}) for tp in ['4weeks', '1weeks']):
        # No data, create placeholder
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax in axes:
            ax.text(0.5, 0.5, f'Placeholder: No backtest data for {target}', 
                    ha='center', va='center', fontsize=14, color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        if save_path is None:
            save_path = IMAGES_DIR / f"nowcasting_trend_error_{target.lower().replace('.', '_')}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated placeholder: {save_path.name}")
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
    fig.text(0.5, 0.02, f'주: 2024-01~2025-10 평균 예측오차', 
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
    Plot2: accuracy_heatmap
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
    
    # Plot1: Forecast vs actual (one plot per target)
    print("\n   Plot1: Forecast vs Actual (3 plots)")
    targets = ['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
    for target in targets:
        print(f"   - forecast_vs_actual_{target.lower().replace('.', '_')}.png")
        plot_forecast_vs_actual(target)
    
    # Plot2: Accuracy heatmap
    print("\n   Plot2: Accuracy Heatmap")
    print("   - accuracy_heatmap.png")
    plot_accuracy_heatmap()
    
    # Plot3: Horizon trend (1-22 months, sMSE)
    print("\n   Plot3: Horizon Performance Trend (1-22 months, sMSE)")
    print("   - horizon_trend.png")
    plot_horizon_trend()
    
    # Plot4: Nowcasting comparison (one pair per target)
    print("\n   Plot4: Nowcasting Comparison (3 pairs)")
    targets = ['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
    for target in targets:
        print(f"   - nowcasting_comparison_{target.lower().replace('.', '_')}.png")
        plot_nowcasting_comparison(target)
    
    # Plot5: Nowcasting trend and error comparison (one plot per target)
    print("\n   Plot5: Nowcasting Trend and Error Comparison (3 plots)")
    targets = ['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
    for target in targets:
        print(f"   - nowcasting_trend_error_{target.lower().replace('.', '_')}.png")
        plot_nowcasting_trend_and_error(target)
    
    print("\n" + "=" * 70)
    print("Image generation complete!")
    print("=" * 70)
    print(f"\nImages saved in: {IMAGES_DIR}")


if __name__ == "__main__":
    generate_all_plots()
