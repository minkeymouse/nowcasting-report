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


def load_comparison_results(outputs_dir: Path) -> Dict[str, List[Dict]]:
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
        except Exception as e:
            continue
    
    return all_results


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
    horizons = [1, 7, 28]
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


def plot_model_comparison(save_path: Optional[Path] = None):
    """Create model comparison bar plot (fig:model_comparison)."""
    # Load data
    all_results = load_comparison_results(OUTPUTS_DIR)
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
            save_path = IMAGES_DIR / "model_comparison.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated placeholder: {save_path.name}")
        return
    
    # Aggregate by model (average across all targets and horizons)
    model_avg = df.groupby(['model', 'metric'])['value'].mean().reset_index()
    model_avg_pivot = model_avg.pivot(index='model', columns='metric', values='value')
    
    # Sort by sRMSE (ascending)
    model_avg_pivot = model_avg_pivot.sort_values('sRMSE')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_avg_pivot))
    width = 0.25
    
    metrics = ['sMSE', 'sMAE', 'sRMSE']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, model_avg_pivot[metric].values, width, 
               label=metric, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Standardized Metric Value', fontsize=11)
    ax.set_title('Model Performance Comparison (Standardized Metrics)', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_avg_pivot.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = IMAGES_DIR / "model_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {save_path.name}")


def plot_horizon_trend(save_path: Optional[Path] = None):
    """Create horizon trend plot (fig:horizon_trend)."""
    # Load data
    all_results = load_comparison_results(OUTPUTS_DIR)
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
    
    # Aggregate by model and horizon
    # Exclude None values
    df_valid = df[df['value'].notna()].copy()
    if len(df_valid) == 0:
        # Already handled above, but keep for safety
        return
    
    horizon_avg = df_valid.groupby(['model', 'horizon', 'metric'])['value'].mean().reset_index()
    horizon_avg_pivot = horizon_avg[horizon_avg['metric'] == 'sRMSE'].pivot(
        index='model', columns='horizon', values='value'
    )
    
    if horizon_avg_pivot.empty:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    horizons = [1, 7, 28]
    for model in horizon_avg_pivot.index:
        values = []
        for h in horizons:
            if h in horizon_avg_pivot.columns:
                val = horizon_avg_pivot.loc[model, h]
                values.append(val if pd.notna(val) else None)
            else:
                values.append(None)
        # Only plot if we have at least one valid value
        if any(v is not None and not np.isnan(v) for v in values if v is not None):
            ax.plot(horizons, values, marker='o', label=model, linewidth=2, markersize=6)
    
    ax.set_xlabel('Forecast Horizon (days)', fontsize=11)
    ax.set_ylabel('Standardized RMSE', fontsize=11)
    ax.set_title('Performance Trend by Forecast Horizon (Standardized RMSE)', fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(horizons)
    
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
    all_results = load_comparison_results(OUTPUTS_DIR)
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
    
    Shows 60 months total: 30 months of original series (single line) before cutoff,
    then 30 months of forecasts (5 lines: original, ARIMA, VAR, DFM, DDFM) after cutoff.
    
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
    
    # Add src and dfm-python to path for imports
    project_root = Path(__file__).parent.parent.parent
    src_path = project_root / "src"
    dfm_path = project_root / "dfm-python" / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(dfm_path) not in sys.path:
        sys.path.insert(0, str(dfm_path))
    
    # Load comparison results to get model paths
    all_results = load_comparison_results(OUTPUTS_DIR)
    
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
        data = pd.read_csv(data_file, parse_dates=['date'], index_col='date')
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
        
        # Get last 30 months of training data for historical plot
        n_historical_months = min(30, len(y_train_monthly))
        y_historical = y_train_monthly.iloc[-n_historical_months:]
        
        # Get first 30 months of test data for forecast period
        n_forecast_months = min(30, len(y_test_monthly))
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
                project_root = Path(__file__).parent.parent.parent
                src_path = project_root / "src"
                dfm_path = project_root / "dfm-python" / "src"
                if str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))
                if str(dfm_path) not in sys.path:
                    sys.path.insert(0, str(dfm_path))
                
                # Load model
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                forecaster = model_data.get('forecaster')
                if forecaster is None:
                    continue
                
                # Generate forecasts for test period
                # Fit on training data
                try:
                    forecaster.fit(y_train_data)
                    # Predict for entire test period at once
                    n_forecast = len(y_test_data)
                    pred = forecaster.predict(fh=list(range(1, n_forecast + 1)))
                    
                    # Extract predictions
                    if isinstance(pred, pd.DataFrame):
                        if target in pred.columns:
                            forecast_series = pred[target]
                        else:
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
                    continue
                    
            except Exception as e:
                print(f"Warning: Failed to load/generate forecasts for {model_key}: {e}")
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


def generate_all_plots():
    """Generate all plots required by RESULTS_NEEDED.md section 6."""
    print("=" * 70)
    print("Generating Report Images")
    print("=" * 70)
    
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print("\n1. Loading comparison results...")
    all_results = load_comparison_results(OUTPUTS_DIR)
    print(f"   Found results for {len(all_results)} target series")
    
    # Generate images
    print("\n2. Generating images...")
    
    # Model comparison
    print("   - model_comparison.png")
    plot_model_comparison()
    
    # Horizon trend
    print("   - horizon_trend.png")
    plot_horizon_trend()
    
    # Heatmap
    print("   - accuracy_heatmap.png")
    plot_accuracy_heatmap()
    
    # Forecast vs actual (one plot per target)
    targets = ['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
    for target in targets:
        print(f"   - forecast_vs_actual_{target.lower().replace('.', '_')}.png")
        plot_forecast_vs_actual(target)
    
    print("\n" + "=" * 70)
    print("Image generation complete!")
    print("=" * 70)
    print(f"\nImages saved in: {IMAGES_DIR}")


if __name__ == "__main__":
    generate_all_plots()
