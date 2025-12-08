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
    """Create forecast vs actual time series plot for a specific target."""
    project_root = _setup_import_paths()
    
    from src.evaluation import collect_all_comparison_results
    all_results = collect_all_comparison_results(OUTPUTS_DIR)
    
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
        
        hist_start = pd.Timestamp('2023-01-01')
        hist_end = pd.Timestamp('2023-12-31')
        forecast_start = pd.Timestamp('2024-01-01')
        forecast_end = pd.Timestamp('2025-10-31')
        
        y_historical = y_full[(y_full.index >= hist_start) & (y_full.index <= hist_end)]
        y_historical_monthly = y_historical.resample('ME').last()
        y_actual_forecast = y_full[(y_full.index >= forecast_start) & (y_full.index <= forecast_end)]
        y_actual_forecast_monthly = y_actual_forecast.resample('ME').last()
        
        trans_type = _get_transformation_type(target)
        
        # Get base value for historical data
        hist_base_data = y_full[(y_full.index < hist_start)]
        hist_base_value = 0.0
        if len(hist_base_data) > 0:
            hist_base_monthly = hist_base_data.resample('ME').last()
            hist_base_values = hist_base_monthly[target].dropna()
            if len(hist_base_values) > 0:
                cumulative_before_2023 = np.cumsum(hist_base_values.values)
                hist_base_value = float(cumulative_before_2023[-1]) if len(cumulative_before_2023) > 0 else 0.0
        
        hist_values_transformed = y_historical_monthly[target].values
        hist_values_original = None
        if trans_type == 'chg' and len(hist_values_transformed) > 0:
            hist_values_original = _inverse_transform_chg(hist_values_transformed, hist_base_value)
            base_value = float(hist_values_original[-1])
        else:
            if len(hist_values_transformed) > 0:
                base_value = float(hist_values_transformed[-1])
            else:
                base_value = _get_base_value(target, forecast_start)
        
        forecast_data = {}
        actual_transformed = y_actual_forecast_monthly[target].values
        if trans_type == 'chg':
            forecast_data['Actual'] = _inverse_transform_chg(actual_transformed, base_value)
        else:
            forecast_data['Actual'] = actual_transformed
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        hist_dates = y_historical_monthly.index
        if trans_type == 'chg' and hist_values_original is not None:
            hist_values = hist_values_original
        else:
            hist_values = hist_values_transformed
        
        ax.plot(hist_dates, hist_values, color='gray', linestyle='-', linewidth=2, 
               label='Historical (2023)', alpha=0.8)
        
        forecast_dates = y_actual_forecast_monthly.index
        n_forecast_periods = len(forecast_dates)
        
        for model_name in forecast_data.keys():
            if len(forecast_data[model_name]) > n_forecast_periods:
                forecast_data[model_name] = forecast_data[model_name][:n_forecast_periods]
            elif len(forecast_data[model_name]) < n_forecast_periods:
                padded = np.full(n_forecast_periods, np.nan)
                padded[:len(forecast_data[model_name])] = forecast_data[model_name]
                forecast_data[model_name] = padded
        
        ax.plot(forecast_dates, forecast_data['Actual'], 'k-', linewidth=2.5, 
               label='Actual (2024-2025)', alpha=0.9)
        
        ax.axvline(x=forecast_start, color='red', linestyle=':', linewidth=1, alpha=0.5, 
                  label='Forecast Start (2024-01)')
        
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel(f'{target} Value (Original Scale)', fontsize=11)
        ax.set_title(f'Forecast vs Actual: {target} (Original Scale)', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
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
    
    print("\n2. Horizon Performance Trend")
    print("   - horizon_trend.png")
    plot_horizon_trend()
    
    print("\n" + "=" * 70)
    print("Forecast plot generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    generate_forecast_plots()

