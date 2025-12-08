"""Plot creation code for nowcasting report visualizations.

This module generates nowcasting-related plots:
- plot_nowcasting_comparison: Side-by-side 4-week vs 1-week nowcasting comparison
- plot_nowcasting_trend_and_error: Nowcasting trend and forecast error comparison

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
    project_root = Path(__file__).parent.parent.parent
    if metadata_file is None:
        metadata_file = project_root / "data" / "metadata.csv"
    
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
    project_root = Path(__file__).parent.parent.parent
    if data_file is None:
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


def _get_target_display_name(target: str) -> str:
    """Get display name for a target series."""
    target_names = {
        'KOEQUIPTE': 'Equipment Investment',
        'KOWRCCNSE': 'Wholesale/Retail Sales',
        'KOIPALL.G': 'Industrial Production'
    }
    return target_names.get(target, target)


def _load_backtest_results(target: str, outputs_dir: Path, model_filter: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load all backtest JSON files for a target."""
    backtest_dir = outputs_dir / "backtest"
    if not backtest_dir.exists():
        return {}
    
    if model_filter:
        models = [model_filter.lower()]
    else:
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
                    
                    if data_by_timepoint[tp][month_str]['actual'] is None:
                        data_by_timepoint[tp][month_str]['actual'] = month_data.get('actual_value')
        except Exception as e:
            print(f"Warning: Failed to load {backtest_file}: {e}")
            continue
    
    return data_by_timepoint


def plot_nowcasting_comparison(target: str, save_path: Optional[Path] = None):
    """Create nowcasting comparison plot - side-by-side 4-week vs 1-week comparison."""
    trans_type = _get_transformation_type(target)
    base_value = _get_base_value(target)
    
    data_by_timepoint = _load_backtest_results(target, OUTPUTS_DIR)
    
    if not any(data_by_timepoint.get(tp, {}) for tp in ['4weeks', '1weeks']):
        if save_path is None:
            save_path = IMAGES_DIR / f"nowcasting_comparison_{target.lower().replace('.', '_')}.png"
        _create_placeholder_plot(f'Placeholder: No backtest data for {target}', 
                                 figsize=(14, 5), save_path=save_path, n_subplots=2)
        return
    
    predictions_by_timepoint_model = {'4weeks': {'DFM': {}, 'DDFM': {}}, '1weeks': {'DFM': {}, 'DDFM': {}}}
    actual_values = {}
    
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
    
    has_data = any(predictions_by_timepoint_model[tp][model] 
                   for tp in ['4weeks', '1weeks'] for model in ['DFM', 'DDFM'])
    
    if not has_data:
        if save_path is None:
            save_path = IMAGES_DIR / f"nowcasting_comparison_{target.lower().replace('.', '_')}.png"
        _create_placeholder_plot(f'Placeholder: No backtest data for {target}', 
                                 figsize=(14, 5), save_path=save_path, n_subplots=2)
        return
    
    months = sorted(actual_values.keys())
    
    dfm_predictions_4weeks, dfm_predictions_1weeks = [], []
    ddfm_predictions_4weeks, ddfm_predictions_1weeks = [], []
    actual_vals = []
    
    for month in months:
        for model, pred_list_4w, pred_list_1w in [
            ('DFM', dfm_predictions_4weeks, dfm_predictions_1weeks),
            ('DDFM', ddfm_predictions_4weeks, ddfm_predictions_1weeks)
        ]:
            preds_4w = predictions_by_timepoint_model['4weeks'][model].get(month, [])
            if preds_4w:
                pred_list_4w.append(np.mean([p for p in preds_4w if p is not None and not np.isnan(p)]))
            else:
                pred_list_4w.append(np.nan)
            
            preds_1w = predictions_by_timepoint_model['1weeks'][model].get(month, [])
            if preds_1w:
                pred_list_1w.append(np.mean([p for p in preds_1w if p is not None and not np.isnan(p)]))
            else:
                pred_list_1w.append(np.nan)
        
        actual_vals.append(actual_values.get(month))
    
    if trans_type == 'chg':
        for lst in [dfm_predictions_4weeks, dfm_predictions_1weeks, 
                    ddfm_predictions_4weeks, ddfm_predictions_1weeks]:
            if len(lst) > 0:
                lst[:] = _inverse_transform_chg(np.array(lst), base_value).tolist()
        if len(actual_vals) > 0:
            actual_vals_clean = [v if v is not None and not np.isnan(v) else 0.0 for v in actual_vals]
            actual_vals[:] = _inverse_transform_chg(np.array(actual_vals_clean), base_value).tolist()
    
    month_dates = []
    for month_str in months:
        try:
            if len(month_str) == 7:
                dt = pd.to_datetime(month_str + "-01")
            else:
                dt = pd.to_datetime(month_str)
            month_dates.append(dt)
        except:
            month_dates.append(pd.Timestamp('2024-01-01') + pd.DateOffset(months=len(month_dates)))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    target_name = _get_target_display_name(target)
    
    all_values = [v for v in actual_vals if v is not None and not np.isnan(v)]
    for lst in [dfm_predictions_4weeks, dfm_predictions_1weeks, 
                ddfm_predictions_4weeks, ddfm_predictions_1weeks]:
        all_values.extend([v for v in lst if v is not None and not np.isnan(v)])
    y_min = min(all_values) - 1 if all_values else -2
    y_max = max(all_values) + 1 if all_values else 2
    
    for ax, tp, dfm_preds, ddfm_preds, title_suffix in [
        (axes[0], '4weeks', dfm_predictions_4weeks, ddfm_predictions_4weeks, '4 weeks before'),
        (axes[1], '1weeks', dfm_predictions_1weeks, ddfm_predictions_1weeks, '1 week before')
    ]:
        ax.plot(month_dates, actual_vals, 'b-', linewidth=2, label=f'{target_name} (Actual)', alpha=0.9)
        ax.plot(month_dates, dfm_preds, '--', color='#FF8C00', marker='o', 
                linewidth=1.5, markersize=5, label='DFM', alpha=0.9)
        ax.plot(month_dates, ddfm_preds, '--', color='#d62728', marker='s', 
                linewidth=1.5, markersize=5, label='DDFM', alpha=0.9)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Value (%)', fontsize=11)
        ax.set_title(f'{target_name} Nowcasting ({title_suffix})', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_ylim([y_min, y_max])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: pd.Timestamp(x).strftime('%Y.%m')))
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    if save_path is None:
        save_path = IMAGES_DIR / f"nowcasting_comparison_{target.lower().replace('.', '_')}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated: {save_path.name}")


def plot_nowcasting_trend_and_error(target: str, save_path: Optional[Path] = None):
    """Create nowcasting trend and forecast error comparison plot."""
    trans_type = _get_transformation_type(target)
    base_value = _get_base_value(target)
    data_by_timepoint = _load_backtest_results(target, OUTPUTS_DIR)
    
    if not any(data_by_timepoint.get(tp, {}) for tp in ['4weeks', '1weeks']):
        if save_path is None:
            save_path = IMAGES_DIR / f"nowcasting_trend_error_{target.lower().replace('.', '_')}.png"
        _create_placeholder_plot(f'Placeholder: No backtest data for {target}', 
                                 figsize=(14, 5), save_path=save_path, n_subplots=2)
        return
    
    all_months = set()
    for tp in ['4weeks', '1weeks']:
        all_months.update(data_by_timepoint.get(tp, {}).keys())
    months = sorted(all_months)
    
    month_dates = []
    for month_str in months:
        try:
            if len(month_str) == 7:
                dt = pd.to_datetime(month_str + "-01")
            else:
                dt = pd.to_datetime(month_str)
            month_dates.append(dt)
        except:
            month_dates.append(pd.Timestamp('2024-01-01') + pd.DateOffset(months=len(month_dates)))
    
    avg_forecasts_4w, avg_forecasts_1w, actual_vals = [], [], []
    
    for month in months:
        for tp, forecasts_list in [('4weeks', avg_forecasts_4w), ('1weeks', avg_forecasts_1w)]:
            month_data = data_by_timepoint.get(tp, {}).get(month, {})
            forecasts = month_data.get('forecasts', [])
            if forecasts:
                forecasts_list.append(np.mean([f for f in forecasts if not np.isnan(f)]))
            else:
                forecasts_list.append(np.nan)
        
        month_data_4w = data_by_timepoint.get('4weeks', {}).get(month, {})
        month_data_1w = data_by_timepoint.get('1weeks', {}).get(month, {})
        actual = month_data_4w.get('actual') or month_data_1w.get('actual')
        actual_vals.append(actual)
    
    if trans_type == 'chg':
        if len(avg_forecasts_4w) > 0:
            avg_forecasts_4w = _inverse_transform_chg(np.array(avg_forecasts_4w), base_value).tolist()
        if len(avg_forecasts_1w) > 0:
            avg_forecasts_1w = _inverse_transform_chg(np.array(avg_forecasts_1w), base_value).tolist()
        if len(actual_vals) > 0:
            actual_vals_clean = [v if v is not None and not np.isnan(v) else 0.0 for v in actual_vals]
            actual_vals = _inverse_transform_chg(np.array(actual_vals_clean), base_value).tolist()
    
    errors_4w = [abs(avg_forecasts_4w[i] - actual_vals[i]) 
                 if actual_vals[i] is not None and not np.isnan(actual_vals[i]) else np.nan 
                 for i in range(len(avg_forecasts_4w))]
    errors_1w = [abs(avg_forecasts_1w[i] - actual_vals[i]) 
                 if actual_vals[i] is not None and not np.isnan(actual_vals[i]) else np.nan 
                 for i in range(len(avg_forecasts_1w))]
    
    avg_error_4w = np.nanmean(errors_4w) if any(not np.isnan(e) for e in errors_4w) else np.nan
    avg_error_1w = np.nanmean(errors_1w) if any(not np.isnan(e) for e in errors_1w) else np.nan
    
    target_name = _get_target_display_name(target)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Nowcasting trend
    ax1 = axes[0]
    actual_valid = [(d, v) for d, v in zip(month_dates, actual_vals) if v is not None and not np.isnan(v)]
    if actual_valid:
        actual_dates, actual_values = zip(*actual_valid)
        ax1.plot(actual_dates, actual_values, 'rs', markersize=8, label='Actual', alpha=0.9, zorder=3)
    ax1.plot(month_dates, avg_forecasts_4w, '--', color='gray', linewidth=1.5, label='4 weeks before', alpha=0.8)
    ax1.plot(month_dates, avg_forecasts_1w, '-', color='#FFD700', linewidth=2, label='1 week before', alpha=0.9)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Value (%)', fontsize=11)
    ax1.set_title(f'{target_name} Nowcasting Trend', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: pd.Timestamp(x).strftime('%Y.%m')))
    
    # Plot 2: Forecast error comparison
    ax2 = axes[1]
    ax2.plot(['4 weeks', '1 week'], [avg_error_4w, avg_error_1w], '-', color='#FFD700', 
            linewidth=2, marker='o', markersize=8, label='Avg Error', alpha=0.9)
    ax2.set_xlabel('Forecast Timepoint', fontsize=11)
    ax2.set_ylabel('Average Forecast Error', fontsize=11)
    ax2.set_title('Nowcasting Average Forecast Error by Timepoint', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--', axis='y')
    ax2.set_ylim(bottom=0)
    
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


def generate_nowcast_plots():
    """Generate all nowcasting-related plots."""
    print("=" * 70)
    print("Generating Nowcast Plots")
    print("=" * 70)
    
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    targets = ['KOEQUIPTE', 'KOWRCCNSE', 'KOIPALL.G']
    
    print("\n1. Nowcasting Comparison (3 pairs)")
    for target in targets:
        print(f"   - nowcasting_comparison_{target.lower().replace('.', '_')}.png")
        plot_nowcasting_comparison(target)
    
    print("\n2. Nowcasting Trend and Error (3 plots)")
    for target in targets:
        print(f"   - nowcasting_trend_error_{target.lower().replace('.', '_')}.png")
        plot_nowcasting_trend_and_error(target)
    
    print("\n" + "=" * 70)
    print("Nowcast plot generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    generate_nowcast_plots()
