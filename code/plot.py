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
    
    models = ['ARIMA', 'VAR', 'VECM', 'DeepAR', 'TFT', 'XGBoost', 'LightGBM', 'DFM', 'DDFM']
    targets = ['KOGDP...D', 'KOCNPER.D', 'KOGFCF..D']
    horizons = [1, 7, 28]
    metrics = ['sMSE', 'sMAE', 'sRMSE']
    
    for target in targets:
        for model in models:
            for horizon in horizons:
                for metric in metrics:
                    # Try to find actual result
                    value = None
                    if target in all_results:
                        for result_data in all_results[target]:
                            comparison = result_data.get('comparison')
                            if comparison:
                                metrics_table = comparison.get('metrics_table')
                                if metrics_table:
                                    if isinstance(metrics_table, dict):
                                        metrics_table = pd.DataFrame([metrics_table])
                                    elif not isinstance(metrics_table, pd.DataFrame):
                                        continue
                                    
                                    for _, row in metrics_table.iterrows():
                                        if row.get('model', '').lower() == model.lower():
                                            col_name = f"{metric}_h{horizon}"
                                            if col_name in row and pd.notna(row[col_name]):
                                                value = row[col_name]
                                                break
                    
                    # Use None for missing values (will be displayed as "-")
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        value = None
                    
                    rows.append({
                        'model': model,
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
    if df['value'].isna().all():
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
    if df['value'].isna().all():
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
    if df['value'].isna().all():
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
    
    # Rename targets for display
    target_avg_pivot.columns = ['GDP', 'Consumption', 'Investment']
    
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


def plot_forecast_vs_actual(save_path: Optional[Path] = None, target: str = 'GDP'):
    """Create forecast vs actual time series plot (fig:forecast_vs_actual)."""
    # No actual data available, create placeholder text image
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0.5, 0.5, 'Placeholder: No data available', 
            ha='center', va='center', fontsize=16, color='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    if save_path is None:
        save_path = IMAGES_DIR / "forecast_vs_actual.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Generated placeholder: {save_path.name}")


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
    
    # Forecast vs actual
    print("   - forecast_vs_actual.png")
    plot_forecast_vs_actual()
    
    print("\n" + "=" * 70)
    print("Image generation complete!")
    print("=" * 70)
    print(f"\nImages saved in: {IMAGES_DIR}")


if __name__ == "__main__":
    generate_all_plots()
