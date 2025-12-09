"""Table generation code for forecasting-related LaTeX tables.

This module generates forecasting-related LaTeX tables:
- tab_dataset_params.tex: Dataset details and model parameters
- tab_forecasting_results.tex: Forecasting results by model-target

All tables are saved to nowcasting-report/tables/ directory.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = PROJECT_ROOT / "nowcasting-report" / "tables"
CONFIG_DIR = PROJECT_ROOT / "config"


def generate_dataset_params_table(
    config_dir: Path = CONFIG_DIR,
    output_path: Path = None
) -> str:
    """Generate dataset parameters table from config files."""
    if output_path is None:
        output_path = TABLES_DIR / "tab_dataset_params.tex"
    
    # Define dataset parameters (from experiment configs)
    # n_series: total series count, n_weekly_agg: weekly variables aggregated to monthly
    params = {
        'KOIPALL.G (생산)': {'n_series': 45, 'n_weekly_agg': 7, 'train_period': '1985-2019', 'forecast_period': '2024-2025'},
        'KOEQUIPTE (투자)': {'n_series': 41, 'n_weekly_agg': 9, 'train_period': '1985-2019', 'forecast_period': '2024-2025'},
        'KOWRCCNSE (소비)': {'n_series': 47, 'n_weekly_agg': 8, 'train_period': '1985-2019', 'forecast_period': '2024-2025'},
    }
    
    latex = r"""\begin{table}[h]
\centering
\caption{Dataset and Model Parameters}
\label{tab:dataset_params}
\begin{tabular}{lcccc}
\toprule
Target Variable & Series Count & Weekly Agg. & Training Period & Forecast Period \\
\midrule
"""
    
    for target, p in params.items():
        latex += f"{target} & {p['n_series']} & {p['n_weekly_agg']} & {p['train_period']} & {p['forecast_period']} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\vspace{0.3em}
\small\textit{Note: Weekly Agg. = 월말 평균으로 집계된 주간 변수 수 (\_agg 접미사)}
\end{table}
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    
    return latex


def generate_forecasting_results_table(
    outputs_dir: Path = OUTPUTS_DIR,
    output_path: Path = None
) -> str:
    """Generate forecasting results table from aggregated results."""
    if output_path is None:
        output_path = TABLES_DIR / "tab_forecasting_results.tex"
    
    # Load aggregated results
    aggregated_file = outputs_dir / "experiments" / "aggregated_results.csv"
    if not aggregated_file.exists():
        logger.warning(f"Aggregated results file not found: {aggregated_file}")
        return ""
    
    df = pd.read_csv(aggregated_file)
    
    # Group by model and target, average across horizons
    summary = df.groupby(['model', 'target']).agg({
        'sMSE': 'mean',
        'sMAE': 'mean'
    }).reset_index()
    
    # Pivot for table format
    models = ['VAR', 'DFM', 'DDFM']  # Skip ARIMA (failed)
    targets = ['KOIPALL.G', 'KOEQUIPTE', 'KOWRCCNSE']
    
    latex = r"""\begin{table}[h]
\centering
\caption{Forecasting Results by Model-Target (Average across Horizons)}
\label{tab:forecasting_results}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{KOIPALL.G} & \multicolumn{2}{c}{KOEQUIPTE} & \multicolumn{2}{c}{KOWRCCNSE} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
Model & sMAE & sMSE & sMAE & sMSE & sMAE & sMSE \\
\midrule
"""
    
    for model in models:
        row = f"{model}"
        for target in targets:
            mask = (summary['model'] == model) & (summary['target'] == target)
            if mask.any():
                smae = summary.loc[mask, 'sMAE'].values[0]
                smse = summary.loc[mask, 'sMSE'].values[0]
                # Format with 2 decimal places
                row += f" & {smae:.2f} & {smse:.2f}"
            else:
                row += " & N/A & N/A"
        row += r" \\" + "\n"
        latex += row
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    
    return latex


def generate_appendix_table(
    outputs_dir: Path = OUTPUTS_DIR,
    output_path: Path = None,
    target: str = None
) -> str:
    """Generate appendix table with all horizons."""
    if output_path is None:
        suffix = target.lower().replace('.', '_') if target else 'all'
        output_path = TABLES_DIR / f"tab_appendix_forecasting_{suffix}.tex"
    
    # Load aggregated results
    aggregated_file = outputs_dir / "experiments" / "aggregated_results.csv"
    if not aggregated_file.exists():
        return ""
    
    df = pd.read_csv(aggregated_file)
    
    if target:
        df = df[df['target'] == target]
        title = f"Forecasting Results: {target}"
    else:
        # Average across targets
        df = df.groupby(['model', 'horizon']).agg({
            'sMSE': 'mean',
            'sMAE': 'mean'
        }).reset_index()
        title = "Forecasting Results: All Targets (Average)"
    
    models = ['VAR', 'DFM', 'DDFM']
    horizons = list(range(1, 23))
    
    latex = f"""\\begin{{table}}[h]
\\centering
\\caption{{{title}}}
\\label{{tab:appendix_forecasting_{target.lower().replace('.', '_') if target else 'all'}}}
\\small
\\begin{{tabular}}{{l{'c' * len(models) * 2}}}
\\toprule
"""
    
    # Header row
    header = "Horizon"
    for model in models:
        header += f" & \\multicolumn{{2}}{{c}}{{{model}}}"
    header += r" \\" + "\n"
    latex += header
    
    # Sub-header
    subheader = ""
    for model in models:
        subheader += f" & sMAE & sMSE"
    subheader += r" \\" + "\n"
    latex += subheader
    latex += r"\midrule" + "\n"
    
    # Data rows
    for h in horizons:
        row = f"{h}"
        for model in models:
            mask = (df['model'] == model) & (df['horizon'] == h)
            if mask.any():
                smae = df.loc[mask, 'sMAE'].values[0]
                smse = df.loc[mask, 'sMSE'].values[0]
                if pd.notna(smae) and pd.notna(smse):
                    row += f" & {smae:.2f} & {smse:.2f}"
                else:
                    row += " & - & -"
            else:
                row += " & - & -"
        row += r" \\" + "\n"
        latex += row
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    
    return latex


def generate_forecast_tables():
    """Generate all forecasting-related LaTeX tables."""
    logger.info("=" * 70)
    logger.info("Generating Forecasting LaTeX Tables")
    logger.info("=" * 70)
    
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Table 1: Dataset parameters
    logger.info("\n[1/6] Generating tab_dataset_params.tex...")
    try:
        generate_dataset_params_table()
        logger.info("  ✓ Dataset parameters table generated")
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
    
    # Table 2: Forecasting results summary
    logger.info("\n[2/6] Generating tab_forecasting_results.tex...")
    try:
        generate_forecasting_results_table()
        logger.info("  ✓ Forecasting results table generated")
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
    
    # Appendix tables
    targets = ['KOIPALL.G', 'KOEQUIPTE', 'KOWRCCNSE', None]
    for idx, target in enumerate(targets, start=3):
        name = target if target else "all"
        logger.info(f"\n[{idx}/6] Generating tab_appendix_forecasting_{name.lower().replace('.', '_') if target else 'all'}.tex...")
        try:
            generate_appendix_table(target=target)
            logger.info(f"  ✓ Appendix table ({name}) generated")
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Table generation complete!")
    logger.info(f"Tables saved to: {TABLES_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    generate_forecast_tables()
