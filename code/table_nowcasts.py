"""Table generation code for nowcasting-related LaTeX tables.

This module generates nowcasting-related LaTeX tables:
- tab_nowcasting_backtest.tex: Nowcasting backtest results by model-timepoint and target-metric

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


def generate_nowcasting_backtest_table(
    outputs_dir: Path = OUTPUTS_DIR,
    output_path: Path = None
) -> str:
    """Generate nowcasting backtest results table from backtest JSON files."""
    if output_path is None:
        output_path = TABLES_DIR / "tab_nowcasting_backtest.tex"
    
    backtest_dir = outputs_dir / "backtest"
    if not backtest_dir.exists():
        logger.warning(f"Backtest directory not found: {backtest_dir}")
        # Generate placeholder table
        latex = r"""\begin{table}[h]
\centering
\caption{Nowcasting Backtest Results}
\label{tab:nowcasting_backtest}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{KOIPALL.G} & \multicolumn{2}{c}{KOEQUIPTE} & \multicolumn{2}{c}{KOWRCCNSE} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
Model-Timepoint & sMAE & sMSE & sMAE & sMSE & sMAE & sMSE \\
\midrule
DFM-4weeks & N/A & N/A & N/A & N/A & N/A & N/A \\
DFM-1week & N/A & N/A & N/A & N/A & N/A & N/A \\
DDFM-4weeks & N/A & N/A & N/A & N/A & N/A & N/A \\
DDFM-1week & N/A & N/A & N/A & N/A & N/A & N/A \\
\bottomrule
\end{tabular}
\vspace{0.5em}
\small\textit{Note: Backtest data not available. Run nowcasting pipeline first.}
\end{table}
"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)
        return latex
    
    # Load backtest results
    targets = ['KOIPALL.G', 'KOEQUIPTE', 'KOWRCCNSE']
    models = ['dfm', 'ddfm']
    timepoints = ['4weeks', '1weeks']
    
    results = {}
    for target in targets:
        results[target] = {}
        for model in models:
            backtest_file = backtest_dir / f"{target}_{model}_backtest.json"
            if not backtest_file.exists():
                continue
            
            try:
                with open(backtest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Try new structure first (results_by_timepoint)
                results_by_tp = data.get('results_by_timepoint', {})
                if results_by_tp:
                    # New structure with timepoint organization
                    for tp in timepoints:
                        if tp not in results_by_tp:
                            continue
                        
                        tp_data = results_by_tp[tp]
                        key = f"{model.upper()}-{tp}"
                        if key not in results[target]:
                            results[target][key] = {}
                        
                        # Calculate average metrics
                        monthly_results = tp_data.get('monthly_results', [])
                        if monthly_results:
                            errors = []
                            for m in monthly_results:
                                # Check if this is a successful result with forecast and actual values
                                if m.get('status') == 'ok' or (m.get('forecast_value') is not None and m.get('actual_value') is not None):
                                    forecast_val = m.get('forecast_value')
                                    actual_val = m.get('actual_value')
                                    if forecast_val is not None and actual_val is not None:
                                        try:
                                            error = float(forecast_val) - float(actual_val)
                                            if not np.isnan(error) and not np.isinf(error):
                                                errors.append(abs(error))
                                        except (ValueError, TypeError):
                                            pass
                            if errors:
                                results[target][key]['sMAE'] = np.mean(errors)
                                results[target][key]['sMSE'] = np.mean([e**2 for e in errors])
                else:
                    # Fallback: Check for flat results structure (old format or failed backtests)
                    flat_results = data.get('results', [])
                    if flat_results:
                        # Check if all results are failed
                        all_failed = all(r.get('status') == 'failed' for r in flat_results)
                        if all_failed:
                            # All failed - no metrics to calculate, will show N/A
                            continue
                        # If some succeeded, we'd need to reconstruct timepoint structure
                        # For now, skip flat structure (requires re-run with fixed code)
                        logger.debug(f"Skipping flat results structure for {backtest_file.name} - needs results_by_timepoint structure")
            except Exception as e:
                logger.warning(f"Failed to load {backtest_file}: {e}")
                continue
    
    # Generate LaTeX table
    latex = r"""\begin{table}[h]
\centering
\caption{Nowcasting Backtest Results by Model-Timepoint}
\label{tab:nowcasting_backtest}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{KOIPALL.G} & \multicolumn{2}{c}{KOEQUIPTE} & \multicolumn{2}{c}{KOWRCCNSE} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
Model-Timepoint & sMAE & sMSE & sMAE & sMSE & sMAE & sMSE \\
\midrule
"""
    
    row_keys = ['DFM-4weeks', 'DFM-1weeks', 'DDFM-4weeks', 'DDFM-1weeks']
    for row_key in row_keys:
        row = row_key.replace('weeks', ' week' if '1' in row_key else ' weeks')
        for target in targets:
            if row_key in results.get(target, {}):
                smae = results[target][row_key].get('sMAE')
                smse = results[target][row_key].get('sMSE')
                if smae is not None and smse is not None:
                    row += f" & {smae:.2f} & {smse:.2f}"
                else:
                    row += " & N/A & N/A"
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


def generate_nowcast_tables():
    """Generate all nowcasting-related LaTeX tables."""
    logger.info("=" * 70)
    logger.info("Generating Nowcasting LaTeX Tables")
    logger.info("=" * 70)
    
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n[1/1] Generating tab_nowcasting_backtest.tex...")
    try:
        generate_nowcasting_backtest_table()
        logger.info("  ✓ Nowcasting backtest table generated")
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Table generation complete!")
    logger.info(f"Tables saved to: {TABLES_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    generate_nowcast_tables()
