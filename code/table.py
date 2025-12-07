"""Table generation code for nowcasting report LaTeX tables.

This module generates all LaTeX tables required by the report:
- tab_dataset_params.tex: Dataset details and model parameters
- tab_forecasting_results.tex: Forecasting results by model-horizon and target-metric
- tab_nowcasting_backtest.tex: Nowcasting backtest results by model-timepoint and target-metric

All tables are saved to nowcasting-report/tables/ directory.
"""

from pathlib import Path
import sys
import pandas as pd
import json
import logging

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import table generation functions from src.evaluation
from src.evaluation import (
    generate_latex_table_dataset_params,
    generate_latex_table_forecasting_results,
    generate_latex_table_nowcasting_backtest,
    aggregate_overall_performance,
    collect_all_comparison_results
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = PROJECT_ROOT / "nowcasting-report" / "tables"
CONFIG_DIR = PROJECT_ROOT / "config"


def generate_all_tables(
    outputs_dir: Path = OUTPUTS_DIR,
    tables_dir: Path = TABLES_DIR,
    config_dir: Path = CONFIG_DIR
) -> dict:
    """Generate all LaTeX tables for the report.
    
    Parameters
    ----------
    outputs_dir : Path
        Directory containing experiment outputs (comparisons, backtest)
    tables_dir : Path
        Directory to save LaTeX table files
    config_dir : Path
        Directory containing config files for dataset parameters
        
    Returns
    -------
    dict
        Dictionary mapping table names to LaTeX code strings
    """
    logger.info("=" * 70)
    logger.info("Generating LaTeX Tables for Nowcasting Report")
    logger.info("=" * 70)
    
    # Ensure tables directory exists
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    tables = {}
    
    # Table 1: Dataset and model parameters
    logger.info("\n[1/3] Generating tab_dataset_params.tex...")
    try:
        tables['dataset_params'] = generate_latex_table_dataset_params(
            config_dir=config_dir,
            output_path=tables_dir / "tab_dataset_params.tex"
        )
        logger.info("  ✓ Dataset parameters table generated")
    except Exception as e:
        logger.error(f"  ✗ Failed to generate dataset parameters table: {e}")
        tables['dataset_params'] = None
    
    # Table 2: Forecasting results
    logger.info("\n[2/3] Generating tab_forecasting_results.tex...")
    try:
        # Load aggregated results
        aggregated_file = outputs_dir / "experiments" / "aggregated_results.csv"
        if not aggregated_file.exists():
            logger.warning(f"  ⚠ Aggregated results file not found: {aggregated_file}")
            logger.info("  Attempting to aggregate from comparison results...")
            # Try to aggregate from comparison results
            all_results = collect_all_comparison_results(outputs_dir)
            if all_results:
                aggregated_df = aggregate_overall_performance(all_results)
                logger.info(f"  Aggregated {len(aggregated_df)} rows from comparison results")
            else:
                logger.error("  ✗ No comparison results found")
                aggregated_df = pd.DataFrame()
        else:
            aggregated_df = pd.read_csv(aggregated_file)
            logger.info(f"  Loaded {len(aggregated_df)} rows from aggregated_results.csv")
        
        if not aggregated_df.empty:
            tables['forecasting_results'] = generate_latex_table_forecasting_results(
                aggregated_df=aggregated_df,
                output_path=tables_dir / "tab_forecasting_results.tex"
            )
            logger.info("  ✓ Forecasting results table generated")
        else:
            logger.warning("  ⚠ No data available for forecasting results table")
            tables['forecasting_results'] = None
    except Exception as e:
        logger.error(f"  ✗ Failed to generate forecasting results table: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        tables['forecasting_results'] = None
    
    # Table 3: Nowcasting backtest results
    logger.info("\n[3/3] Generating tab_nowcasting_backtest.tex...")
    try:
        tables['nowcasting_backtest'] = generate_latex_table_nowcasting_backtest(
            outputs_dir=outputs_dir,
            output_path=tables_dir / "tab_nowcasting_backtest.tex"
        )
        logger.info("  ✓ Nowcasting backtest table generated")
    except Exception as e:
        logger.error(f"  ✗ Failed to generate nowcasting backtest table: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        tables['nowcasting_backtest'] = None
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Table Generation Summary")
    logger.info("=" * 70)
    successful = sum(1 for v in tables.values() if v is not None)
    total = len(tables)
    logger.info(f"  Successful: {successful}/{total}")
    logger.info(f"  Tables saved to: {tables_dir}")
    
    if successful == total:
        logger.info("  ✓ All tables generated successfully!")
    else:
        logger.warning(f"  ⚠ {total - successful} table(s) failed to generate")
    
    return tables


def main():
    """Main entry point for table generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for nowcasting report")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Directory containing experiment outputs (default: project_root/outputs)"
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=TABLES_DIR,
        help="Directory to save LaTeX tables (default: project_root/nowcasting-report/tables)"
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=CONFIG_DIR,
        help="Directory containing config files (default: project_root/config)"
    )
    
    args = parser.parse_args()
    
    generate_all_tables(
        outputs_dir=args.outputs_dir,
        tables_dir=args.tables_dir,
        config_dir=args.config_dir
    )


if __name__ == "__main__":
    main()

