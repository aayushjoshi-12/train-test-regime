import argparse
import logging
import os
import glob
from pathlib import Path
import subprocess
import yaml


def setup_logging():
    """Configure logging for the evaluation process."""
    log_dir = Path("./experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(log_dir, "group_eval_process.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    # Also print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    
    return logging.getLogger(__name__)

def get_config_files(config_dir):
    """Get all YAML config files in the specified directory."""
    return sorted(glob.glob(os.path.join(config_dir, "*.yaml")))

def run_evaluation(config_path, logger):
    """Run evaluation for a single config file."""
    # Load config to get experiment name for logging
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    experiment_name = config.get("experiment_name", os.path.basename(config_path))
    
    logger.info(f"Starting evaluation for experiment: {experiment_name}")
    logger.info(f"Config file: {config_path}")
    
    # Run evaluation
    eval_cmd = ["python", "scripts/unsloth_eval.py", "--config", config_path]
    try:
        logger.info(f"Running command: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True)
        logger.info(f"Evaluation completed successfully for {experiment_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed for {experiment_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run evaluation with multiple config files")
    parser.add_argument(
        "--config-dir",
        type=str,
        default="./experiments",
        help="Directory containing YAML configuration files",
    )
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("Starting group evaluation process")
    
    config_files = get_config_files(args.config_dir)
    if not config_files:
        logger.error(f"No config files found in {args.config_dir}")
        return
    
    logger.info(f"Found {len(config_files)} config files")
    
    results = {}
    for config_file in config_files:
        logger.info(f"Processing config: {config_file}")
        success = run_evaluation(config_file, logger)
        results[os.path.basename(config_file)] = "Success" if success else "Failed"
    
    # Log summary
    logger.info("===== Evaluation Summary =====")
    for config, status in results.items():
        logger.info(f"{config}: {status}")
    logger.info("===========================")

if __name__ == "__main__":
    main()