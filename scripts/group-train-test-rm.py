import argparse
import logging
import os
import glob
from pathlib import Path
import subprocess
import yaml


def setup_logging():
    """Configure logging for the main process."""
    log_dir = Path("./experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(log_dir, "reward_model_process.log"),
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


def run_training_test(config_path, logger):
    """Run training and testing for a single reward model config file."""
    # Load config to get experiment name for logging
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("experiment_name", os.path.basename(config_path))

    logger.info(f"Starting reward model training for experiment: {experiment_name}")
    logger.info(f"Config file: {config_path}")

    # Run training
    train_cmd = ["python", "scripts/reward_model_train.py", "--config", config_path]
    try:
        logger.info(f"Running command: {' '.join(train_cmd)}")
        subprocess.run(train_cmd, check=True)
        logger.info(
            f"Reward model training completed successfully for {experiment_name}"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Reward model training failed for {experiment_name}: {e}")
        return False

    # Run testing
    logger.info(f"Starting reward model testing for experiment: {experiment_name}")
    test_cmd = ["python", "scripts/test_reward_model.py", "--config", config_path]
    try:
        logger.info(f"Running command: {' '.join(test_cmd)}")
        subprocess.run(test_cmd, check=True)
        logger.info(
            f"Reward model testing completed successfully for {experiment_name}"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Reward model testing failed for {experiment_name}: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run reward model training and testing with multiple config files"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="./experiments/reward_models",
        help="Directory containing YAML configuration files for reward models",
    )
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("Starting reward model process")

    config_files = get_config_files(args.config_dir)
    if not config_files:
        logger.error(f"No config files found in {args.config_dir}")
        return

    logger.info(f"Found {len(config_files)} reward model config files")

    results = {}
    for config_file in config_files:
        logger.info(f"Processing reward model config: {config_file}")
        success = run_training_test(config_file, logger)
        results[os.path.basename(config_file)] = "Success" if success else "Failed"

    # Log summary
    logger.info("===== Reward Model Summary =====")
    for config, status in results.items():
        logger.info(f"{config}: {status}")
    logger.info("===============================")


if __name__ == "__main__":
    main()
