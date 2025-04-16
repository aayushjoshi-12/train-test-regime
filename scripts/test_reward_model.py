import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def setup_logging(experiment_name, log_dir="./experiments/logs"):
    """Configure logging for the testing process."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"{experiment_name}_test.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def get_score(model, tokenizer, prompt, response, max_length):
    """Calculate scores for a prompt-response pair."""
    kwargs = {"padding": "max_length", "truncation": True, "max_length": max_length, "return_tensors": "pt"}
    inputs = tokenizer.encode_plus(prompt+"\n"+response, **kwargs).to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.logits

def test_example(model, tokenizer, example, max_length):
    """Test a single example and return result."""
    score_chosen = get_score(model, tokenizer, example['instruction'], example['choice_w'], max_length)
    score_rejected = get_score(model, tokenizer, example['instruction'], example['choice_l'], max_length)

    if score_chosen.sum() > score_rejected.sum():
        return "Works"
    elif score_chosen.sum() < score_rejected.sum():
        return "Fails"
    else:
        return "Tie"

def test_model(config_path):
    """Main testing function."""
    cfg = load_config(config_path)
    logger = setup_logging(cfg["experiment_name"])

    # Load model and tokenizer
    logger.info(f"Loading model from: {cfg['model_path']}")
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    
    # Set padding token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Move model to device
    device = "cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu"
    logger.info(f"Using device: {device}")
    model.to(device)
    model.eval()

    # Load test data
    logger.info(f"Loading test data from: {cfg['test_data_file']}")
    df = pd.read_csv(cfg["test_data_file"])
    
    # Get number of examples to test
    num_examples = cfg.get("num_test_examples", 10)
    if num_examples > len(df):
        num_examples = len(df)
        logger.warning(f"Requested more examples than available. Testing on all {num_examples} examples.")
    
    # Test examples
    results = {"works": 0, "fails": 0, "ties": 0}
    logger.info(f"Testing {num_examples} examples...")
    
    for i, row in df.sample(num_examples).iterrows():
        logger.info(f"Testing example {i+1}/{num_examples}")
        logger.info(f"Instruction: {row['instruction']}")
        logger.info(f"Chosen response: {row['choice_w']}")
        logger.info(f"Rejected response: {row['choice_l']}")
        
        result = test_example(model, tokenizer, row, cfg.get("max_length", 512))
        logger.info(f"Result: {result}")
        
        if result == "Works":
            results["works"] += 1
        elif result == "Fails":
            results["fails"] += 1
        else:
            results["ties"] += 1
        
        # print(f"Example {i+1}:")
        # print(f"Instruction: {row['instruction']}")
        # print(f"Chosen response: {row['choice_w']}")
        # print(f"Rejected response: {row['choice_l']}")
        # print(f"Result: {result}")
        # print("="*60)
    
    # Summary
    success_rate = results["works"] / num_examples * 100
    logger.info(f"Testing completed. Results: {results}")
    logger.info(f"Success rate: {success_rate:.2f}%")
    
    print("\nTesting Summary:")
    print(f"Total examples tested: {num_examples}")
    print(f"Works: {results['works']}")
    print(f"Fails: {results['fails']}")
    print(f"Ties: {results['ties']}")
    print(f"Success rate: {success_rate:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a reward model with configuration")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    test_model(args.config)