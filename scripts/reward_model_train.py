import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from trl import RewardConfig, RewardTrainer


def setup_logging(experiment_name, log_dir="./experiments/logs"):
    """Configure logging for the training process."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"{experiment_name}.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def format_data(example, tokenizer, max_length):
    """Format a data row into the expected training format."""
    kwargs = {
        "padding": "max_length",
        "truncation": True,
        "max_length": max_length,
        "return_tensors": "pt",
    }

    instruction = example["instruction"]
    choice_w = example["choice_w"]
    choice_l = example["choice_l"]

    prompt_plus_response_chosen = f"{instruction}\n{choice_w}"
    prompt_plus_response_rejected = f"{instruction}\n{choice_l}"

    tokens_chosen = tokenizer.encode_plus(prompt_plus_response_chosen, **kwargs)
    tokens_rejected = tokenizer.encode_plus(prompt_plus_response_rejected, **kwargs)

    return {
        "input_ids_chosen": tokens_chosen["input_ids"][0],
        "attention_mask_chosen": tokens_chosen["attention_mask"][0],
        "input_ids_rejected": tokens_rejected["input_ids"][0],
        "attention_mask_rejected": tokens_rejected["attention_mask"][0],
    }


def prepare_dataset(cfg, tokenizer):
    """Load and prepare dataset for training."""
    dataset = load_dataset(
        "csv",
        data_files=cfg["data_file"],
    )
    
    # Apply formatting to the dataset
    formatted_dataset = dataset.map(
        lambda example: format_data(example, tokenizer, cfg["max_length"])
    )
    
    return formatted_dataset


def initialize_model(cfg):
    """Initialize and configure the model and tokenizer."""
    # Initialize quantization config if specified
    quantization_config = None
    if cfg.get("quantization", {}).get("enabled", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=cfg["quantization"].get("load_in_4bit", False),
            bnb_4bit_use_double_quant=cfg["quantization"].get("use_double_quant", True),
            bnb_4bit_quant_type=cfg["quantization"].get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, cfg["quantization"].get("compute_dtype", "bfloat16"))
        )
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        quantization_config=quantization_config,
        device_map="auto" if cfg.get("use_device_map", True) else None,
        token=os.getenv("HF_TOKEN"),
    )
    
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    
    # Handle padding tokens
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer


def create_lora_config(cfg):
    """Create LoRA configuration based on config file."""
    return LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        task_type=TaskType.SEQ_CLS
    )


def create_reward_config(cfg):
    """Create reward training configuration based on config file."""
    return RewardConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 1),
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=cfg["training"]["weight_decay"],
        num_train_epochs=cfg["training"]["num_epochs"],
        fp16=not torch.cuda.is_bf16_supported() and cfg["training"].get("mixed_precision", True),
        bf16=torch.cuda.is_bf16_supported() and cfg["training"].get("mixed_precision", True),
        optim=cfg["training"].get("optim", "adamw_torch"),
        lr_scheduler_type=cfg["training"].get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.1),
    )


def train_model(config_path):
    """Main training function."""
    cfg = load_config(config_path)
    logger = setup_logging(cfg["experiment_name"])

    # Initialize model and tokenizer
    logger.info(f"Initializing model: {cfg['model_name']}")
    model, tokenizer = initialize_model(cfg)

    # Prepare dataset
    logger.info("Preparing dataset")
    dataset = prepare_dataset(cfg, tokenizer)

    # Setup LoRA and reward configs
    lora_config = create_lora_config(cfg)
    reward_config = create_reward_config(cfg)

    # Initialize trainer
    logger.info("Setting up reward trainer")
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        peft_config=lora_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation") if "validation" in dataset else None,
    )

    # Train model
    logger.info("Training started...")
    trainer.train()
    logger.info("Training completed.")

    # Save model
    model_save_path = cfg["output_dir"]
    logger.info(f"Saving model to {model_save_path}...")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Model saved successfully to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reward model with configuration")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    train_model(args.config)
