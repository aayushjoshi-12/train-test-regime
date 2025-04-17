import unsloth
import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from unsloth import FastLanguageModel
import pandas as pd


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


def format_data(row):
    """Format a data row into the expected training format."""
    return {
        "text": f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{row["system_prompt"]}

cateory: {row["category"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{row["instruction"]}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{row["response"]}<|eot_id|><|end_of_text|>
"""
    }


def prepare_datasets(cfg):
    """Load and prepare datasets for training and validation."""
    train_df = pd.read_csv(os.path.join(cfg["data_dir"], "training_dataset.csv"))
    val_df = pd.read_csv(os.path.join(cfg["data_dir"], "validation_dataset.csv"))

    train_dataset = Dataset.from_pandas(
        train_df.sample(cfg["train_size"], random_state=42).apply(format_data, axis=1, result_type="expand")
    )
    val_dataset = Dataset.from_pandas(
        val_df.sample((cfg["val_size"]), random_state=42).apply(format_data, axis=1, result_type="expand")
    )

    return train_dataset, val_dataset


def initialize_model(cfg):
    """Initialize and configure the model and tokenizer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["bnb"]["load_in_4bit"]
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=4096,
        dtype=None,  # auto-detect
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Add LoRA configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        modules_to_save=cfg["lora"]["modules_to_save"],
    )

    return model, tokenizer


def create_training_args(cfg):
    """Create training arguments based on configuration."""
    return TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["training"]["num_epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        optim=cfg["training"]["optim"],
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=cfg["training"]["weight_decay"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=cfg["training"]["max_grad_norm"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        evaluation_strategy=cfg["training"]["evaluation_strategy"],
        eval_steps=cfg["training"]["eval_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        save_steps=cfg["training"]["save_steps"],
        load_best_model_at_end=cfg["training"]["load_best_model_at_end"],
        group_by_length=cfg["training"]["group_by_length"],
    )


def train_model(config_path):
    """Main training function."""
    cfg = load_config(config_path)
    logger = setup_logging(cfg["experiment_name"])

    # Initialize model and tokenizer
    logger.info(f"Initializing model: {cfg['model_name']}")
    model, tokenizer = initialize_model(cfg)

    # Prepare datasets
    logger.info("Preparing datasets")
    train_dataset, val_dataset = prepare_datasets(cfg)

    # Setup training arguments
    training_args = create_training_args(cfg)

    # Initialize trainer
    logger.info("Setting up trainer")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        packing=False,
        max_seq_length=4096,
    )

    # Train model
    logger.info("Training started...")
    trainer.train()
    logger.info("Training completed.")

    # Save model
    model_save_path = cfg["output_dir"]
    logger.info(f"Saving model to {model_save_path}...")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Model saved successfully to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with Unsloth")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    train_model(args.config)
