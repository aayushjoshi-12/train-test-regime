import argparse
import gc
import logging
import os
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def setup_logging(experiment_name, log_dir="./experiments/logs"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"{experiment_name}.log"),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def format_data(row):
    return {
        "text": f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{row["system_prompt"]}<|eot_id|>

cateory: {row["category"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{row["instruction"]}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    }


def prepare_dataset(cfg):
    df = pd.read_csv(os.path.join(cfg["data_dir"], "training_dataset.csv"))
    sample_df = df.sample(cfg["train_size"], random_state=42)
    dataset = Dataset.from_pandas(sample_df.apply(format_data, axis=1, result_type="expand"))
    return dataset


def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors=None,
        )
    return dataset.map(tokenize_function, batched=True, remove_columns=["text"])


def initialize_models(cfg):

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        cfg["reward_model_path"],
        device_map="auto",
    )

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg["policy_model_path"],
        device_map="auto",
    )
    policy.generation_config = GenerationConfig()

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg["policy_model_path"],
        device_map="auto",
    )
    ref_model.generation_config = GenerationConfig()

    tokenizer = AutoTokenizer.from_pretrained(cfg["policy_model_path"])
    tokenizer.pad_token = tokenizer.eos_token

    return policy, tokenizer, reward_model, ref_model


def train_model(config_path):
    cfg = load_config(config_path)
    logger = setup_logging(cfg["experiment_name"])
    logger.info("Loading models and tokenizer")

    policy, tokenizer, reward_model, ref_model = initialize_models(cfg)

    logger.info("Preparing dataset")
    raw_dataset = prepare_dataset(cfg)
    dataset = tokenize_dataset(raw_dataset, tokenizer)

    logger.info("Setting up PPO config and PEFT config")
    ppo_config = PPOConfig(
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        batch_size=1,
        mini_batch_size=1,
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_ppo_epochs=2,
        per_gpu_train_batch_size=1,
        kl_coef=0.2,
        num_train_epochs=cfg["training"]["num_epochs"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        cliprange=0.2,
        cliprange_value=0.2,
        gamma=1.0,
        lam=0.95,
        vf_coef=0.1,
    )

    policy.pretrained_model.gradient_checkpointing_enable()

    logger.info("Training with PPOTrainer...")
    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=policy,
        reward_model=reward_model,
        ref_model=ref_model,
        train_dataset=dataset,
    )

    gc.collect()
    torch.cuda.empty_cache()

    trainer.train()
    logger.info("Training complete. Saving model...")

    policy.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    logger.info(f"Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO fine-tuning model with config YAML")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    train_model(args.config)
