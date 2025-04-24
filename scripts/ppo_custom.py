import argparse
import gc
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
)


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


def set_seed(seed_val=42):
    """Set all seeds for reproducibility"""
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    os.environ["PYTHONHASHSEED"] = str(seed_val)
    torch.backends.cudnn.deterministic = True


def prepare_dataset(cfg):
    df = pd.read_csv(os.path.join(cfg["data_dir"], "training_dataset.csv"))
    sample_df = df.sample(cfg["train_size"], random_state=42)
    
    def format_data(row):
        return {
            "query": f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{row["system_prompt"]}<|eot_id|>

cateory: {row["category"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{row["instruction"]}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
        }
    
    dataset = Dataset.from_pandas(sample_df.apply(format_data, axis=1, result_type="expand"))
    return dataset


def initialize_models(cfg):
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["policy_model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load reward model
    reward_model = AutoModelForCausalLM.from_pretrained(
        cfg["reward_model_path"],
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Load policy model
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg["policy_model_path"],
        device_map="auto",
        torch_dtype=torch.float16,
    )
    policy_model.generation_config = GenerationConfig()
    
    # Create reference model
    ref_model = create_reference_model(policy_model)
    
    return policy_model, ref_model, reward_model, tokenizer


def reward_fn(query, response, tokenizer, reward_model):
    # Concatenate query and response
    texts = [q + r for q, r in zip(query, response)]

    # Tokenize the input
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(reward_model.device)

    # Get logits from the reward model
    with torch.no_grad():
        outputs = reward_model(**inputs)
        logits = outputs.logits

    # Calculate rewards
    rewards = logits.mean(dim=1)
    return rewards


def train_model(config_path):
    cfg = load_config(config_path)
    logger = setup_logging(cfg["experiment_name"])
    set_seed(cfg.get("seed", 42))
    
    logger.info("Loading models and tokenizer")
    policy_model, ref_model, reward_model, tokenizer = initialize_models(cfg)
    
    logger.info("Preparing dataset")
    dataset = prepare_dataset(cfg)
    
    # PPO configuration from the config file
    ppo_config = PPOConfig(
        learning_rate=cfg["training"]["learning_rate"],
        batch_size=cfg["training"].get("batch_size", 4),
        mini_batch_size=cfg["training"].get("mini_batch_size", 1),
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["num_epochs"],
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
    )
    
    logger.info("Initializing PPO trainer")
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=lambda data: {k: [d[k] for d in data] for k in data[0]},
    )
    
    logger.info("Starting PPO training...")
    
    # Training loop
    for epoch in range(cfg["training"]["num_epochs"]):
        logger.info(f"Starting epoch {epoch+1}/{cfg['training']['num_epochs']}")
        for batch in tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch+1}"):
            # Get query from batch
            query_tensors = [
                tokenizer(q, return_tensors="pt").input_ids.squeeze()
                for q in batch["query"]
            ]

            # Generate responses
            response_tensors = []
            for query in query_tensors:
                response = ppo_trainer.generate(
                    query.unsqueeze(0),
                    max_new_tokens=512,
                    do_sample=True,
                    # top_k=0,
                    # top_p=0.9,
                )
                response_tensors.append(response.squeeze())

            # Decode responses
            batch_responses = [
                tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors
            ]
            batch_queries = [
                tokenizer.decode(q, skip_special_tokens=True) for q in query_tensors
            ]

            # Compute rewards
            rewards = reward_fn(batch_queries, batch_responses, tokenizer, reward_model)

            # Update policy with PPO
            ppo_trainer.step(query_tensors, response_tensors, rewards)
    
    logger.info("Training complete. Saving model...")
    
    # Save the fine-tuned model
    policy_model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    
    logger.info(f"Model saved to {cfg['output_dir']}")
    
    # Free up memory
    gc.collect()
    torch.cuda.empty_cache()
    
    return cfg["output_dir"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO model with config YAML")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    output_dir = train_model(args.config)
    
    # Test the final model
    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = AutoModelForCausalLM.from_pretrained(output_dir)
    
    # Create a text generation pipeline
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=True,
        top_p=0.9,
    )
    
    # Test with example prompt if provided in config
    if "test_prompt" in cfg:
        results = gen_pipeline(cfg["test_prompt"], num_return_sequences=3)
        
        # Print results
        for i, result in enumerate(results):
            print(f"Generation {i + 1}:")
            print(result["generated_text"])
            print("-" * 50)
