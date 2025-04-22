import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    AutoModelForCausalLM
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler
from peft import PeftModel
from unsloth import FastLanguageModel


def setup_logging(experiment_name, log_dir="./experiments/logs"):
    """Configure logging for the PPO training process."""
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


def prepare_dataset(cfg):
    """Load and prepare dataset for PPO training."""
    train_df = pd.read_csv(os.path.join(cfg["data_dir"], "training_dataset.csv"))
    
    # Sample dataset if specified in config
    if cfg.get("train_size"):
        train_df = train_df.sample(cfg["train_size"], random_state=42)
        
    # Create a dataset with just the prompts (instructions)
    dataset = Dataset.from_pandas(train_df[["instruction", "system_prompt", "category"]])
    
    return dataset


def format_prompt(example, cfg):
    """Format the instruction into the expected prompt format for the model."""
    system_prompt = example["system_prompt"] if "system_prompt" in example else cfg["default_system_prompt"]
    category = example["category"] if "category" in example else ""
    
    # Format according to Llama-3 chat template but adapted for PPO
    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}

category: {category}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{example["instruction"]}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""
    return formatted_prompt


def initialize_sft_model(cfg):
    """Initialize and configure the SFT model."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["bnb"]["load_in_4bit"],
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Initialize the model from the fine-tuned SFT checkpoint
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["sft_model_path"],
        max_seq_length=cfg["max_seq_length"],
        dtype=None,  # auto-detect
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Add value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model,
        device_map="auto"
    )
    
    # Create a reference model for KL penalty
    ref_model = create_reference_model(model)
    
    return model, ref_model, tokenizer


def initialize_reward_model(cfg):
    """Initialize the reward model from the fine-tuned checkpoint."""
    # Load the fine-tuned reward model
    reward_model = AutoModelForCausalLM.from_pretrained(
        cfg["reward_model_path"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    reward_tokenizer = AutoTokenizer.from_pretrained(cfg["reward_model_path"])
    
    if reward_tokenizer.pad_token_id is None:
        reward_tokenizer.pad_token_id = reward_tokenizer.eos_token_id
    
    # Create a reward pipeline
    reward_pipe = pipeline(
        "text-classification",
        model=reward_model,
        tokenizer=reward_tokenizer,
        device_map="auto",
    )
    
    return reward_pipe


def create_ppo_config(cfg):
    """Create PPO training configuration."""
    return PPOConfig(
        learning_rate=float(cfg["ppo"]["learning_rate"]),
        batch_size=cfg["ppo"]["batch_size"],
        mini_batch_size=cfg["ppo"]["mini_batch_size"],
        gradient_accumulation_steps=cfg["ppo"]["gradient_accumulation_steps"],
        optimize_cuda_cache=True,
        early_stopping=cfg["ppo"].get("early_stopping", True),
        target_kl=cfg["ppo"].get("target_kl", 0.1),
        ppo_epochs=cfg["ppo"].get("ppo_epochs", 4),
        seed=cfg["seed"],
        init_kl_coef=cfg["ppo"].get("init_kl_coef", 0.2),
        adap_kl_ctrl=cfg["ppo"].get("adap_kl_ctrl", True),
        use_score_scaling=cfg["ppo"].get("use_score_scaling", True),
        use_score_norm=cfg["ppo"].get("use_score_norm", True),
        cliprange=cfg["ppo"].get("cliprange", 0.2),
        cliprange_value=cfg["ppo"].get("cliprange_value", 0.2),
        vf_coef=cfg["ppo"].get("vf_coef", 0.1),
        horizon=cfg["ppo"].get("horizon", 10000),
        output_dir=cfg["output_dir"],
    )


def reward_fn(samples, reward_pipe, tokenizer):
    """Compute rewards for generated responses using the reward model."""
    rewards = []
    
    for sample in samples:
        # Get the score from the reward model
        result = reward_pipe(sample)
        if isinstance(result, list):
            # If result is a list (pipeline returns a list for each input)
            score = result[0]["score"]
        else:
            # Otherwise try to get the score directly
            score = result["score"]
        
        rewards.append(float(score))
    
    return rewards


def train_model(config_path):
    """Main PPO training function."""
    cfg = load_config(config_path)
    logger = setup_logging(cfg["experiment_name"])
    
    # Set random seed for reproducibility
    torch.manual_seed(cfg["seed"])
    
    # Initialize models
    logger.info("Initializing SFT model and reward model")
    model, ref_model, tokenizer = initialize_sft_model(cfg)
    reward_pipe = initialize_reward_model(cfg)
    
    # Prepare dataset
    logger.info("Preparing dataset")
    dataset = prepare_dataset(cfg)
    
    # Create PPO config
    ppo_config = create_ppo_config(cfg)
    
    # Initialize PPO trainer
    logger.info("Setting up PPO trainer")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=None,
    )
    
    # Set up response length sampler
    response_length_sampler = LengthSampler(
        cfg["generation"]["min_length"],
        cfg["generation"]["max_length"]
    )
    
    # Training loop
    logger.info("Starting PPO training...")
    
    for epoch in range(cfg["ppo"]["num_epochs"]):
        logger.info(f"Starting epoch {epoch+1}/{cfg['ppo']['num_epochs']}")
        
        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            # Format prompts
            query_tensors = []
            for example in batch:
                prompt = format_prompt(example, cfg)
                query_tensor = tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=cfg["max_prompt_length"]
                ).input_ids.to(model.device)
                query_tensors.append(query_tensor)
            
            # Generate responses
            generation_kwargs = {
                "max_new_tokens": response_length_sampler(),
                "do_sample": True,
                "temperature": cfg["generation"]["temperature"],
                "top_p": cfg["generation"]["top_p"],
                "top_k": cfg["generation"]["top_k"],
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            # Generate responses from the policy model
            response_tensors = []
            for query_tensor in query_tensors:
                response = ppo_trainer.generate(
                    query_tensor, 
                    **generation_kwargs
                )
                response_tensors.append(response.squeeze())
            
            # Decode responses
            batch_responses = []
            for response_tensor in response_tensors:
                decoded_response = tokenizer.decode(
                    response_tensor, 
                    skip_special_tokens=True
                )
                batch_responses.append(decoded_response)
            
            # Compute rewards
            rewards = reward_fn(batch_responses, reward_pipe, tokenizer)
            
            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)
            
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: Mean reward: {torch.mean(torch.tensor(rewards)):.4f}")
            
            # Save checkpoint periodically
            if (batch_idx + 1) % cfg["save_steps"] == 0:
                checkpoint_path = os.path.join(
                    cfg["output_dir"], 
                    f"checkpoint-epoch-{epoch+1}-batch-{batch_idx+1}"
                )
                ppo_trainer.save_pretrained(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Final save
    logger.info("Training completed.")
    ppo_trainer.save_pretrained(cfg["output_dir"])
    logger.info(f"Model saved successfully to {cfg['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with PPO using Unsloth and reward model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    train_model(args.config)