import torch
from transformers import (
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import pandas as pd
from datasets import Dataset


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./models/reward_models/llama3.2-rm", quantization_config=bnb_config
)
value_model = AutoModelForSequenceClassification.from_pretrained(
    "./models/reward_models/llama3.2-rm", quantization_config=bnb_config
)
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    "../trained_models/llama3.1-mortgage-finetuned_v4", quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(
    "../trained_models/llama3.1-mortgage-finetuned_v4", quantization_config=bnb_config
)

tokenizer.pad_token = tokenizer.eos_token


def format_data(row):
    """Format a data row into the expected evaluation format."""
    text = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{row["system_prompt"]}<|eot_id|>

cateory: {row["category"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{row["instruction"]}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    return {"text": text}


df = pd.read_csv("./data/training_dataset.csv")

sample_df = df.sample(1000, random_state=42)
dataset = Dataset.from_pandas(
    sample_df.apply(format_data, axis=1, result_type="expand")
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors=None,
    )


dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

config = PPOConfig(
    learning_rate=1.41e-5,
    weight_decay=0.01,
    mini_batch_size=1,
    gradient_accumulation_steps=8,
    num_ppo_epochs=2,
    per_gpu_train_batch_size=8,
    kl_coef=0.2,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    cliprange=0.2,
    cliprange_value=0.2,
    gamma=1.0,
    lam=0.95,
    vf_coef=0.1,
)

trainer = PPOTrainer(
    args=config,
    model=policy,
    reward_model=reward_model,
    ref_model=None,
    processing_class=tokenizer,
    value_model=value_model,
    train_dataset=dataset,
)

trainer.train()

policy.save_pretrained("./models/llama3.1-ppo-w-llama3.2-rm")
tokenizer.save_pretrained("./models/llama3.1-ppo-w-llama3.2-rm")
