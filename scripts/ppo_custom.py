import gc

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    get_scheduler,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

# Memory optimization: Clear any existing CUDA cache
torch.cuda.empty_cache()
gc.collect()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./models/reward_models/llama3.2-rm",
    quantization_config=bnb_config,
    device_map="auto",
)

value_model = AutoModelForSequenceClassification.from_pretrained(
    "./models/reward_models/llama3.2-rm",
    quantization_config=bnb_config,
    device_map="auto",
)

policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    "../trained_models/llama3.1-mortgage-finetuned_v4",
    quantization_config=bnb_config,
    device_map="auto",
)
policy.generation_config = GenerationConfig(top_k=0, top_p=1.0)

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

sample_df = df.sample(500, random_state=42)
dataset = Dataset.from_pandas(
    sample_df.apply(format_data, axis=1, result_type="expand")
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors=None,
    )


dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

config = PPOConfig(
    learning_rate=1.41e-5,
    weight_decay=0.01,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=8,
    num_ppo_epochs=2,
    per_gpu_train_batch_size=1,
    kl_coef=0.2,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    cliprange=0.2,
    cliprange_value=0.2,
    gamma=1.0,
    lam=0.95,
    vf_coef=0.1,
    ds3_gather_for_generation=False,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

policy.pretrained_model.gradient_checkpointing_enable()

trainer = PPOTrainer(
    args=config,
    processing_class=tokenizer,
    model=policy,
    reward_model=reward_model,
    value_model=value_model,
    ref_model=None,
    train_dataset=dataset,
    peft_config=peft_config,
)

del df, sample_df
gc.collect()
torch.cuda.empty_cache()


# Set up optimizer
optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)

# Create dataset loader
train_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# Setup lr scheduler
lr_scheduler = get_scheduler(
    name=config.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_dataloader) * config.num_train_epochs,
)

# Training loop
for epoch in range(config.num_train_epochs):
    policy.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_dataloader):
        # Move batch to GPU
        batch = {k: v.to(policy.device_map) for k, v in batch.items()}
        
        # Generate responses
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        with torch.no_grad():
            # Generate policy output
            outputs = policy.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
            generated_ids = outputs.sequences
            
            # Get rewards from reward model
            reward_model.to(policy.device_map)
            reward_outputs = reward_model(generated_ids, attention_mask=torch.ones_like(generated_ids))
            rewards = reward_outputs.logits.squeeze(-1)
            reward_model.to("cpu")  # Offload reward model to CPU
            torch.cuda.empty_cache()
            
            # Get value estimates
            value_model.to(policy.device_map)
            value_outputs = value_model(input_ids, attention_mask=attention_mask)
            values = value_outputs.logits.squeeze(-1)
            value_model.to("cpu")  # Offload value model to CPU
            torch.cuda.empty_cache()
        
        # Forward pass with policy gradients enabled
        policy_outputs = policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=generated_ids,
        )
        policy_loss = policy_outputs.loss
        
        # Calculate PPO loss components
        advantages = rewards - values
        
        # Apply PPO clip
        ratio = torch.exp(policy_loss - policy_outputs.loss.detach())
        clip_adv = torch.clamp(ratio, 1 - config.cliprange, 1 + config.cliprange) * advantages
        ppo_loss = -torch.min(ratio * advantages, clip_adv).mean()
        
        # Value function loss
        value_loss = config.vf_coef * torch.mean((rewards - values) ** 2)
        
        # KL penalty (simplified)
        kl_loss = config.kl_coef * policy_loss
        
        # Total loss
        loss = ppo_loss + value_loss + kl_loss
        
        # Gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        loss.backward()
        
        # Optimize
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        # Clear memory
        del batch, input_ids, attention_mask, rewards, values, advantages
        torch.cuda.empty_cache()
        gc.collect()
        
        # Logging
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # End of epoch
    print(f"Epoch {epoch+1} completed. Avg loss: {total_loss/len(train_dataloader):.4f}")

# Save model at the end
policy.save_pretrained("./models/llama3.1-ppo-w-llama3.2-rm")
tokenizer.save_pretrained("./models/llama3.1-ppo-w-llama3.2-rm")
