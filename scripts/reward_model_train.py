import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

model = AutoModelForSequenceClassification.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


def format_data(example):
    kwargs = {
        "padding": "max_length",
        "truncation": True,
        "max_length": 512,
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


dataset = load_dataset(
    "csv",
    data_files="./data/llama3-loan-mortgage-ranked-responses.csv",
).map(format_data)

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1, bias="none", task_type=TaskType.SEQ_CLS
)

reward_config = RewardConfig(
    output_dir="./models/reward_model_qwen2.5-3b",
    per_device_train_batch_size=21,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=2,
)

trainer = RewardTrainer(
    model=model,
    args=reward_config,
    peft_config=lora_config,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
)

trainer.train()
model.save_pretrained("./models/reward_model_gpt2")
tokenizer.save_pretrained("./models/reward_model_gpt2")
