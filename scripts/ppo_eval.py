#!/usr/bin/env python3
import argparse
import csv
import logging
import os
from pathlib import Path

import evaluate
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLMWithValueHead, AutoTokenizer, GenerationConfig


def setup_logging(experiment_name, log_dir="./experiments/logs"):
    """Configure logging for the evaluation process."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}_ppo_eval.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)
    return logging.getLogger(__name__), log_file


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_model(model_path):
    """Load the PPO-trained policy model for inference."""
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.generation_config = GenerationConfig()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def format_data(row):
    """Format a data row into the expected evaluation format."""
    return {
        "text": f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{row["system_prompt"]}

cateory: {row["category"]}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{row["instruction"]}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    }


def prepare_dataset(data_path, test_size):
    """Load and prepare dataset for evaluation."""
    df = pd.read_csv(data_path)
    return df.sample(test_size, random_state=42)


def extract_components(row):
    """Extract instruction and ground truth from a dataset row."""
    instruction = row["instruction"]
    ground_truth = row["response"]
    prompt = format_data(row)["text"]

    return instruction, ground_truth, prompt


def generate_prediction(
    model, tokenizer, prompt, temperature=0.7, top_p=None, top_k=None, max_new_tokens=256
):
    """Generate model prediction for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Set generation parameters
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": temperature > 0,
    }

    # Add either top_p or top_k if specified
    if top_p is not None:
        generation_kwargs["top_p"] = top_p
    if top_k is not None:
        generation_kwargs["top_k"] = top_k

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    
    pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Clean up the prediction to extract only the assistant's response
    if "<|eot_id|>" in pred:
        pred = pred.split("<|eot_id|>")[0].strip()
    
    return pred


def compute_metrics(prediction, ground_truth, bleu, rouge, sim_model):
    """Compute evaluation metrics."""
    bleu_score = bleu.compute(predictions=[prediction], references=[ground_truth])["bleu"]
    rouge_score = rouge.compute(predictions=[prediction], references=[ground_truth])["rougeL"]
    
    # Compute semantic similarity
    try:
        pred_embedding = sim_model.encode(prediction)
        truth_embedding = sim_model.encode(ground_truth)
        sim_score = util.cos_sim(pred_embedding, truth_embedding)[0][0].item()
    except Exception as e:
        logging.error(f"Error computing semantic similarity: {e}")
        sim_score = 0.0
    
    return bleu_score, rouge_score, sim_score


def save_results(results, csv_path):
    """Save evaluation results to CSV."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "instruction",
            "ground_truth",
            "prediction",
            "bleu",
            "rougeL",
            "semantic_similarity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def evaluate_model(config_path):
    """Main evaluation function."""
    # Load configuration
    cfg = load_config(config_path)
    logger, log_file = setup_logging(cfg["experiment_name"])
    csv_file = os.path.join(
        "./experiments/logs", f"{cfg['experiment_name']}_ppo_results.csv"
    )

    # Load model
    logger.info(f"Loading PPO-trained model from: {cfg['output_dir']}")
    model_path = cfg["output_dir"]
    model, tokenizer = load_model(model_path)

    # Load dataset
    test_size = cfg.get("test_size", 100)
    data_path = cfg.get("eval_data_path", "./data/validation_dataset.csv")
    logger.info(f"Loading evaluation dataset from {data_path} with {test_size} samples")
    val_df = prepare_dataset(data_path, test_size)

    # Initialize metrics
    logger.info("Initializing evaluation metrics")
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    results = []
    logger.info("Starting evaluation")

    for idx, row in enumerate(val_df.iterrows()):
        _, row_data = row
        
        # Extract components
        instruction, ground_truth, prompt = extract_components(row_data)
        
        # Log progress
        if idx % 10 == 0:
            logger.info(f"Processing sample {idx}/{len(val_df)}")

        # Get generation parameters from config
        gen_config = cfg.get("test", {})
        temperature = gen_config.get("temperature", 0.1)
        top_p = gen_config.get("top_p", None)
        top_k = gen_config.get("top_k", None)
        max_new_tokens = gen_config.get("max_new_tokens", 512)

        # Generate prediction
        try:
            prediction = generate_prediction(
                model, tokenizer, prompt, 
                temperature=temperature, 
                top_p=top_p, 
                top_k=top_k,
                max_new_tokens=max_new_tokens
            )
            
            # Compute metrics
            bleu_score, rouge_score, sim_score = compute_metrics(
                prediction, ground_truth, bleu, rouge, sim_model
            )
            
            # Store results
            results.append({
                "instruction": instruction,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "bleu": bleu_score,
                "rougeL": rouge_score,
                "semantic_similarity": sim_score,
            })
            
            # Log individual results
            if idx % 10 == 0:
                logger.info(f"Sample {idx} metrics - BLEU: {bleu_score:.4f}, ROUGE-L: {rouge_score:.4f}, Similarity: {sim_score:.4f}")
                
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")

    # Aggregate metrics
    if results:
        avg_bleu = sum(r["bleu"] for r in results) / len(results)
        avg_rouge = sum(r["rougeL"] for r in results) / len(results)
        avg_sim = sum(r["semantic_similarity"] for r in results) / len(results)

        logger.info("===== EVALUATION RESULTS =====")
        logger.info(f"Total samples evaluated: {len(results)}")
        logger.info(f"Average BLEU: {avg_bleu:.4f}")
        logger.info(f"Average ROUGE-L: {avg_rouge:.4f}")
        logger.info(f"Average Semantic Similarity: {avg_sim:.4f}")
    else:
        logger.error("No results were computed. Check for errors.")

    # Save results
    if results:
        save_results(results, csv_file)
        logger.info(f"Saved evaluation results to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO-trained policy model")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    evaluate_model(args.config)