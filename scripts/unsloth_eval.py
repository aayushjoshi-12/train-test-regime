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
from tqdm import tqdm
from unsloth import FastLanguageModel


def setup_logging(experiment_name, log_dir="./experiments/logs"):
    """Configure logging for the evaluation process."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_name}_eval.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger(__name__), log_file


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_model(cfg, model_path):
    """Load and prepare the model for inference."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"], max_seq_length=4096, dtype=None, load_in_4bit=True
    )
    model.load_adapter(model_path)
    FastLanguageModel.for_inference(model)
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
    """Extract instruction and ground truth from formatted text."""
    instruction = row["instruction"]
    ground_truth = row["response"]
    prompt = format_data(row)["text"]

    return instruction, ground_truth, prompt


def generate_prediction(
    model, tokenizer, prompt, temperature=0.7, top_p=None, top_k=None
):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Set generation parameters
    generation_kwargs = {
        "max_new_tokens": 256,
        "temperature": temperature,
    }

    # Add either top_p or top_k, but not both
    if top_p is not None:
        generation_kwargs["top_p"] = top_p
    elif top_k is not None:
        generation_kwargs["top_k"] = top_k

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up the prediction - extract only the assistant's response
    pred_parts = pred.split("<|start_header_id|>assistant<|end_header_id|>")
    if len(pred_parts) > 1:
        prediction = pred_parts[1].split("<|eot_id|>")[0].strip()
    else:
        prediction = pred.strip()

    return prediction


def compute_metrics(prediction, ground_truth, bleu, rouge, sim_model):
    """Compute evaluation metrics."""
    bleu_score = bleu.compute(predictions=[prediction], references=[ground_truth])[
        "bleu"
    ]
    rouge_score = rouge.compute(predictions=[prediction], references=[ground_truth])[
        "rougeL"
    ]
    sim_score = util.cos_sim(
        sim_model.encode(prediction), sim_model.encode(ground_truth)
    )[0][0].item()

    return bleu_score, rouge_score, sim_score


def save_results(results, csv_path):
    """Save evaluation results to CSV."""
    with open(csv_path, "w", newline="") as f:
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
        "./experiments/logs", f"{cfg['experiment_name']}_results.csv"
    )

    # Load model
    logger.info(f"Loading model: {cfg['model_name']}")
    model_path = cfg["model_dir"]
    model, tokenizer = load_model(cfg, model_path)

    # Load dataset
    logger.info("Loading evaluation dataset")
    val_df = prepare_dataset("./data/validation_dataset.csv", cfg.get("test_size", 100))

    # Initialize metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    results = []
    logger.info("Starting evaluation")

    for row in tqdm(val_df.itterrows(), total=len(val_df)):
        # Extract components
        instruction, ground_truth, prompt = extract_components(row)

        # Generate prediction
        # Get generation parameters from config
        temperature = cfg["test"].get("temperature", 0.1)
        top_p = cfg["test"].get("top_p", None)
        top_k = cfg["test"].get("top_k", None)

        # Generate prediction with appropriate parameters
        prediction = generate_prediction(
            model, tokenizer, prompt, temperature=temperature, top_p=top_p, top_k=top_k
        )

        # Compute metrics
        bleu_score, rouge_score, sim_score = compute_metrics(
            prediction, ground_truth, bleu, rouge, sim_model
        )

        # Store results
        results.append(
            {
                "instruction": instruction,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "bleu": bleu_score,
                "rougeL": rouge_score,
                "semantic_similarity": sim_score,
            }
        )

    # Aggregate metrics
    avg_bleu = sum(r["bleu"] for r in results) / len(results)
    avg_rouge = sum(r["rougeL"] for r in results) / len(results)
    avg_sim = sum(r["semantic_similarity"] for r in results) / len(results)

    logger.info(
        f"BLEU: {avg_bleu:.4f} | ROUGE-L: {avg_rouge:.4f} | Semantic Similarity: {avg_sim:.4f}"
    )

    # Save results
    save_results(results, csv_file)
    logger.info(f"Saved evaluation results to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model with Unsloth")
    parser.add_argument(
        "--config",
        type=str,
        default="./experiments/config_1.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    evaluate_model(args.config)
