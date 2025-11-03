#!/usr/bin/env python3
"""
Evaluation Script for Stage C
Evaluates trained models on validation set with multiple metrics.
"""

import argparse
import json
import re
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_examples(path: str) -> List[dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_model(model_path: str, base_model: str = None):
    """
    Load a fine-tuned model (LoRA or full).
    """
    logger.info(f"Loading model from {model_path}")

    # Try to load as PEFT model first
    try:
        if base_model is None:
            # Try to infer base model from adapter_config.json
            import os
            config_path = os.path.join(model_path, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                    base_model = config.get("base_model_name_or_path")

        if base_model:
            logger.info(f"Loading LoRA adapter with base model: {base_model}")
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base, model_path)
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        else:
            raise ValueError("Could not determine base model for LoRA adapter")
    except:
        # Load as full model
        logger.info("Loading as full model (not LoRA)")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def format_prompt(example: dict) -> str:
    """Format the input prompt."""
    prompt_parts = []

    prompt_parts.append(f"### Student Turn:\n{example['input']}")

    if "conversation_context" in example and example.get("conversation_context"):
        prompt_parts.append(f"\n### Context:\n{example['conversation_context']}")

    if "previous_operations" in example and example.get("previous_operations"):
        ops = ", ".join(example["previous_operations"])
        prompt_parts.append(f"\n### Previous Operations:\n{ops}")

    prompt_parts.append("\n\n### Response:\n")

    return "\n".join(prompt_parts)


def generate_prediction(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate prediction from model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        try:
            # Use greedy decoding for stability
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding instead of sampling
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
                repetition_penalty=1.0
            )
        except RuntimeError as e:
            logger.error(f"Generation failed: {e}")
            return "ERROR: Generation failed due to numerical instability"

    # Decode only the generated part (skip the prompt)
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def parse_prediction(text: str) -> Dict[str, any]:
    """Parse model prediction into structured fields."""
    parsed = {}

    # Extract fields using regex
    patterns = {
        "operation": r"Operation:\s*(.+?)(?:\n|$)",
        "thought": r"Thought:\s*(.+?)(?:\n|$)",
        "dependency_reason": r"Dependency Reason:\s*(.+?)(?:\n|$)",
        "ref_turn": r"Reference Turn:\s*(\d+|null|None)(?:\n|$)",
        "hint_type": r"Hint Type:\s*(.+?)(?:\n|$)",
        "symbolic_comment": r"Symbolic Comment:\s*(.+?)(?:\n|$)",
        "confidence": r"Confidence:\s*([\d\.]+)(?:\n|$)"
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if field == "ref_turn":
                try:
                    parsed[field] = int(value) if value.lower() not in ["null", "none"] else None
                except:
                    parsed[field] = None
            elif field == "confidence":
                try:
                    parsed[field] = float(value)
                except:
                    parsed[field] = 0.5
            else:
                parsed[field] = value

    return parsed


def evaluate_predictions(examples: List[dict], predictions: List[Dict]) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    """
    metrics = {}

    # Hint Type F1
    if all("hint_type" in p and "hint_type" in ex["output"] for p, ex in zip(predictions, examples)):
        y_true = [ex["output"]["hint_type"] for ex in examples]
        y_pred = [p.get("hint_type", "directive") for p in predictions]

        # Map to common labels
        label_map = {"directive": 0, "procedural": 1, "conceptual": 2}
        y_true_encoded = [label_map.get(y, 0) for y in y_true]
        y_pred_encoded = [label_map.get(y, 0) for y in y_pred]

        hint_f1 = f1_score(y_true_encoded, y_pred_encoded, average="weighted")
        hint_acc = accuracy_score(y_true_encoded, y_pred_encoded)
        metrics["hint_type_f1"] = hint_f1
        metrics["hint_type_accuracy"] = hint_acc

    # Ref Turn Accuracy
    if all("ref_turn" in p and "ref_turn" in ex["output"] for p, ex in zip(predictions, examples)):
        correct = sum(
            1 for p, ex in zip(predictions, examples)
            if p.get("ref_turn") == ex["output"].get("ref_turn")
        )
        metrics["ref_turn_accuracy"] = correct / len(examples)

    # Confidence Calibration MAE
    if all("confidence" in p and "confidence" in ex["output"] for p, ex in zip(predictions, examples)):
        y_true = [ex["output"]["confidence"] for ex in examples]
        y_pred = [p.get("confidence", 0.5) for p in predictions]
        mae = np.mean([abs(t - p) for t, p in zip(y_true, y_pred)])
        metrics["confidence_mae"] = mae

    # Symbolic Comment Coverage (% of predictions that include it)
    if all("symbolic_comment" in ex["output"] for ex in examples):
        coverage = sum(
            1 for p in predictions
            if "symbolic_comment" in p and p["symbolic_comment"]
        ) / len(predictions)
        metrics["symbolic_comment_coverage"] = coverage

    # Operation Accuracy (if available)
    if all("operation" in p for p in predictions):
        correct = sum(
            1 for p, ex in zip(predictions, examples)
            if p.get("operation") == ex.get("operation")
        )
        metrics["operation_accuracy"] = correct / len(examples)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name (for LoRA adapters)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to validation examples")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for evaluation results (JSON)")
    parser.add_argument("--save_predictions", type=str, default=None,
                        help="Optional path to save predictions (JSONL)")

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model, args.base_model)

    # Load validation data
    logger.info(f"Loading validation data from {args.data}...")
    examples = load_examples(args.data)
    logger.info(f"  Loaded {len(examples)} examples")

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = []
    error_count = 0

    for i, example in enumerate(examples):
        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i+1}/{len(examples)}")

        try:
            prompt = format_prompt(example)
            generated = generate_prediction(model, tokenizer, prompt)

            if "ERROR:" in generated:
                error_count += 1
                if error_count > 10:
                    logger.error(f"Too many errors ({error_count}), stopping evaluation")
                    break
                continue

            parsed = parse_prediction(generated)

            predictions.append({
                "qid": example["qid"],
                "turn_id": example["turn_id"],
                "input": example["input"],
                "generated_text": generated,
                "parsed": parsed,
                "ground_truth": example.get("output", {})
            })
        except Exception as e:
            logger.error(f"Error on example {i}: {e}")
            error_count += 1
            if error_count > 10:
                logger.error(f"Too many errors ({error_count}), stopping evaluation")
                break
            continue

    # Save predictions if requested
    if args.save_predictions:
        logger.info(f"Saving predictions to {args.save_predictions}...")
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

    # Calculate metrics
    logger.info("Calculating metrics...")
    parsed_predictions = [p["parsed"] for p in predictions]
    metrics = evaluate_predictions(examples, parsed_predictions)

    # Save metrics
    logger.info(f"Saving metrics to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Print metrics
    logger.info("\n📊 Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    logger.info("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
