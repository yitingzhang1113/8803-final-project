#!/usr/bin/env python3
"""
Fine-tuning Script for Stage C
Trains models with LoRA on the 4 ablation conditions.
"""

import argparse
import json
import os
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathDialDataset(Dataset):
    """Dataset for MathDial tutor training."""

    def __init__(self, examples: List[dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Format prompt based on what fields are available in the ablation condition
        prompt = self._format_prompt(example)
        target = self._format_target(example)

        # Combine prompt and target for causal LM training
        full_text = f"{prompt}\n\n### Response:\n{target}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }

    def _format_prompt(self, example: dict) -> str:
        """Format the input prompt based on available fields."""
        prompt_parts = []

        # Always include the student input
        prompt_parts.append(f"### Student Turn:\n{example['input']}")

        # Add conversation context if available (for dependency and full conditions)
        if "conversation_context" in example and example.get("conversation_context"):
            prompt_parts.append(f"\n### Context:\n{example['conversation_context']}")

        # Add previous operations if available
        if "previous_operations" in example and example.get("previous_operations"):
            ops = ", ".join(example["previous_operations"])
            prompt_parts.append(f"\n### Previous Operations:\n{ops}")

        return "\n".join(prompt_parts)

    def _format_target(self, example: dict) -> str:
        """Format the target output based on ablation condition."""
        output = example.get("output", {})
        target_parts = []

        # Operation (baseline)
        if "operation" in output:
            target_parts.append(f"Operation: {output['operation']}")

        # Thought (CoT)
        if "thought" in output and output.get("thought"):
            target_parts.append(f"Thought: {output['thought']}")

        # Dependency fields (+Dependency)
        if "dependency_reason" in output and output.get("dependency_reason"):
            target_parts.append(f"Dependency Reason: {output['dependency_reason']}")

        if "ref_turn" in output and output.get("ref_turn") is not None:
            target_parts.append(f"Reference Turn: {output['ref_turn']}")

        # Full pedagogy fields (Full)
        if "hint_type" in output and output.get("hint_type"):
            target_parts.append(f"Hint Type: {output['hint_type']}")

        if "symbolic_comment" in output and output.get("symbolic_comment"):
            target_parts.append(f"Symbolic Comment: {output['symbolic_comment']}")

        if "confidence" in output and output.get("confidence") is not None:
            target_parts.append(f"Confidence: {output['confidence']:.2f}")

        return "\n".join(target_parts)


def load_examples(path: str) -> List[dict]:
    """Load training examples from JSONL file."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def setup_model_and_tokenizer(model_name: str, use_lora: bool = True):
    """
    Load model and tokenizer, optionally with LoRA.
    """
    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    if use_lora:
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Common for most LLMs
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train MathDial tutor models")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to training examples JSONL file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Model name or path (e.g., Qwen/Qwen2.5-1.5B-Instruct, meta-llama/Llama-3.2-1B)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--no_lora", action="store_true",
                        help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--eval_split", type=float, default=0.1,
                        help="Fraction of data to use for evaluation")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    logger.info(f"Loading training data from {args.dataset}...")
    examples = load_examples(args.dataset)
    logger.info(f"  Loaded {len(examples)} examples")

    # Split into train/eval
    split_idx = int(len(examples) * (1 - args.eval_split))
    train_examples = examples[:split_idx]
    eval_examples = examples[split_idx:]
    logger.info(f"  Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model, use_lora=not args.no_lora)

    # Create datasets
    train_dataset = MathDialDataset(train_examples, tokenizer, args.max_length)
    eval_dataset = MathDialDataset(eval_examples, tokenizer, args.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{args.output_dir}/logs",
        report_to="none",
        save_total_limit=2,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving final model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()
