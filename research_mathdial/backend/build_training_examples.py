#!/usr/bin/env python3
"""
Build Training Examples for Stage C
Merges raw dialogues, dependencies (Stage A), and teacher signals (Stage B)
into structured training examples for fine-tuning.
"""

import argparse
import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


def parse_conversation(conversation_str: str) -> List[Tuple[str, str]]:
    """
    Parse conversation string into list of (role, text) tuples.
    Format: "Teacher: text|EOM|Student: text|EOM|..."
    """
    turns = []
    parts = conversation_str.split("|EOM|")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Match role and text
        if ":" in part:
            # Extract role (before first colon)
            role, text = part.split(":", 1)
            role = role.strip()
            text = text.strip()

            # Normalize student names to "Student"
            if role != "Teacher":
                role = "Student"

            # Remove teacher annotations like "(generic)", "(telling)"
            if role == "Teacher" and text.startswith("(") and ")" in text[:20]:
                text = text.split(")", 1)[1].strip()

            turns.append((role, text))

    return turns


def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def index_by_qid_turn(records: List[dict]) -> Dict[Tuple[str, int], dict]:
    """Index records by (qid, turn_id) tuple."""
    index = {}
    for record in records:
        qid = str(record.get("qid"))
        turn_id = record.get("turn_id")
        if qid and turn_id is not None:
            key = (qid, turn_id)
            index[key] = record
    return index


def build_training_examples(
    raw_data: List[dict],
    dependencies: Dict[Tuple[str, int], dict],
    teacher_signals: Dict[Tuple[str, int], dict]
) -> List[dict]:
    """
    Build training examples by merging raw dialogues with dependencies and signals.
    """
    training_examples = []

    for dialogue in raw_data:
        qid = str(dialogue["qid"])
        conversation = dialogue["conversation"]
        question = dialogue["question"]
        ground_truth = dialogue["ground_truth"]
        student_solution = dialogue["student_incorrect_solution"]

        # Parse conversation into turns
        turns = parse_conversation(conversation)

        # Extract student turns with their indices
        student_turns = []
        for i, (role, text) in enumerate(turns):
            if role == "Student":
                student_turns.append((len(student_turns), i, text))

        # Build training example for each student turn
        for student_turn_id, absolute_idx, student_text in student_turns:
            key = (qid, student_turn_id)

            # Look up dependency and teacher signal
            dep = dependencies.get(key)
            signal = teacher_signals.get(key)

            # Skip if we don't have teacher signal (required for training)
            if not signal:
                continue

            # Build context: previous student turns and their operations
            previous_operations = []
            previous_confusions = []
            dependency_edges = []

            for prev_turn_id, _, prev_text in student_turns[:student_turn_id]:
                prev_key = (qid, prev_turn_id)
                prev_dep = dependencies.get(prev_key)
                if prev_dep:
                    previous_operations.append(prev_dep["operation"])

                # If this turn depends on previous turn, add to edges
                if dep and dep.get("depends_on") == prev_turn_id:
                    dependency_edges.append(prev_turn_id)

            # Build confusion summary from teacher signals of previous turns
            confusion_parts = []
            for prev_turn_id, _, _ in student_turns[:student_turn_id]:
                prev_key = (qid, prev_turn_id)
                prev_signal = teacher_signals.get(prev_key)
                if prev_signal and prev_signal.get("thought"):
                    confusion_parts.append(prev_signal["thought"])

            previous_confusion_summary = " ".join(confusion_parts[-2:]) if confusion_parts else "none"

            # Build the training example
            example = {
                "qid": qid,
                "turn_id": student_turn_id,
                "question": question,
                "ground_truth": ground_truth,
                "student_solution": student_solution,
                "input": student_text,
                "conversation_context": " ".join([f"{r}: {t}" for r, t in turns[:absolute_idx]]),
                "dependency_edges": dependency_edges if dependency_edges else [dep.get("depends_on")] if dep and dep.get("depends_on") is not None else [],
                "previous_operations": previous_operations,
                "previous_confusion_summary": previous_confusion_summary,
                "operation": dep["operation"] if dep else "other",
                "operation_confidence": dep["confidence"] if dep else 0.0,
                "output": {
                    "thought": signal.get("thought", ""),
                    "dependency_reason": signal.get("dependency_reason", ""),
                    "ref_turn": signal.get("ref_turn"),
                    "hint_type": signal.get("hint_type", "directive"),
                    "symbolic_comment": signal.get("symbolic_comment", ""),
                    "confidence": signal.get("confidence", 0.5)
                }
            }

            training_examples.append(example)

    return training_examples


def create_ablation_variants(examples: List[dict]) -> Dict[str, List[dict]]:
    """
    Create 4 ablation variants of the dataset:
    1. Baseline: student turn only → operation
    2. +CoT: + thought
    3. +Dependency: + dependency_reason + ref_turn
    4. Full: + symbolic_comment + confidence
    """
    variants = {
        "baseline": [],
        "cot": [],
        "dependency": [],
        "full": []
    }

    for example in examples:
        # Baseline: only operation
        baseline = {
            "qid": example["qid"],
            "turn_id": example["turn_id"],
            "input": example["input"],
            "output": {
                "operation": example["operation"]
            }
        }
        variants["baseline"].append(baseline)

        # +CoT: add thought
        cot = baseline.copy()
        cot["output"] = {
            "operation": example["operation"],
            "thought": example["output"]["thought"]
        }
        variants["cot"].append(cot)

        # +Dependency: add dependency reasoning
        dependency = cot.copy()
        dependency["conversation_context"] = example["conversation_context"]
        dependency["previous_operations"] = example["previous_operations"]
        dependency["output"] = {
            "operation": example["operation"],
            "thought": example["output"]["thought"],
            "dependency_reason": example["output"]["dependency_reason"],
            "ref_turn": example["output"]["ref_turn"]
        }
        variants["dependency"].append(dependency)

        # Full: add all pedagogical features
        full = example.copy()
        variants["full"].append(full)

    return variants


def main():
    parser = argparse.ArgumentParser(description="Build training examples for Stage C")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to raw dialogue data (train.jsonl)")
    parser.add_argument("--deps", type=str, required=True,
                        help="Path to dependencies from Stage A")
    parser.add_argument("--signals", type=str, required=True,
                        help="Path to teacher signals from Stage B")
    parser.add_argument("--out", type=str, required=True,
                        help="Output path for training examples")
    parser.add_argument("--ablation", action="store_true",
                        help="Create ablation variants (baseline, cot, dependency, full)")

    args = parser.parse_args()

    print(f"Loading raw data from {args.data}...")
    raw_data = load_jsonl(args.data)
    print(f"  Loaded {len(raw_data)} dialogues")

    print(f"Loading dependencies from {args.deps}...")
    deps_list = load_jsonl(args.deps)
    dependencies = index_by_qid_turn(deps_list)
    print(f"  Loaded {len(dependencies)} dependency records")

    print(f"Loading teacher signals from {args.signals}...")
    signals_list = load_jsonl(args.signals)
    teacher_signals = index_by_qid_turn(signals_list)
    print(f"  Loaded {len(teacher_signals)} teacher signal records")

    print("\nBuilding training examples...")
    training_examples = build_training_examples(raw_data, dependencies, teacher_signals)
    print(f"  Created {len(training_examples)} training examples")

    if args.ablation:
        print("\nCreating ablation variants...")
        variants = create_ablation_variants(training_examples)

        for variant_name, variant_examples in variants.items():
            variant_path = args.out.replace(".jsonl", f"_{variant_name}.jsonl")
            print(f"  Writing {len(variant_examples)} examples to {variant_path}")
            with open(variant_path, "w", encoding="utf-8") as f:
                for example in variant_examples:
                    f.write(json.dumps(example) + "\n")
    else:
        # Write full examples
        print(f"\nWriting to {args.out}...")
        with open(args.out, "w", encoding="utf-8") as f:
            for example in training_examples:
                f.write(json.dumps(example) + "\n")

    print("\n✅ Done!")

    # Print statistics
    print("\n📊 Statistics:")
    print(f"  Total training examples: {len(training_examples)}")

    # Count examples by hint type
    hint_types = defaultdict(int)
    for ex in training_examples:
        hint_types[ex["output"]["hint_type"]] += 1

    print("  Hint type distribution:")
    for hint_type, count in sorted(hint_types.items()):
        print(f"    {hint_type}: {count} ({count/len(training_examples)*100:.1f}%)")


if __name__ == "__main__":
    main()
