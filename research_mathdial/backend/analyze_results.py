#!/usr/bin/env python3
"""
Analyze and visualize ablation study results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_training_logs(log_dir="logs"):
    """Parse training logs to extract metrics."""
    results = {
        "baseline": {},
        "cot": {},
        "dependency": {},
        "full": {}
    }

    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"Warning: {log_dir} not found")
        return results

    for model_name in results.keys():
        # Find the latest log file for this model
        log_files = list(log_path.glob(f"{model_name}_*.out"))
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)

            with open(latest_log) as f:
                content = f.read()

                # Extract training loss
                if "train_loss" in content:
                    for line in content.split('\n'):
                        if 'train_loss' in line:
                            try:
                                # Parse the dictionary string
                                import re
                                loss_match = re.search(r"'train_loss':\s*([\d.]+)", line)
                                if loss_match:
                                    results[model_name]['train_loss'] = float(loss_match.group(1))

                                runtime_match = re.search(r"'train_runtime':\s*([\d.]+)", line)
                                if runtime_match:
                                    results[model_name]['train_runtime'] = float(runtime_match.group(1))

                                epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)
                                if epoch_match:
                                    results[model_name]['epochs'] = float(epoch_match.group(1))
                            except:
                                pass

    return results


def load_evaluation_results(eval_dir="research_mathdial/results/evaluation"):
    """Load evaluation metrics from JSON files."""
    results = {}
    eval_path = Path(eval_dir)

    if not eval_path.exists():
        print(f"Warning: {eval_dir} not found. Run evaluation first.")
        return results

    for model_name in ["baseline", "cot", "dependency", "full"]:
        eval_file = eval_path / f"eval_{model_name}.json"
        if eval_file.exists():
            with open(eval_file) as f:
                results[model_name] = json.load(f)

    return results


def plot_training_loss(training_results, output_dir="research_mathdial/results/charts"):
    """Plot training loss comparison."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    models = []
    losses = []

    for model, metrics in training_results.items():
        if 'train_loss' in metrics:
            models.append(model.replace('_', ' ').title())
            losses.append(metrics['train_loss'])

    if not models:
        print("No training loss data found")
        return

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, losses, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Training Loss Comparison Across Ablation Conditions',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(losses) * 1.15)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_loss_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/training_loss_comparison.png")
    plt.close()


def plot_improvement_over_baseline(training_results, output_dir="research_mathdial/results/charts"):
    """Plot improvement over baseline."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if 'baseline' not in training_results or 'train_loss' not in training_results['baseline']:
        print("Baseline results not found")
        return

    baseline_loss = training_results['baseline']['train_loss']

    models = []
    improvements = []

    for model, metrics in training_results.items():
        if model != 'baseline' and 'train_loss' in metrics:
            improvement = (baseline_loss - metrics['train_loss']) / baseline_loss * 100
            models.append(model.replace('_', ' ').title())
            improvements.append(improvement)

    if not models:
        print("No improvement data to plot")
        return

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
    bars = ax.bar(models, improvements, color=colors, alpha=0.8)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Relative Improvement in Training Loss vs Baseline',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/improvement_over_baseline.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/improvement_over_baseline.png")
    plt.close()


def plot_training_time(training_results, output_dir="research_mathdial/results/charts"):
    """Plot training time comparison."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    models = []
    times = []

    for model, metrics in training_results.items():
        if 'train_runtime' in metrics:
            models.append(model.replace('_', ' ').title())
            times.append(metrics['train_runtime'] / 60)  # Convert to minutes

    if not models:
        print("No training time data found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, times, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}m',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Training Time (minutes)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Training Time Comparison (H100 GPU)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(times) * 1.15)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_time_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/training_time_comparison.png")
    plt.close()


def create_summary_table(training_results, eval_results=None, output_dir="research_mathdial/results"):
    """Create a summary table of all results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    data = []

    baseline_loss = training_results.get('baseline', {}).get('train_loss', 0)

    for model_name in ['baseline', 'cot', 'dependency', 'full']:
        if model_name in training_results and 'train_loss' in training_results[model_name]:
            row = {
                'Model': model_name.replace('_', ' ').title(),
                'Training Loss': training_results[model_name].get('train_loss', 'N/A'),
                'Training Time (min)': f"{training_results[model_name].get('train_runtime', 0) / 60:.1f}",
                'Improvement vs Baseline (%)': 'Baseline' if model_name == 'baseline' else
                    f"{(baseline_loss - training_results[model_name]['train_loss']) / baseline_loss * 100:.1f}%"
            }

            # Add evaluation metrics if available
            if eval_results and model_name in eval_results:
                metrics = eval_results[model_name]
                row['Hint Type F1'] = f"{metrics.get('hint_type_f1', 'N/A'):.3f}" if isinstance(metrics.get('hint_type_f1'), float) else 'N/A'
                row['Ref Turn Accuracy'] = f"{metrics.get('ref_turn_accuracy', 'N/A'):.3f}" if isinstance(metrics.get('ref_turn_accuracy'), float) else 'N/A'

            data.append(row)

    df = pd.DataFrame(data)

    # Save as CSV
    csv_path = f"{output_dir}/ablation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")

    # Print to console
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

    return df


def main():
    print("🔍 Analyzing Stage C Results")
    print("=" * 60)

    # Load training results
    print("\n📊 Loading training logs...")
    training_results = load_training_logs()

    if not any(training_results.values()):
        print("❌ No training results found. Make sure logs/ directory exists.")
        return

    # Load evaluation results (if available)
    print("\n📊 Loading evaluation results...")
    eval_results = load_evaluation_results()

    # Create visualizations
    print("\n📈 Creating visualizations...")
    plot_training_loss(training_results)
    plot_improvement_over_baseline(training_results)
    plot_training_time(training_results)

    # Create summary table
    print("\n📋 Creating summary table...")
    create_summary_table(training_results, eval_results)

    print("\n✅ Analysis complete!")
    print(f"\nCharts saved to: research_mathdial/results/charts/")
    print(f"Summary saved to: research_mathdial/results/ablation_summary.csv")


if __name__ == "__main__":
    main()
