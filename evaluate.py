import json
import os
from config import VALID_LABELS, DATA_PATH, TEST_FILE
from classifier import classify_episode, load_labeled_examples

# [run_evaluation() remains unchanged]

def compute_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    """
    Compute overall classification accuracy.
    """
    if not predictions or not ground_truth or len(predictions) != len(ground_truth):
        return 0.0
        
    correct_count = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct_count / len(predictions)

def compute_per_class_accuracy(
    predictions: list[str], ground_truth: list[str]
) -> dict[str, dict]:
    """
    Compute accuracy broken down by each label class.
    """
    # Initialize the stats dictionary for all valid labels
    stats = {label: {"correct": 0, "total": 0, "accuracy": 0.0} for label in VALID_LABELS}

    # Tally totals and correct predictions
    for p, g in zip(predictions, ground_truth):
        if g in stats:
            stats[g]["total"] += 1
            if p == g:
                stats[g]["correct"] += 1

    # Compute final accuracy for each class
    for label in stats:
        if stats[label]["total"] > 0:
            stats[label]["accuracy"] = stats[label]["correct"] / stats[label]["total"]

    return stats

# [format_evaluation_report() remains unchanged]