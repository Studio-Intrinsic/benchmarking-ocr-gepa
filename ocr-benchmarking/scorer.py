"""
JSON Evaluation Scorer

Wraps the TypeScript JSON evaluation script for use in Python.
Uses the exact same scoring logic as the original TypeScript implementation
to ensure 100% compatibility when comparing results.

The scorer compares predicted JSON to ground truth and returns a score
based on field-level accuracy (additions, deletions, modifications).
"""

import json
import os
import subprocess

import dspy


def call_typescript_json_evaluation(actual_json, predicted_json, ignore_case=False):
    """Call the TypeScript evaluation script via Node.js for consistent scoring.

    Args:
        actual_json: Ground truth JSON (dict or JSON string)
        predicted_json: Predicted JSON (dict or JSON string)
        ignore_case: Whether to ignore case when comparing strings

    Returns:
        Score between 0.0 and 1.0 (1.0 = perfect match)
    """
    script_path = os.path.join(os.path.dirname(__file__), "evaluate_json.js")

    try:
        # Convert to JSON strings for CLI
        actual_str = json.dumps(actual_json)
        predicted_str = json.dumps(predicted_json)
        ignore_case_str = "true" if ignore_case else "false"

        # Set environment variable for simple output (just the score)
        env = os.environ.copy()
        env["SIMPLE_OUTPUT"] = "true"

        # Call Node.js script
        result = subprocess.run(
            ["node", script_path, actual_str, predicted_str, ignore_case_str],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )

        if result.returncode != 0:
            print(f"Node.js evaluation error: {result.stderr}")
            return 0.0

        # Parse the score from output
        score = float(result.stdout.strip())
        return score

    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        ValueError,
        FileNotFoundError,
    ) as e:
        print(f"Error calling TypeScript evaluation: {e}")
        return 0.0


def deterministic_json_metric(example, pred, trace=None):
    """DSPy metric for evaluating JSON extraction accuracy.

    Uses the TypeScript scorer to compute accuracy between predicted and
    ground truth JSON. This is used for final evaluation (not training).

    Returns:
        dspy.Prediction with score field (0.0 to 1.0)
    """
    try:
        actual_json = (
            json.loads(example.true_json)
            if isinstance(example.true_json, str)
            else example.true_json
        )
        predicted_json = (
            json.loads(pred.json_output)
            if isinstance(pred.json_output, str)
            else pred.json_output
        )
    except json.JSONDecodeError:
        return 0.0

    # Use exact TypeScript evaluation for guaranteed compatibility
    score = call_typescript_json_evaluation(actual_json, predicted_json)
    return dspy.Prediction(score=score)
