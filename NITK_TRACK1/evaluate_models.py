import os
import configparser
import importlib.util
import numpy as np
import json

import torch

from uerc26_utils import get_min_max_stats, compute_rt1, compute_rt2


# Global variables
ROOT_DIR = 'SUBMISSIONS'
TRACK = "T1"  # Set to "T1" for Track 1 or "T2" for Track 2
WARMUP_RUNS = 0
WARMUP_EVALUATIONS_PER_RUN = 100
RUNS = 1
BATCH_SIZE = 512
DATA_PATH = 'data/sequestered'
DATA_SPLIT = 'test'
PRELOAD_IMAGES = True  # Set to True to preload images into memory


def init_solution(submission_path, track):
    # Load the stats for this submission
    config_path = os.path.join(submission_path, 'config.ini')

    # Read config
    config = configparser.ConfigParser()
    config.read(config_path)

    submitted_track = config.get("SUBMISSION", "track")

    if submitted_track != track:
        return None

    submission_name = config.get("SUBMISSION", "name")
    model_class_name = config.get("MODEL", "model_class")
    model_weights = config.get("MODEL", "weights")
    model_weights = os.path.join(submission_path, model_weights)

    # Dynamically import the model module from the submission directory
    model_file_path = os.path.join(submission_path, 'model.py')
    spec = importlib.util.spec_from_file_location("model", model_file_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    print(f"Initialized solution for submission: {submission_name}, Track: {submitted_track}, Model Class: {model_class_name}")

    # Load solution model and its parameters
    model_class = getattr(model_module, model_class_name)
    model = model_class().model

    # Choose the appropriate device based on the track and availability
    if submitted_track == "T1":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # You can adjust this based on your GPU memory
        batch_size = BATCH_SIZE
    else:
        # For T2, only CPU with batch size of 1 is allowed to ensure accurate latency measurement
        device = 'cpu'
        batch_size = 1

    # Dynamically import the Solution class from the solution.py in the submission directory
    solution_file_path = os.path.join(submission_path, 'solution.py')
    spec = importlib.util.spec_from_file_location("solution", solution_file_path)
    solution_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution_module)

    # Create a Solution instance and load the model
    solution_class = getattr(solution_module, "Solution")
    solution = solution_class(submission_name, DATA_PATH, DATA_SPLIT, pairs_csv=f"pairs_{DATA_SPLIT}.csv", batch_size=batch_size, device=device, preload_images=PRELOAD_IMAGES)
    solution.load_model(model, model_weights)

    return solution


def get_model_size(model):
    total_param_count = 0
    total_param_bytes = 0

    for _, tensor in model.state_dict().items():
        total_param_count += tensor.numel()
        total_param_bytes += tensor.numel() * tensor.element_size()

    return {"num_parameters": total_param_count, "model_size": total_param_bytes}


def evaluate_submission(solution):
    solution.warmup(runs=WARMUP_RUNS, evaluations_per_run=WARMUP_EVALUATIONS_PER_RUN)

    return solution.evaluate(runs=RUNS)


if __name__ == "__main__":
    print(f"Evaluating Track {TRACK}...")

    # Go through each submission directory
    for submission in os.listdir(ROOT_DIR):
        print(f"Processing submission: {submission}")

        submission_path = os.path.join(ROOT_DIR, submission)
        if os.path.isdir(submission_path):
            # Initialize the solution for this submission and track
            solution = init_solution(submission_path, TRACK)

            stats_file = os.path.join(submission_path, "model_stats.json")

            try:
                model_stats = get_model_size(solution.model)
            except Exception as e:
                print(f"Error occurred while fetching model stats for {submission}: {e}")
                model_stats = {"num_parameters": 0, "model_size": 0}

            # Store the model stats for this submission in a JSON file for later analysis
            with open(stats_file, "w") as f:
                json.dump(model_stats, f, indent=4)

            # Check if the metrics for this submission already exist to avoid redundant evaluation
            metrics_file = os.path.join(submission_path, "metrics.json")
            if os.path.exists(metrics_file):
                print(f"Metrics already exist for {submission}, loading from file.")
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                continue

            metrics = evaluate_submission(solution)

            # Store the metrics for this submission in a JSON file for later analysis
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
