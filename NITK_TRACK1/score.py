import os
import json
import configparser
import numpy as np

from uerc26_utils import get_min_max_stats, compute_rt1, compute_rt2


# Global variables
ROOT_DIR = 'SUBMISSIONS'


if __name__ == "__main__":
    print("Aggregating metrics for all submissions")

    METRICS = {}
    BASELINE_METRICS = {}

    BASELINE_MODEL = "baseline_convnext_base"

    # Go through each submission directory
    for submission in os.listdir(ROOT_DIR):
        submission_path = os.path.join(ROOT_DIR, submission)

        if os.path.isdir(submission_path):
            config_path = os.path.join(submission_path, 'config.ini')

            # Read config
            config = configparser.ConfigParser()
            config.read(config_path)

            submitted_track = config.get("SUBMISSION", "track")
            submission_name = config.get("SUBMISSION", "name")

            metrics_file = os.path.join(submission_path, "metrics.json")

            if not os.path.exists(metrics_file):
                print(f"No metrics found for {submission}, skipping...")
                continue

            # Load model stats for this submission
            stats_file = os.path.join(submission_path, "model_stats.json")
            if os.path.exists(stats_file):
                with open(stats_file, "r") as f:
                    model_stats = json.load(f)

            if model_stats["num_parameters"] == 0 or model_stats["model_size"] == 0:
                print(f"Invalid model stats for {submission}, skipping...")
                continue

            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            # Merge model stats into metrics for this submission
            for key, value in model_stats.items():
                metrics["0"][key] = value

            # Store the metrics for this submission
            if submission_name == BASELINE_MODEL:
                BASELINE_METRICS = metrics["0"]
            else:
                METRICS[submission_name] = metrics

    # Save the aggregated metrics for all submissions in a JSON file for later analysis
    with open("aggregated_metrics.json", "w") as f:
        json.dump(METRICS, f, indent=4)

    # Calculate average metrics across all submissions
    if METRICS:
        avg_metrics = {}
        for submission in METRICS.keys():
            avg_metrics[submission] = {}

            RUN_IDS = [key for key in METRICS[submission].keys()]

            for key in METRICS[submission]["0"].keys():
                avg_metrics[submission][key] = np.mean([METRICS[submission][ri][key] for ri in RUN_IDS])

        stats = get_min_max_stats(avg_metrics, baseline_metrics=BASELINE_METRICS)

        score, v, p, s, t = 0, 0, 0, 0, 0
        for submission in METRICS.keys():
            if submitted_track == "T1":
                score, v, p, s = compute_rt1(avg_metrics[submission], stats)
            else:
                score, v, t = compute_rt2(avg_metrics[submission], stats)

            avg_metrics[submission]["score"] = score
            avg_metrics[submission]["v"] = v
            avg_metrics[submission]["p"] = p
            avg_metrics[submission]["s"] = s
            avg_metrics[submission]["t"] = t

        # Save the aggregated metrics as csv for all submissions
        with open("aggregated_metrics.csv", "w") as f:
            header = 'name,' + ','.join(list(avg_metrics[submission].keys())) + '\n'
            f.write(header)

            for submission in avg_metrics.keys():
                row = submission + ',' + ','.join([str(avg_metrics[submission][key]) for key in avg_metrics[submission].keys()]) + '\n'
                f.write(row)
