import os
import csv
import json
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from uerc26_utils import compute_metrics


# Global variables
ROOT_DIR = 'SUBMISSIONS'
BATCH_SIZE = 512
DATA_PATH = 'data/sequestered'
DATA_SPLIT = 'test'


if __name__ == "__main__":
    print("Evaluating submitted features on the sequestered dataset")

    # Go through each submission directory
    for submission in os.listdir(ROOT_DIR):
        submission_path = os.path.join(ROOT_DIR, submission)

        if os.path.isdir(submission_path):
            # Check if sequestered_features.npy exists in the submission directory
            features_path = os.path.join(submission_path, 'sequestered_features.npy')

            if os.path.exists(features_path):
                print(f"Evaluating features for submission: {submission}")

                # Load the features
                features = np.load(features_path, allow_pickle=True).item()

                # Load the mapping from anonymized image names to original image names from sequestered_anonymized_mapping.csv
                mapping_path = os.path.join(DATA_PATH, 'sequestered_anonymized_mapping.csv')
                mapping = {}
                with open(mapping_path, 'r') as f:
                    csv_reader = csv.reader(f, delimiter='\t')
                    # Skip header
                    next(csv_reader)

                    for row in csv_reader:
                        original_name, anonymized_name = row

                        mapping[original_name] = f'0/{anonymized_name}'

                # Load the pairs and labels from pairs_test.csv
                pairs_path = os.path.join(DATA_PATH, f'pairs_{DATA_SPLIT}.csv')
                scores = []
                labels = []
                with open(pairs_path, 'r') as f:
                    csv_reader = csv.reader(f, delimiter='\t')
                    # Skip header
                    next(csv_reader)

                    for row in csv_reader:
                        img1, img2, label = row
                        label = int(label)

                        # Get the features for the original image names
                        feat1 = features[mapping[img1]]
                        feat2 = features[mapping[img2]]

                        sim = np.diag(cosine_similarity(feat1.reshape(1, -1), feat2.reshape(1, -1)))
                        scores.append(sim)
                        labels.append(label)

                # Compute metrics for the features
                metrics = compute_metrics(np.array(labels), np.array(scores), times=np.array([0]))

                print(f"Metrics for submission {submission}:")
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value}")

                # Add 'run_id' 0 to the metrics for consistency with the expected format in score.py and output of the evalute_models.py
                metrics = {'0': metrics}

                # Store the metrics for this submission in a JSON file for later analysis
                with open(os.path.join(submission_path, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=4)
