import os
import configparser
import importlib.util
import numpy as np

import torch

from uerc26_dataset import UERCDataset

# Global variables
ROOT_DIR = 'SUBMISSIONS'
BATCH_SIZE = 512
DATA_PATH = 'data/sequestered_anonymized'
DATA_SPLIT = 'test'


# Temporary dataset inheriting from UERCDataset to get image names
class UERCDataset(UERCDataset):
    def __init__(self, split, data_path, full_image_list_csv="image_list.csv", data_split_csv="dataset_split.csv"):
        super().__init__(split, data_path, full_image_list_csv, data_split_csv)

    def __getitem__(self, idx):
        # Override to return the image name along with the image tensor
        image, _ = super().__getitem__(idx)
        image_name = self.images[idx]
        return image, image_name


def extract_features(submission_path):
    # Load the stats for this submission
    config_path = os.path.join(submission_path, 'config.ini')

    # Read config
    config = configparser.ConfigParser()
    config.read(config_path)

    model_class_name = config.get("MODEL", "model_class")
    model_weights = config.get("MODEL", "weights")
    model_weights = os.path.join(submission_path, model_weights)

    # Dynamically import the model module from the submission directory
    model_file_path = os.path.join(submission_path, 'model.py')
    spec = importlib.util.spec_from_file_location("model", model_file_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # Load solution model and its parameters
    model_class = getattr(model_module, model_class_name)
    model = model_class().model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.load_state_dict(torch.load(model_weights), strict=False)
    model.to(device)
    model.eval()

    dataset = UERCDataset(DATA_SPLIT, DATA_PATH, full_image_list_csv="image_list.csv", data_split_csv="dataset_split.csv")

    print(f"Extracting features for submission: {submission_path}")
    print(f"Using device: {device}")
    print(f"Number of images to process: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    # Extract features for all images in the dataset
    features = {}

    with torch.no_grad():
        for images, image_names in dataloader:
            images = images.to(device)
            batch_features = model(images)

            # Move features to CPU and convert to numpy
            batch_features = batch_features.cpu().numpy()

            # Store features in the dictionary with image names as keys
            for img_name, feat in zip(image_names, batch_features):
                features[img_name] = feat

    # Save the extracted features to a file for later analysis
    features_path = os.path.join(submission_path, "sequestered_features.npy")
    np.save(features_path, features)
    print(f"Extracted features saved to {features_path}")


if __name__ == "__main__":
    # Go through each submission directory
    for submission in os.listdir(ROOT_DIR):
        submission_path = os.path.join(ROOT_DIR, submission)

        if os.path.isdir(submission_path):
            extract_features(submission_path)


        # Load and check if features were extracted successfully
        features_path = os.path.join(submission_path, "sequestered_features.npy")
        if os.path.exists(features_path):
            print(f"Features successfully extracted for submission: {submission}")

            # Load the features and print some stats
            features = np.load(features_path, allow_pickle=True).item()
            print(f"Number of features extracted: {len(features.keys())}")
            print(f"Example feature vector shape: {next(iter(features.values())).shape}")
