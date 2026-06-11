import gc
import sys
import configparser
import time
import numpy as np
import cv2
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2

sys.path.append("../../")
from uerc26_dataset import UERCPairwiseDataset
from uerc26_utils import compute_metrics


# ── Preprocessing utilities (matching training pipeline exactly) ─────────────
def apply_clahe(image_np, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _np_to_tensor(img_np):
    return torch.from_numpy(img_np.transpose(2, 0, 1).copy()).float().div_(255.0)


# ── Custom dataset: overrides image loading to use cv2 ───────────────────────
class CustomUERCPairwiseDataset(UERCPairwiseDataset):
    """
    Inherits from UERCPairwiseDataset but overrides __getitem__ to use
    OpenCV for image loading and ImageNet normalization — matching the
    notebook's EarPairwiseDataset exactly (no CLAHE, no explicit resize).
    
    preload_images is always forced to False because the parent preloads
    with PIL, which is incompatible with our cv2 pipeline.
    """
    def __init__(self, *args, **kwargs):
        # Force preload_images=False to prevent PIL-based preloading
        kwargs['preload_images'] = False
        super().__init__(*args, **kwargs)
        self._norm = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _load(self, relpath):
        full = os.path.join(self.root_dir, relpath)
        raw = cv2.imread(full)
        if raw is None:
            raise FileNotFoundError(f'Cannot read: {full}')
        # Convert BGR to RGB — NO CLAHE for pairwise evaluation
        # (matches notebook's EarPairwiseDataset exactly).
        # Model's forward() handles resize to 224x224 via F.interpolate.
        raw = cv2.resize(raw, (224, 224))
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        return self._norm(_np_to_tensor(image))

    def __getitem__(self, idx):
        (img1_name, img2_name), label = self.image_pairs[idx], self.labels[idx]
        img1 = self._load(img1_name)
        img2 = self._load(img2_name)
        return (img1, img2), label


# ── Solution class ───────────────────────────────────────────────────────────
class Solution:
    def __init__(self, name, test_data_path, split, image_list_csv="image_list.csv", dataset_split_csv="dataset_split.csv", pairs_csv="pairs_test.csv", batch_size=1, device='cpu', preload_images=False):
        self.name = name
        self.test_data_path = test_data_path
        self.split = split
        self.image_list_csv = image_list_csv
        self.dataset_split_csv = dataset_split_csv
        self.pairs_csv = pairs_csv
        self.batch_size = batch_size
        self.device = device

        # Prepare the dataset and dataloader for evaluation using CustomUERCPairwiseDataset
        test_dataset = CustomUERCPairwiseDataset(self.split, self.test_data_path, self.image_list_csv, self.dataset_split_csv, pairs_csv=self.pairs_csv, preload_images=preload_images)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model = None

    def load_model(self, model, model_parameters_file_path):
        # Load the model parameters from the specified file path and move the model to the appropriate device
        self.model = model
        self.model.load_state_dict(torch.load(model_parameters_file_path, map_location='cpu', weights_only=False), strict=False)
        self.model.to(self.device)
        self.model.eval()

    def get_model_stats(self):
        metrics = {}
        if self.model:
            metrics["num_parameters"] = sum(p.numel() for p in self.model.parameters())
            metrics["model_size"] = sum(p.numel() * p.element_size() for p in self.model.parameters())
        else:
            metrics["num_parameters"] = 0
            metrics["model_size"] = 0

        return metrics

    def __reset_cache(self):
        # Clear GPU and CPU memory to ensure accurate timing and performance measurements
        torch.cuda.empty_cache()
        gc.collect()

    def warmup(self, runs=10, evaluations_per_run=0):
        # Perform warmup runs to stabilize performance and ensure accurate timing for the evaluation runs
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model before warmup.")

        self.__reset_cache()
        self.model.eval()

        if evaluations_per_run == 0:
            evaluations_per_run = len(self.test_dataloader)

        with torch.no_grad():
            for _ in tqdm(range(runs), desc="Warmup", unit="run"):
                for batch_idx, ((img1, img2), _) in enumerate(self.test_dataloader):
                    if batch_idx >= evaluations_per_run:
                        break

                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    _ = self.model(img1)
                    _ = self.model(img2)

    def evaluate(self, runs=1):
        # Function to evaluate the model on the test dataset and compute the specified metrics.
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model before evaluation.")

        self.__reset_cache()
        self.model.eval()

        RUN_DATA = {}
        with torch.no_grad():
            for i in tqdm(range(runs), desc="Evaluating", unit="run"):
                scores, labels, times = [], [], []

                for (img1, img2), label in tqdm(self.test_dataloader, desc="Processing batches", unit="batch"):
                    img1, img2 = img1.to(self.device), img2.to(self.device)

                    start = time.time()
                    e1 = F.normalize(self.model(img1))
                    e2 = F.normalize(self.model(img2))
                    end = time.time()
                    times.append(end - start)

                    sim = F.cosine_similarity(e1, e2)
                    scores.extend(sim.cpu().numpy())
                    labels.extend(label.cpu().numpy())

                RUN_DATA[str(i)] = compute_metrics(np.array(labels), np.array(scores), times=np.array(times))

        return RUN_DATA


if __name__ == "__main__":
    # Load information from config.ini
    config = configparser.ConfigParser()
    config.read("config.ini")

    submitted_track = config.get("SUBMISSION", "track")
    submission_name = config.get("SUBMISSION", "name")

    model_class_name = config.get("MODEL", "model_class")
    model_weights = config.get("MODEL", "weights")

    # Load solution model and its parameters
    # Adjust the initialization as needed based on your model's requirements
    from model import *
    solution_model = eval(model_class_name)()
    # Get the actual model from the wrapper
    model = solution_model.model

    # Choose the appropriate device based on the track and availability
    if submitted_track == "T1":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # You can adjust this based on your GPU memory
        batch_size = 1024
    else:
        # For T2, only CPU with batch size of 1 is allowed to ensure accurate latency measurement
        device = 'cpu'
        batch_size = 1

    data_path = config.get("DATA", "data_path")
    data_split = config.get("DATA", "data_split")

    # Create a Solution instance and load the model
    solution = Solution(submission_name, data_path, data_split, pairs_csv=f"pairs_{data_split}.csv", batch_size=batch_size, device=device)
    solution.load_model(model, model_weights)

    # Adjust the number of warmup runs as needed
    solution.warmup(runs=10, evaluations_per_run=100)

    solution.evaluate(runs=1)
