# UERC26 Starting package

## Installation

```
pip install -r requirements.txt
```

## Dataset

The public part of the dataset comprises of 248,655 images of 1,310 subjects with .jpg and .png files same as for the UERC23. Additionally there is an anonymized sequestered dataset of 1,670 images of 70 subjects.

## Files

As a starting point for the UERC26, we provide the following files:

### uerc26_dataset.py

Pytorch dataset classes for loading the data. It provides two classes: UERCDataset for the identification task and UERCPairwiseDataset for the verification task. Both classes can be used to load the public and sequestered datasets. There are also some helper functions for splitting the data and creating pairs for the verification task.

Feel free to modify the dataset classes as needed, but make sure to keep the same interface for loading the data.

### uerc26_utils.py

This file contains utility functions for computing evaluation metrics, such as ROC AUC, and for computing the RT1 and RT2 metrics based on the labels, scores, and inference times. The compute_metrics function takes in the labels, scores, inference times, and optionally the model to compute the evaluation metrics.

### extract_features.py

This is a script that can be used to extract features from the images for all solutions in the SUBMISSIONS folder. It will go through all the solutions, load the corresponding model and extract features from the images in the anonymized sequestered dataset. The extracted features will be stored in the solution's folder in a file called sequestered_features.npy. Extracted features will be used for the final evaluation of the solutions and should be submitted along with the solution for the final evaluation in Track 1.

### evaluate_models.py

This is the main evaluation script for the UERC26. It will compute performance for all submitted models using the sequestered dataset. The script will go through all the solutions in SUBMISSIONS folder, read configuration files (config.ini) for each solution, load the corresponding model and solution classes and compute performance on the sequestered dataset. Computed performance metrics for each solution will be stored in the metrics.json file in the same folder as the solution. The script will also print out the performance metrics for each solution and the overall ranking based on the computed metrics.

### evaluate_features.py

This is a script that can be used to evaluate the performance of the extracted features from the images in the anonymized sequestered dataset. It will go through all the solutions in the SUBMISSIONS folder, load the corresponding extracted features and compute performance metrics based on the extracted features. The computed metrics will be stored in a file called metrics.json in the same folder as the solution. The script will also print out the performance metrics for each solution and the overall ranking based on the computed metrics.

### score.py

This file contains the implementation of the scoring function for the UERC26. The scoring function takes in the computed metrics for each solution in corresponding submission folder and computes a final score based on the defined evaluation criteria. The score is computed based on the RT1 and RT2 metrics, as well as the inference time. The scoring function will be used to determine the final ranking of the solutions based on their performance on the sequestered dataset.

### SUBMISSIONS folder

This folder is where all the submitted solutions will be stored. Each solution should be in its own subfolder with the following structure:

```SUBMISSIONS/
    solution_name/
        config.ini
        model.py
        solution.py
        submitted_model_weights.pt
        sequestered_features.npy
```

- config.ini: Configuration file for the solution. It should contain the following information:
    - track: The track for which the solution is submitted (T1 or T2)
    - name: Name of the solution
    - model_class: Name of the model class defined in model.py
    - weights: Relative path to the model weights file (submitted_model_weights.pt)
    - any other hyperparameters needed for the solution

- model.py: This file should contain the definition of the model class that will be used for the solution. The model class should inherit from torch.nn.Module and implement the forward method.

- solution.py: This file should contain the implementation of the solution class that will be used for the evaluation. The solution class should have a method called evaluate that takes in the sequestered dataset and computes the performance metrics for the solution.

- submitted_model_weights.pt: This file should contain the weights of the model that will be used for the solution. The weights should be saved using torch.save() and should be compatible with the model class defined in model.py. Solutions file computes performance metrics over defined number of runs and returns the computed metrics as a dictionary.

- sequestered_features.npy: This file should contain the extracted features from the images in the anonymized sequestered dataset. The features should be extracted using the model defined in model.py and should be stored as a numpy array using np.save(). The extracted features will be used for the final evaluation of the solutions and should be submitted along with the solution for the final evaluation.

## Submission Guidelines

Each solution should be submitted as a zip file containing the solution folder with the structure defined above. The zip file should be named as solution_name.zip and should be sent to the organizers before the submission deadline. The organizers will extract the zip files and place the solution folders in the SUBMISSIONS folder for evaluation. It is important to ensure that the submitted solution follows the defined structure and includes all the necessary files for evaluation. The organizers will not be responsible for any issues arising from incorrectly structured submissions or missing files. Participants are encouraged to test their solutions using the provided evaluation scripts before submission to ensure that they are working correctly and producing valid results.

## Baseline model

Baseline model is ConvNext-Base pretrained on ImageNet-1K and finetuned on the public dataset for the identification task with following hyperparameters:
- optimizer: AdamW
- learning rate: 1e-4
- weight decay: 1e-3
- batch size: 64
- epochs: 100

Baseline model performance on the sequestered dataset is as follows:
- VER@0.1%FAR: 26.23%
- VER@1%FAR: 46.31%
- EER: 15.69%
- AUC: 92.15%
- Number of parameters: 87564416
- Model size: 350257664

## Scoring

Submissions are evaluated against a provided baseline model that serves as the anchor for all score normalization. Both ranking tracks use Verification Rate at FAR=0.1% (VER@0.1%) as the primary accuracy metric, computed from the ROC curve over all sequestered dataset pairs. The baseline's VER@0.1% sets the floor for performance normalization. Similarly, the baseline's parameter count and model size define the upper bounds for **Track 1**, so only models smaller and more compact than the baseline can achieve a positive size/parameter score. The final composite scores are RT1 = 0.5v + 0.2p + 0.3s and RT2 = 0.7v + 0.3t, where all components are min–max normalized within these baseline-anchored bounds requiring competitors to surpass the baseline in both accuracy and efficiency to rank well.

## Evaluation Process

The evaluation process will be conducted in two phases:
1. Interim Evaluations: In this phase, the submitted solution feature vectors of the sequestered dataset will be evaluated. The interim evaluations will provide feedback to the participants on the performance of their solutions and allow them to make improvements before the final evaluation. Participants are encouraged to use the interim evaluations to improve their solutions and achieve better performance in the final evaluation.
2. Final Evaluation: In this phase, the submitted solutions will be evaluated on the sequestered dataset using the defined evaluation metrics. The final evaluation will determine the final ranking of the solutions based on their performance on the sequestered dataset. The final evaluation will be conducted after the submission deadline and the results will be announced shortly after the evaluation is completed. During the final evaluation, the organizers will compute the performance metrics for each solution based on the defined evaluation criteria and compute the final score using the scoring function defined in score.py. For both Tracks, accuracy metrics will be computed based on the performance of the solutions on the extracted features from the anonymized sequestered dataset. For Track 2, the final evaluation of inference time will be conducted using the submitted model weights and the defined evaluation protocol on defined hardware (Raspberry Pi 4). The organizers will ensure that the evaluation is conducted in a fair and consistent manner for all solutions.

Only solutions outperforming the baseline will be included in the final ranking. The baseline performance will be determined based on the performance of a simple model on the sequestered dataset.

## Questions and Support

If you have any questions or need support during the development of your solution, please feel free to reach out to the organizers. We are here to help and provide guidance throughout the competition. If you notice any issues with the provided code or have suggestions for improvements, please let us know. We want to ensure that the provided code is as helpful as possible for all participants and we welcome any feedback or contributions to improve the codebase. You can contact the organizers by email.
