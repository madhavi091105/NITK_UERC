# README

## UERC Final Submission Evaluation on NITJ Dataset

This repository contains the evaluation performed on the **NITJ dataset** using our **final UERC submission model**.

### Evaluation Procedure

* The dataset was divided into **Gallery** and **Probe** sets.
* Images were passed through our **final submission model** to generate feature embeddings.
* **Euclidean Distance** was computed between probe and gallery embeddings.
* The gallery image with the minimum distance was selected as the predicted match.

### Distance Metric

Euclidean Distance was used to measure similarity between embeddings:

* Smaller distance → Higher similarity
* Larger distance → Lower similarity

### Pipeline

```text
Input Images
     ↓
Gallery / Probe Split
     ↓
Feature Extraction using Final UERC Model
     ↓
Euclidean Distance Calculation
     ↓
Matching & Evaluation
```

### Note

This repository contains only the evaluation pipeline used to test the final submission model on the NITJ dataset.
