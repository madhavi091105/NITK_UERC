# NITK_UERC — Unconstrained Ear Recognition

**Competition:** UERC (Unconstrained Ear Recognition Challenge) 
**Result:** 🥈 2nd Place
**Publication:** ✅ Accepted — IEEE Council*(awaiting publication)*

---

## About

End-to-end pipeline for biometric ear recognition on unconstrained, real-world images. Covers computer vision preprocessing, multi-model experimentation, and cross-subject evaluation.

Full methodology and results are documented in `/Docs`.

---

## Models Explored

TinyViT · ResNet-50 · EfficientNet-B3 · MobileNetV3 · ViT-Base

> TinyViT (Vision Transformer) gave the best results and was used for the final submission.

---

## Tech Stack

- PyTorch, timm
- OpenCV (CV preprocessing)
- scikit-learn, NumPy
- albumentations

---

## Results

| Dataset | Protocol | Outcome |
|---------|----------|---------|
| UERC | Ear-matching | **2nd Place** |
| NITJ | Full Cross-Subject | `TinyViT_NITJ_Full_CrossSubject_Scores.txt` |

---

## Structure

```
NITK_UERC/
├── Docs/          # Full methodology and documentation
├── MEM_Ear/       # Feature extraction module
├── NITJ/          # NITJ dataset experiments
├── NITK_TRACK1/   # Track 1 files
├── TRACK2/        # Track 2 final submission
└── README.md
```

---

## Setup

```bash
pip install torch torchvision timm opencv-python scikit-learn albumentations
```

---

## Citation

> Accepted for publication — IEEE Council. Citation details will be updated once published.
