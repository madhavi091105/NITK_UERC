# UERC26 Ear Verification — Submission
**Team:** NITK Submission  
**Model:** Triple-Backbone Siamese Network (19M params)  
**Backbones:** ConvNeXt-Atto + EfficientNet-B0 + ResNet-18

---

## Folder Structure

```
submission_nitk/
├── README.md
├── pairs_test.csv          ← Test pairs (img1, img2, label)
├── weights/
│   └── best_model.pth      ← Trained model weights
└── code/
    ├── model.py            ← Model architecture (EarSiameseModel)
    ├── dataset.py          ← PairDataset class
    ├── metrics.py          ← AUC, EER, FIF, G, VER@FAR
    └── run_inference.py    ← Main inference script
```

---

## How to Run

### 1. Install dependencies
```bash
pip install torch torchvision timm pandas numpy scikit-learn scipy pillow
```

### 2. Run inference
```bash
python code/run_inference.py \
    --csv      pairs_test.csv \
    --img_dir  /path/to/UERC26_Oriented/data/sequestered_anonymized/ \
    --weights  weights/best_model.pth \
    --output   submission.csv
```

### 3. Output files
- `submission.csv` — Predictions with distances
- `submission_metrics.json` — AUC, EER, FIF, G, VER metrics

---

## Model Details

| Property        | Value                                     |
|----------------|-------------------------------------------|
| Architecture   | Triple-Backbone Siamese                   |
| Backbones      | ConvNeXt-Atto, EfficientNet-B0, ResNet-18 |
| Feature dim    | 320 + 1280 + 512 = 2112                   |
| Embedding dim  | 256                                       |
| Loss           | Contrastive Loss                          |
| Image size     | 128 × 128                                 |
| Parameters     | ~19.8M                                    |

---

## CSV Format

`pairs_test.csv` columns (no header):
```
img1_filename, img2_filename, label
```
- `label = 1` → same person (genuine pair)  
- `label = 0` → different persons (impostor pair)
