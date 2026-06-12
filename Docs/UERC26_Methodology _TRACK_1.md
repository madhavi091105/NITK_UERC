# UERC 2026 — System Description

**Team Name:** *(fill in)*
**Contact E-mail:** *(fill in)*
**Model Name:** Pruned-TinyViT-11M (KD Chain: SwinV2-Base → TinyViT-21M → TinyViT-11M → Taylor Pruning)

---

## 1. Overview

We present a cascaded knowledge distillation (KD) pipeline for ear biometric verification. A large, architecturally strong teacher model (SwinV2-Base, 88M parameters) first trains on the UERC 2026 dataset. Its learned representations are then progressively compressed through two distillation hops — first into TinyViT-21M, then into TinyViT-11M — before the final model undergoes structured Taylor pruning with post-pruning re-distillation. The final deployed model achieves **VER@0.1% FAR ≈ 50.6%**, **EER ≈ 8%**, and **AUC ≈ 96.8%** on the test set, at substantially reduced parameter count.

No external datasets were used. All training was performed exclusively on the official UERC 2026 competition data.

---

## 2. Dataset & Preprocessing

### 2.1 Dataset

- **Source**: Official UERC 2026 competition package only. No external data was used at any stage.
- **Training set**: approximately 180,000 images across multiple identities.
- **Format**: Images were originally provided as PNG/JPEG. We converted the entire dataset to **WebP lossless** format prior to training.
  - WebP lossless is typically 25–35% smaller than PNG with identical pixel values, reducing disk I/O latency.
  - Unlike JPEG, lossless WebP introduces no compression artefacts. JPEG artefacts in ear texture can be mistaken for real anatomical features by a model, injecting noise into learned embeddings — a particular risk in biometrics. WebP eliminates this entirely.
- **Resolution**: All images were resized to **224×224** pixels using bilinear interpolation (`cv2.INTER_LINEAR`). Bilinear was chosen over nearest-neighbour (blocky artefacts) and bicubic (ringing artefacts at edges). For SwinV2-Base, which requires 256×256 input, a secondary dynamic resize from 224→256 was applied inside the model's `forward()` method using bicubic interpolation so the preprocessing pipeline remained unified at 224×224.

### 2.2 CPU-Side Preprocessing & Augmentation

Applied inside DataLoader workers on the CPU:

**CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- Applied randomly with probability **p = 0.5** during training; applied deterministically at inference.
- CLAHE operates in the LAB colour space: the image is converted RGB→LAB, the L-channel histogram is equalised per tile (tile grid `8×8`, clip limit `2.0`), and the result is converted back LAB→RGB.
- Standard global histogram equalisation can blow out bright regions; CLAHE avoids this by clipping each tile's histogram redistribution at the clip limit and blending tile boundaries via bilinear interpolation.
- Ear images frequently suffer from low local contrast (highlights, shadows, hair occlusion). CLAHE improves visibility of fine anatomical structures — the helix, antihelix, tragus, concha — without introducing noise amplification.

**Random Unsharp Masking (Sharpening)**
- Applied randomly with probability **p = 0.3** during training; not applied at inference.
- Implemented as: `sharpened = (1 + amount) × original − amount × GaussianBlur(original, σ=1.0)` with `amount=0.5`.
- Probability kept at 0.3 (not higher) because over-sharpening introduces ringing artefacts that do not occur naturally in ear images and could mislead the model's edge detectors.

### 2.3 GPU-Side Augmentation

Applied on-device after the CPU→GPU transfer, inside the training loop:

- `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)`
- `RandomGrayscale(p=0.2)`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` (ImageNet statistics)

Moving colour augmentations to the GPU removed them from the CPU DataLoader bottleneck and allowed them to run concurrently with the next batch's disk I/O.

### 2.4 DataLoader Optimisations

| Setting | Value | Reason |
|---|---|---|
| `num_workers` | 6–12 (varies per notebook) | Parallel CPU preprocessing |
| `pin_memory=True` | ✓ | Allocates tensors in page-locked memory, enabling direct DMA transfer without an intermediate CPU copy |
| `persistent_workers=True` | ✓ | Workers stay alive between epochs, avoiding process fork + module reimport overhead each epoch |
| `prefetch_factor` | 2–4 | Workers prefetch ahead so the GPU is never idle waiting for data |
| `drop_last=True` | training only | Avoids partial-batch shape issues with BatchNorm |
| `non_blocking=True` | `.to(device)` calls | Makes CPU→GPU transfers asynchronous; Python returns immediately and starts next prefetch while transfer finishes on a CUDA copy stream |
| `torch.backends.cudnn.benchmark = True` | ✓ | cuDNN profiles available conv algorithms on the first few batches and locks in the fastest for the fixed input shape |
| `torch.backends.cuda.matmul.allow_tf32 = True` | ✓ | Enables TensorFloat-32 on Ampere GPUs for ~8× faster matmuls with float32-range outputs |

---

## 3. Pipeline Architecture

The full pipeline consists of four sequential stages:

```
Stage 1: SwinV2-Base training (teacher)         — 88M parameters
              ↓ Knowledge Distillation
Stage 2: TinyViT-21M distillation (student 1)  — 21M parameters
              ↓ Knowledge Distillation
Stage 3: TinyViT-11M distillation (student 2)  — 11M parameters
              ↓ Taylor Pruning + Re-distillation
Stage 4: Pruned TinyViT-11M (final model)       — ~6M parameters (45% reduction)
```

The motivation for using architecturally similar models in the KD chain is that knowledge transfers more effectively between models with similar inductive biases. SwinV2 and TinyViT both use hierarchical window-based self-attention, making TinyViT a natural learner from SwinV2 — and a capable intermediate teacher for the smaller TinyViT-11M — compared to pairing with a CNN or a standard ViT.

---

## 4. Loss Function: AdaFace

All stages use **AdaFace** [Kim et al., 2022] as the classification loss head.

AdaFace builds on ArcFace (additive angular margin softmax). ArcFace imposes a fixed angular margin `m` on the target class logit:

```
L = −log [ exp(s · cos(θ_yi + m)) / Σ_j exp(s · cos(θ_j)) ]
```

AdaFace extends this by **estimating image quality from the L2 norm of the feature embedding** and modulating the margin adaptively:

- High-norm embeddings (high-quality images) → **larger margin** → tighter, more discriminative class boundaries
- Low-norm embeddings (low-quality/occluded images) → **smaller margin** → model is more lenient, not penalised for noisy gradient signal

The margin modulation is:

```
margin_scaler = clamp(h × (‖z‖ − batch_mean) / batch_std, −1, 1)
g_add         = m × (1 + margin_scaler)
target_logit  = cos(θ + m) − g_add
```

where `batch_mean` and `batch_std` are exponential moving averages of embedding norms, updated each batch with `t_alpha=0.01`.

**Hyperparameters used throughout all stages:**

| Parameter | Value |
|---|---|
| Scale `s` | 64.0 |
| Base margin `m` | 0.25 |
| Quality coefficient `h` | 0.333 |
| EMA decay `t_alpha` | 0.01 |
| Weight initialisation | Xavier uniform |

**Why AdaFace for ears:** Ear images have high quality variance — motion blur, hair occlusion, pose extremes, poor lighting. A fixed-margin loss (ArcFace) penalises all samples equally regardless of quality. AdaFace's quality-adaptive margin prevents low-quality images from injecting large, noisy gradients that corrupt the embedding space.

**Cosine similarity at inference:** Embeddings are L2-normalised inside the model's `forward()` (making them unit vectors). At verification time, identity matching is performed via cosine similarity — this is geometrically consistent with angular-margin training, as cosine similarity directly measures the angle between embedding vectors.

---

## 5. Stage 1: SwinV2-Base Teacher Training

### 5.1 Architecture

**Backbone:** `swin_v2_b` (SwinV2-Base) from torchvision, pretrained on ImageNet-22K fine-tuned on ImageNet-1K.

SwinV2-Base uses **shifted window self-attention** rather than global self-attention:
- The feature map is divided into non-overlapping local windows (default `7×7`).
- Self-attention is computed only within each window — O(n) in image size vs O(n²) for standard ViT.
- **Shifted windows** alternate between regular and shifted partitions across layers, allowing cross-window information flow while maintaining computational efficiency.
- A **hierarchical feature pyramid** produces representations at multiple scales (like CNNs), capturing both fine-grained local texture (ear anatomy) and global shape.

**Why SwinV2 over a standard ViT (e.g., DINO):** Ear recognition depends on fine local structures — the precise topology of the helix, antihelix, tragus, and concha. Standard ViT global self-attention allows all patches to attend to all others equally, diluting spatial locality. SwinV2's local window attention preserves these fine-grained spatial features. The hierarchical pyramid additionally means the model can simultaneously detect low-level texture and high-level anatomical shape.

**Input resolution:** SwinV2-Base requires `256×256`. Images loaded at 224×224 are dynamically resized to 256×256 inside `forward()` via bilinear interpolation.

**Model structure:**
```
SwinV2EarModel
├── backbone: swin_v2_b (pretrained)
│   └── head: nn.Identity()          ← original classifier removed
│   └── output dim: 1024
└── projection: nn.Sequential
    ├── nn.Linear(1024, 512)
    └── nn.BatchNorm1d(512)
[forward]: L2-normalise projection output → 512-dim unit embedding
```

Total trainable parameters: ~88M (backbone) + projection head.

### 5.2 Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 15 (single unfrozen phase) |
| Batch size | 192 |
| Optimiser | AdamW |
| Backbone LR | 1e-4 |
| Projection LR | 5e-4 |
| AdaFace head LR | 5e-4 |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealingLR, T_max=15, η_min=1e-6 |
| AMP dtype | float16 + GradScaler |
| Gradient clipping | max norm 1.0 (backbone + head) |
| Early stopping patience | 4 epochs (on val AUC) |
| Checkpoint metric | VER@0.1% FAR |

**Single-phase training rationale:** Rather than a two-phase freeze-then-unfreeze approach, the backbone is fully unfrozen from epoch 1. The AdaFace head is initialised with Xavier uniform weights, and backbone LR is set conservatively (1e-4 vs 5e-4 for new layers) to prevent catastrophic forgetting of pretrained features while still allowing full end-to-end fine-tuning from the start.

**AMP note:** This stage ran on an L4 GPU (Turing architecture, no native bfloat16). Float16 AMP with GradScaler was used. GradScaler multiplies the loss by a large scale factor before `.backward()` to prevent float16 gradient underflow (float16 range: ~6×10⁻⁵ to 65504), then unscales before `optimizer.step()`.

**Best checkpoint selection:** Model saved on highest **VER@0.1% FAR** (not AUC) on the validation pairwise set.

### 5.3 Problems Encountered

**GPU starvation from CPU bottleneck:** Early training runs showed the GPU utilised at ~15% because the CPU data pipeline (cv2.imread, PIL↔cv2 conversions, CLAHE, augmentations all running synchronously) took ~300ms per batch while the GPU finished in ~50ms. Fixed by: removing redundant PIL↔cv2 round-trips (unified into a single `numpy→tensor` path via `torch.from_numpy().permute(2,0,1).float().div_(255.0)`), moving colour augmentations to GPU, and adding `persistent_workers`, `pin_memory`, and higher `prefetch_factor`. A profiling block (20 batches) was run at epoch 1 to confirm the bottleneck location before optimising.

---

## 6. Stage 2: TinyViT-21M Student Distillation (Teacher: SwinV2-Base)

### 6.1 Architecture

**Backbone:** `tiny_vit_21m_224.dist_in22k_ft_in1k` from `timm`, pretrained via knowledge distillation on ImageNet-22K, fine-tuned on ImageNet-1K.

TinyViT is a compact vision transformer that adopts a window-attention mechanism architecturally similar to SwinV2 — the reason it was selected as the student in this chain. The similarity in inductive bias (local window attention, hierarchical structure) makes the distilled knowledge from SwinV2 directly usable rather than requiring the student to re-learn fundamentally different representations.

**Model structure:**
```
TinyViT21MStudent
├── backbone: tiny_vit_21m_224 (pretrained, num_classes=0)
│   └── output dim: 576
└── projection: nn.Sequential
    ├── nn.Linear(576, 512)
    └── nn.BatchNorm1d(512)
[forward]: L2-normalise → 512-dim unit embedding (+ norms if return_norms=True)
```

### 6.2 Knowledge Distillation Setup

The distillation loss combines a **hard loss** (AdaFace classification against ground-truth labels) and a **soft loss** (MSE embedding alignment with the frozen teacher):

```
L_total = (1 − α) × L_hard  +  α × L_soft

L_hard = AdaFace(student_embeddings, student_norms, labels)
L_soft = MSE(student_embeddings, teacher_embeddings)
```

**Dynamic alpha schedule:** Alpha linearly decays from 0.90 to 0.20 over 40 epochs.
- Early epochs: 90% distillation, 10% classification. The student is guided strongly by the teacher's representation before it has learned anything useful.
- Late epochs: 20% distillation, 80% classification. The student refines on the actual task rather than over-fitting to the teacher.

**Resolution handling:** SwinV2-Base teacher requires 256×256. Images are loaded and processed at 224×224 for the student. Inside the training loop, `F.interpolate(..., size=(256,256), mode='bicubic', align_corners=False)` creates a separate 256×256 copy for the teacher forward pass only. The student always sees 224×224.

**Teacher handling:** Loaded from the Stage 1 best checkpoint, compiled with `torch.compile()` for faster frozen inference, then all parameters frozen (`requires_grad=False`). A `_orig_mod.` prefix introduced by `torch.compile` into state_dict keys is stripped during loading with a key-renaming pass.

### 6.3 Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 40 (full run) |
| Batch size | 256 |
| Optimiser | AdamW (fused=True) |
| Student backbone LR | 1e-4 |
| AdaFace head LR | 5e-4 |
| Weight decay | 1e-4 |
| LR schedule | LinearLR warmup (2 epochs, start_factor=0.1) → CosineAnnealingLR (T_max=38, η_min=1e-6) |
| AMP dtype | bfloat16 (no GradScaler) |
| Gradient clipping | max norm 1.0 |
| Alpha KD schedule | 0.90 → 0.20 (linear decay) |
| Early stopping patience | 8 epochs (on VER@0.1%) |
| Checkpoint metric | VER@0.1% FAR |

**bfloat16 vs float16:** This stage ran on an A100 (Ampere). BFloat16 has the same 8-bit exponent range as float32 (vs float16's 5-bit), so gradients do not underflow and GradScaler is not needed. Using GradScaler with bfloat16 would over-scale gradients, producing NaN weights → BatchNorm corruption → DataLoader worker segfault at the next epoch boundary. This was a bug encountered in early experiments and corrected.

**LR warmup:** 2-epoch linear warmup prevents large gradient steps in the first epochs when the AdaFace head is still randomly initialised. Without warmup the head's large random gradients can corrupt the pretrained backbone's features in the first few batches.

---

## 7. Stage 3: TinyViT-11M Student Distillation (Teacher: TinyViT-21M)

### 7.1 Architecture

**Backbone:** `tiny_vit_11m_224.dist_in22k_ft_in1k` from `timm`.

TinyViT-11M is a smaller variant of TinyViT-21M with a reduced backbone output dimension of **448** (vs 576 for the 21M model).

**Model structure:**
```
TinyViT11MStudent
├── backbone: tiny_vit_11m_224 (pretrained, num_classes=0)
│   └── output dim: 448
└── projection: nn.Sequential
    ├── nn.Linear(448, 512)
    └── nn.BatchNorm1d(512)
[forward]: L2-normalise → 512-dim unit embedding (+ norms if return_norms=True)
```

### 7.2 Knowledge Distillation Setup

Identical dual-loss setup to Stage 2:

```
L_total = (1 − α) × L_hard  +  α × L_soft

L_hard = AdaFace(student_embeddings, student_norms, labels)
L_soft = MSE(student_embeddings, teacher_embeddings)
```

- **Teacher**: TinyViT-21M best checkpoint from Stage 2 (frozen, compiled)
- **Student**: TinyViT-11M (pretrained backbone, trainable)
- Both teacher and student operate at **224×224** — no resolution split needed at this stage, simplifying the training loop compared to Stage 2.

Dynamic alpha: same 0.90→0.20 linear decay schedule.

### 7.3 Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 30 (full run) |
| Batch size | 256 |
| Optimiser | AdamW (fused=True) |
| Student backbone LR | 1e-4 |
| AdaFace head LR | 5e-4 |
| Weight decay | 1e-4 |
| LR schedule | LinearLR warmup (2 epochs) → CosineAnnealingLR (T_max=28, η_min=1e-6) |
| AMP dtype | bfloat16 (no GradScaler) |
| Gradient clipping | max norm 1.0 |
| Alpha KD schedule | 0.90 → 0.20 (linear decay) |
| Checkpoint metric | VER@0.1% FAR |

---

## 8. Stage 4: Structured Pruning + Re-distillation

### 8.1 Motivation

The TinyViT-11M student from Stage 3, while already compact, still contains redundant capacity after distillation. Structured pruning removes entire output channels from linear/conv layers — reducing both parameter count and inference FLOPs — followed by re-distillation to recover any accuracy lost from pruning.

### 8.2 Pruning Method: First-Order Taylor Importance

**Library:** `torch-pruning` (`tp.importance.TaylorImportance` + `tp.pruner.MagnitudePruner`)

**Method:** First-order Taylor expansion importance scoring. The true cost of removing a parameter `w_i` is:

```
ΔL = L(w \ w_i) − L(w)
```

Computing this exactly requires a forward pass per candidate (intractable). The first-order Taylor approximation gives:

```
ΔL ≈ (∂L/∂w_i) · w_i
```

Each filter's importance score is therefore: **|gradient × activation|**. Filters with low scores contribute little to the loss and are safe to remove.

**Why Taylor over magnitude pruning:** Magnitude pruning removes small-weight filters regardless of their gradient. A small-weight filter with large gradient has high leverage on the loss and should not be pruned. Taylor scoring uses task-specific gradient information, making importance measurement loss-aware.

**Scoring objective:** Taylor scores are accumulated against the **KD distillation loss** (MSE between student and TinyViT-21M teacher embeddings), not the AdaFace classification loss. This means channels are scored by their importance to replicating the teacher's representation — directly matching the training objective.

### 8.3 Pruning Configuration

| Parameter | Value |
|---|---|
| Pruning ratio | **0.45** (45% of channels removed globally) |
| Importance metric | Taylor (first-order, `tp.importance.TaylorImportance`) |
| Pruner | `tp.pruner.MagnitudePruner` |
| Scoring batches | 100 batches from the training set |
| Global pruning | False (per-layer ratio — safer for hybrid window-attention architectures) |
| Ignored layers | projection head (`nn.Linear(448→512)` + `BatchNorm1d`) |
| Ignored module types | All attention modules (identified by name containing `'attn'` or class name containing `'attention'`) |

**Why attention modules are excluded:** TinyViT's window self-attention blocks have non-standard internal tensor shapes (query/key/value projections with head-dimension indexing). Structured pruning of attention heads requires special handling that `torch-pruning`'s generic pruner does not support for this architecture, and pruning them naively causes runtime indexing errors. Excluding them restricts pruning to the MLP blocks and projection layers, which contain the majority of parameters.

**Why the projection head is excluded:** The projection head (`Linear(448→512) + BN1d(512)`) produces the 512-dim embedding that must remain compatible with the AdaFace head's weight matrix (`out_features × 512`). Pruning its output dimension would break this interface.

### 8.4 Pruning Procedure

```
1. Deep-copy the Stage 3 TinyViT-11M student (preserves original for baseline comparison)
2. Run 100 training batches: forward → KD loss (MSE vs TinyViT-21M teacher) → backward
   [accumulates per-channel Taylor importance scores]
3. pruner.step() — applies pruning masks, physically removes channels from all affected layers
4. Zero out residual gradients
5. [Architecture is now modified — fewer channels throughout MLP blocks]
```

**Parameter count change:**
- Before pruning: ~11M parameters
- After pruning (45% ratio): ~6M parameters
- Reduction: ~45% fewer parameters, proportional reduction in FLOPs for affected layers

### 8.5 Post-Pruning Architecture (Pruned TinyViT-11M)

After 45% channel pruning of MLP blocks, the model retains the following structure:

```
Pruned-TinyVITStudentModel
├── backbone: tiny_vit_11m_224 (architecture modified by pruning)
│   ├── patch_embed: Conv2d stem (unchanged — not pruned)
│   ├── layers[0..3]: 4 hierarchical stages
│   │   ├── blocks: window attention blocks
│   │   │   ├── attn: LocalWindowAttention (IGNORED — not pruned, original dims preserved)
│   │   │   └── mlp: MLP blocks (PRUNED — hidden dim reduced by ~45%)
│   │   │       ├── fc1: Linear(in, hidden_pruned)   ← hidden_pruned ≈ 0.55 × original
│   │   │       ├── act: GELU
│   │   │       └── fc2: Linear(hidden_pruned, out)
│   │   └── downsample: patch merging layers (may be partially pruned)
│   └── norm: LayerNorm (output dim preserved)
│   └── output dim: 448 (backbone feature dim — preserved by head exclusion)
└── projection: nn.Sequential (EXCLUDED FROM PRUNING — original dims)
    ├── nn.Linear(448, 512)
    └── nn.BatchNorm1d(512)

Output: 512-dim L2-normalised embedding
```

**Key architectural constraint:** Because the projection head is excluded from pruning, the 448-dim backbone→512-dim embedding interface is preserved. The pruning only affects the internal MLP hidden dimensions within the backbone's transformer blocks.

### 8.6 Re-distillation Fine-tuning

After pruning modifies the architecture, accuracy recovery is performed via re-distillation against the frozen TinyViT-21M teacher.

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Batch size | 512 |
| Backbone LR | 5e-5 (conservative — pruned arch is fragile) |
| Projection LR | 2e-4 |
| AdaFace head LR | 2e-4 |
| Weight decay | 1e-4 |
| Optimiser | AdamW (fused=True) |
| LR schedule | CosineAnnealingLR (T_max=30, η_min=1e-6) |
| AMP dtype | bfloat16 (no GradScaler) |
| Alpha KD | **0.70** (fixed — 70% distillation, 30% classification) |
| Gradient clipping | max norm 1.0 |
| Checkpoint metric | VER@0.1% FAR |

**Alpha fixed at 0.70:** Unlike the dynamic schedule in Stages 2–3, re-distillation uses a fixed alpha of 0.70. After pruning, the model's internal representations are disrupted — keeping a high, stable distillation weight ensures the pruned model is anchored to the teacher's embedding space throughout recovery rather than drifting toward noisy classification gradients from a freshly initialised head.

**Fresh AdaFace head:** A new AdaFace head is initialised for re-distillation. The pruning process invalidates the previous head's compatibility (channel dimension changes upstream of the projection could affect gradient flow), and starting fresh with a lower LR allows the head to stabilise alongside backbone recovery.

**torch.compile applied after pruning:** `torch.compile` is applied to the pruned model before re-distillation (not before pruning). Applying it before would compile the graph to the pre-pruning architecture — after `pruner.step()` changes tensor shapes, the compiled graph becomes invalid. Compiling post-pruning ensures the fused CUDA graph matches the new architecture.

### 8.7 Problems Encountered During Pruning

**Architecture reload after pruning:** Structured pruning changes the number of channels in affected layers (e.g., a Linear layer with 512 hidden units may become 281 after 45% pruning). The original `TinyVITStudentModel()` Python class definition still instantiates full-width layers. Loading the pruned state_dict into the original class fails with a shape mismatch. The full model object (architecture + weights) was saved via `torch.save(model, ...)` rather than `torch.save(model.state_dict(), ...)` to preserve the pruned architecture.

**torch.compile state_dict prefix:** Checkpoints saved from `torch.compile`-wrapped models contain an `_orig_mod.` prefix on all state_dict keys (e.g., `_orig_mod.backbone.layers.0.blocks.0.attn.proj.weight`). Loading these into a non-compiled model fails. A key-renaming pass strips the prefix: `{k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}`.

**GradScaler + bfloat16 incompatibility:** An early attempt to use GradScaler during bfloat16 re-distillation produced NaN gradients. BFloat16 has the same exponent range as float32 and does not experience the gradient underflow that GradScaler is designed to prevent. Applying GradScaler with bfloat16 over-scales the loss by (e.g.) 2¹⁶, causing weight divergence to NaN, which then corrupted BatchNorm running statistics and caused DataLoader workers to segfault at the next epoch reset. Re-distillation was run without GradScaler.

---

## 9. Evaluation Metrics

All evaluations use pairwise verification on the provided `pairs_val.csv` and `pairs_test.csv` files. Similarity is computed as cosine similarity between L2-normalised embeddings.

| Metric | Definition |
|---|---|
| **VER@0.1% FAR** | True accept rate at 0.1% false accept rate (primary checkpoint metric at all stages) |
| **VER@1% FAR** | True accept rate at 1.0% false accept rate |
| **EER** | Equal Error Rate — point where false accept rate equals false reject rate; computed via linear interpolation between adjacent ROC curve points for precision |
| **AUC** | Area under the ROC curve |

### Final Test Set Results (Pruned TinyViT-11M)

| Metric | Value |
|---|---|
| VER @ 0.1% FAR | **50.6%** |
| VER @ 1.0% FAR | — |
| EER | **~8%** |
| AUC | **~96.8%** |
| Parameters | ~6M (45% reduction from 11M baseline) |

---

## 10. Ensemble & Post-Processing

No ensemble or post-processing methods were used. A single pruned model produces embeddings directly compared via cosine similarity.

---

## 11. Summary of Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Teacher backbone | SwinV2-Base | Local window attention preserves fine-grained ear anatomy better than global ViT attention |
| Student 1 | TinyViT-21M | Architecturally similar to SwinV2 (window attention) — enables effective KD transfer |
| Student 2 | TinyViT-11M | Natural next step in the TinyViT family; shares architecture with 21M, enabling effective second-hop distillation |
| Pruning method | Taylor (first-order) | Task-aware importance via gradient×weight; superior to magnitude pruning which ignores gradient information |
| Pruning ratio | 0.45 | Best balance of compression and accuracy recovery after re-distillation |
| Loss function | AdaFace | Quality-adaptive margin handles high variance in ear image quality |
| Image format | WebP lossless | Smaller files, faster I/O, zero compression artefacts vs JPEG |
| Augmentation | CLAHE + sharpening | Enhances local contrast and edge visibility in anatomically complex ear structures |
| AMP on A100 | bfloat16, no GradScaler | Native Ampere support; bfloat16 eliminates float16's gradient underflow risk |
| Checkpoint metric | VER@0.1% FAR | Primary competition metric; AUC can be high even with poor performance at low FAR operating points |

---

## References

[1] Kim, M., Jain, A. K., & Liu, X. (2022). AdaFace: Quality Adaptive Margin for Face Recognition. *CVPR 2022*.

[2] Liu, Z., et al. (2022). Swin Transformer V2: Scaling Up Capacity and Resolution. *CVPR 2022*.

[3] Wu, K., et al. (2022). TinyViT: Fast Pretraining Distillation for Small Vision Transformers. *ECCV 2022*.

[4] Fang, G., et al. (2023). DepGraph: Towards Any Structural Pruning. *CVPR 2023*. [torch-pruning library]

[5] Ž. Emeršič et al., "The Unconstrained Ear Recognition Challenge 2023," *IJCB 2023*. [UERC 2023 summary paper]
