# UERC26 Ear Recognition — Full Pipeline Methodology

> **Task:** Ear verification (1:1 biometric matching) on the UERC26 dataset, targeting fast inference on edge devices (Raspberry Pi).
> **Strategy:** Train a high-accuracy teacher (SwinV2-Base + AdaFace), then compress knowledge into a deployable TinyViT-5M student via a two-stage distillation chain.
> **No external data was used at any stage.**

---

## Table of Contents

1. [Dataset](#1-dataset)
2. [Preprocessing Pipeline](#2-preprocessing-pipeline)
3. [Stage 1 — Teacher Training: SwinV2-Base + AdaFace](#3-stage-1--teacher-training-swinv2-base--adaface)
4. [Stage 2 — Distillation: SwinV2-Base → TinyViT-21M](#4-stage-2--distillation-swinv2-base--tinyvit-21m)
5. [Stage 3 — Distillation: TinyViT-21M → TinyViT-5M](#5-stage-3--distillation-tinyvit-21m--tinyvit-5m)
6. [Final Model Architecture (TinyViT-5M) In Depth](#6-final-model-architecture-tinyvit-5m-in-depth)
7. [Evaluation Protocol](#7-evaluation-protocol)
8. [Final Results](#8-final-results)
9. [Problems Encountered and Solutions](#9-problems-encountered-and-solutions)
10. [Design Decisions and Rationale](#10-design-decisions-and-rationale)

---

## 1. Dataset

| Property | Value |
|---|---|
| Dataset | UERC26 (Unconstrained Ear Recognition Challenge 2026) |
| Training images | ~180,000 |
| Training subjects (identities) | 1,310 |
| Image format (original) | JPEG / PNG |
| Image format (used for training) | WebP lossless |
| Input resolution (all stages) | 224 × 224 |
| External data used | None |
| Splits | `train`, `val` (pairwise), `test` (pairwise), `sequestered` |
| Split file | `dataset_split.csv` (tab-separated: path, split) |
| Pairs file | `pairs_val.csv`, `pairs_test.csv` (tab-separated: img1, img2, label) |

Labels are derived from the subject directory name — each subdirectory corresponds to one identity. Subject-to-integer mappings are built from the sorted set of subject directory names found in the training split.

---

## 2. Preprocessing Pipeline

All preprocessing was done **before training** (offline) and is consistent across all three stages.

### 2.1 Resize

All images were resized **locally** (offline, before any format conversion) to **224 × 224** using bilinear interpolation. Bilinear was chosen over nearest-neighbour (introduces blocky staircase artefacts) and bicubic (can cause ringing at strong edges due to Gibbs phenomenon). The `align_corners=False` convention was used throughout, consistent with standard image-processing semantics.

### 2.2 Format Conversion: PNG/JPEG → WebP Lossless

After resizing, all images were converted to **WebP lossless** format. Reasons:

- WebP lossless is typically 25–35% smaller than equivalent PNG with identical pixel data, reducing disk I/O time per batch.
- Faster decode than PNG in modern `libwebp`.
- Unlike JPEG, lossless WebP introduces zero compression artefacts. JPEG blocking and ringing around edges can be learned by the model as identity-correlated features — a serious problem in biometrics.
- Dataset paths in the CSV were rewritten at load time: `row[0].rsplit('.', 1)[0] + '.webp'`, so no CSV modification was needed.

### 2.3 On-the-fly Augmentations (Training Only)

Applied inside `Dataset.__getitem__`, running in parallel DataLoader worker processes on CPU:

| Augmentation | Probability | Details |
|---|---|---|
| CLAHE | p = 0.5 | Applied in LAB colour space on the L channel only. `clipLimit=2.0`, `tileGridSize=(8,8)`. Enhances local contrast tile-by-tile, prevents global overexposure. |
| Unsharp Masking (Sharpen) | p = 0.3 | `sharpened = clip(original × 1.5 − GaussianBlur(σ=1.0) × 0.5, 0, 255)`. Lower probability than CLAHE to avoid ringing artefacts from over-sharpening. |

**CLAHE rationale:** Standard histogram equalisation stretches global contrast and can blow out local regions. CLAHE divides the image into non-overlapping tiles, equalises each tile's histogram independently, clips at `clipLimit` to prevent noise amplification, and blends at tile boundaries via bilinear interpolation. Applied in LAB space so only luminance is equalised, preserving hue and saturation.

**Sobel filtering was explicitly rejected:** An earlier experiment added Sobel edge-map preprocessing. It was removed because (a) SwinV2's early convolutional stem already learns Sobel/Gabor-like edge detectors tuned to the task, (b) applying a fixed Sobel kernel discards colour and texture information that the model could have used, (c) it is not differentiable in the preprocessing stage so no gradient flows through it, and (d) it added 20–50 ms of CPU time per image at scale.

### 2.4 GPU-side Augmentations (Training Only)

Applied on the GPU after CPU→GPU transfer, using `torchvision.transforms.v2`:

```
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
RandomGrayscale(p=0.2)
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  ← ImageNet stats
```

### 2.5 Evaluation Preprocessing

No augmentation. Only:
```
ToTensor → Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Images are already 224 × 224 so no resize is needed at eval time.

### 2.6 DataLoader Configuration

| Setting | Value | Reason |
|---|---|---|
| `num_workers` | 8 (Stage 1), 12 (Stages 2 & 3) | Parallel CPU preprocessing |
| `pin_memory=True` | All stages | DataLoader allocates output tensors in page-locked memory; enables direct DMA to GPU without intermediate CPU copy |
| `persistent_workers=True` | All stages | Keeps worker processes alive between epochs; avoids fork+reimport overhead at each epoch boundary |
| `prefetch_factor` | 4 | Workers prefetch 4 batches ahead; GPU is never starved |
| `drop_last=True` | Training only | Prevents a partial final batch from destabilising BatchNorm statistics |
| `.to(device, non_blocking=True)` | All stages | CPU→GPU transfer is async; Python thread returns immediately and overlaps with next DataLoader prefetch |

**Pipeline timing block:** A profiling block was run over the first 20 batches of epoch 1, measuring DataLoader load time and GPU transfer time separately (with `torch.cuda.synchronize()` before measuring transfer, since GPU ops are async). This identified whether the bottleneck was disk/CPU or PCIe, guiding optimisation decisions.

---

## 3. Stage 1 — Teacher Training: SwinV2-Base + AdaFace

**File:** `uerc26_Model_training__single_phase_.ipynb`

### 3.1 Goal

Train a high-accuracy ear verification backbone with no model size constraint. This model acts as the root "super-teacher" for the downstream distillation chain.

### 3.2 Why SwinV2 Over ViT / DINO

Ear recognition depends on fine local anatomical structures — the helix, antihelix, tragus, concha, and their topological relationships. Standard Vision Transformers (ViT, DINO) use global self-attention where every patch attends to every other patch (O(n²) complexity). Global attention is powerful for scene-level semantics but dilutes fine-grained local texture: local discriminative detail gets averaged across the full attention matrix.

SwinV2 addresses this with two key properties:

**Shifted local window attention:** Instead of attending globally, each patch attends only within a fixed local window (e.g. 7 × 7 patches). This is O(n) in the number of patches. Windows are shifted between consecutive layers so information propagates globally but hierarchically, preserving local texture at each layer.

**Hierarchical feature pyramid:** Like a CNN, SwinV2 builds a multi-scale feature hierarchy. Early stages operate at high spatial resolution and capture low-level texture; deeper stages downsample and capture high-level semantic shape. Ear recognition needs both: fine texture (skin folds, ridges) and global shape (overall ear geometry).

**Window stitching (SimMIM compatibility):** SwinV2 can be pretrained at one resolution and fine-tuned at another by stitching window-relative position biases. This matters for ear images which may have been captured at variable distances.

### 3.3 Model Architecture

```
SwinV2EarModel
│
├── backbone: SwinV2-Base (torchvision, pretrained on ImageNet-22K → 1K)
│   ├── PatchPartition: 4×4 patches → 96-dim tokens
│   ├── Stage 1: 2 Swin blocks, window 8×8, dim 128
│   ├── Stage 2: 2 Swin blocks, window 8×8, dim 256
│   ├── Stage 3: 18 Swin blocks, window 8×8, dim 512
│   └── Stage 4: 2 Swin blocks, window 8×8, dim 1024
│       └── Global average pool → 1024-dim feature vector
│   head: replaced with nn.Identity()
│
├── projection: Sequential(
│   ├── nn.Linear(1024, 512)
│   └── nn.BatchNorm1d(512)
│   )
│
└── forward():
    ├── Bilinear interpolate input 224×224 → 256×256  [SwinV2-B native resolution]
    ├── backbone(x) → 1024-dim features
    ├── projection → 512-dim embedding
    ├── L2-normalise embedding
    └── return (embedding, norm) or embedding
```

**Total teacher parameters:** ~88M (backbone) + ~0.5M (projection) ≈ **88.5M**

**Embedding dimension:** 512 (L2-normalised)

**Input resolution to model:** 256 × 256 (dynamically upscaled inside `forward()` from the 224 × 224 DataLoader output via bilinear interpolation)

### 3.4 Loss: AdaFace

Standard angular margin losses (ArcFace) add a fixed margin `m` to the angle between the embedding and its class weight:

```
loss = CrossEntropy( s · cos(θ_yi + m),  s · cos(θ_j) )
```

AdaFace modifies this by making the margin adaptive based on estimated image quality. Image quality is proxied by the **L2 norm of the feature embedding**: high-norm embeddings correspond to high-quality, informative images; low-norm embeddings to blurry, occluded, or poorly lit images.

```
norm_i = ||embedding_i||_2

margin_scaler = clamp( h · (norm_i - batch_mean_norm) / batch_std_norm,  -1, 1 )

g_add = m · (1 + margin_scaler)

target_logit = cos(θ_yi + m) − g_add

output = s · [ cos(θ_j) · (1 − one_hot)  +  target_logit · one_hot ]

loss = CrossEntropy(output, labels)
```

`batch_mean_norm` and `batch_std_norm` are updated via exponential moving average each forward pass (`t_alpha=0.01`).

**Effect:** High-quality ears get a larger effective margin → tighter, more discriminative class boundaries. Low-quality ears get a reduced margin → the model is not penalised heavily for imprecise embeddings of inherently difficult images.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Scale `s` | 64.0 |
| Base margin `m` | 0.25 |
| Quality modulation `h` | 0.333 |
| EMA momentum `t_alpha` | 0.01 |
| Initial `batch_mean` | 20.0 |
| Initial `batch_std` | 100.0 |
| Weight init | Xavier uniform |

### 3.5 Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 15 |
| Batch size | 192 |
| Hardware | Google Colab L4 GPU |
| Precision | float16 AMP + GradScaler |
| Gradient clipping | `clip_grad_norm_` at 1.0 (model + head separately) |
| Backbone LR | 1e-4 |
| Projection LR | 5e-4 |
| AdaFace head LR | 5e-4 |
| Optimizer | AdamW, `weight_decay=1e-4` |
| LR schedule | CosineAnnealingLR, `T_max=15`, `eta_min=1e-6` |
| Warmup | None (single-phase, fully unfrozen from epoch 1) |
| Memory format | `torch.channels_last` (NHWC) |
| cuDNN benchmark | True |
| TF32 | True (matmul + cuDNN) |
| Eval interval | Every epoch |
| Checkpoint criterion | Best VER@0.1%FAR on validation pairs |

**Training strategy — single phase, fully unfrozen:** The backbone was not frozen at any point. All parameters, including the pretrained SwinV2-Base weights, were trained from epoch 1 with a healthy backbone LR of 1e-4. Earlier experiments with phase training (freeze backbone, train head first) were not used in the final run.

**GradScaler** was used with float16 to prevent gradient underflow. float16 has a limited dynamic range (~6×10⁻⁵ to 65504); small gradients in early layers can underflow to zero. GradScaler multiplies the loss by a large scale factor before `.backward()` so gradients remain representable, then unscales before the optimizer step and skips the step if inf/NaN is detected.

**Checkpoint saved on VER@0.1%FAR** (Verification Rate at 0.1% False Acceptance Rate) — not AUC. VER@0.1% is the operationally meaningful metric for a biometric system: it measures how many genuine pairs are correctly verified when the system is tuned to allow at most 1-in-1000 false acceptances.

---

## 4. Stage 2 — Distillation: SwinV2-Base → TinyViT-21M

**File:** `swin88_to_tiny21.ipynb`

### 4.1 Goal

Transfer the knowledge of the 88M-parameter SwinV2-Base teacher into a 21M-parameter TinyViT-21M student — a ~4× compression — while retaining as much verification performance as possible.

### 4.2 Why TinyViT as the Student Architecture

TinyViT was designed explicitly as a compressed alternative to Swin-style transformers. Its architecture mirrors Swin's hierarchical, window-based attention design:

- Both use a hierarchical multi-stage structure with patch merging between stages
- Both use local window attention rather than global attention
- Both share similar positional embedding approaches

This architectural alignment means the student's internal representations are structurally compatible with the teacher's. The student can genuinely absorb the teacher's feature geometry rather than being forced to approximate it through a fundamentally different computational pathway. This is the core reason TinyViT was selected over MobileNet, EfficientNet, or other efficient architectures.

### 4.3 Teacher Model (Frozen)

The fine-tuned SwinV2-Base checkpoint from Stage 1 was loaded with two adjustments:

1. **`_orig_mod.` prefix stripping:** `torch.compile()` modifies state_dict key names by prepending `_orig_mod.`. A remapping loop stripped this prefix before loading.
2. **`strict=False`:** Used to handle minor layer naming differences between torchvision versions (e.g. `features.*` vs `patch_embed.*`).

After loading, the teacher was compiled with `torch.compile()` for fast frozen inference and all parameters were frozen (`requires_grad=False`). The teacher runs in `eval()` mode throughout.

### 4.4 Student Model Architecture

```
TinyViT21MStudent
│
├── backbone: tiny_vit_21m_224 (timm, pretrained dist_in22k_ft_in1k)
│   ├── PatchEmbed: 4×4 conv stem → 64-dim tokens
│   ├── Stage 1: MBConv blocks (local CNN-style), dim 64
│   ├── Stage 2: MBConv blocks, dim 128
│   ├── Stage 3: TinyViT attention blocks (local window), dim 256, window 7×7
│   └── Stage 4: TinyViT attention blocks, dim 576, window 7×7
│       └── Global average pool → 576-dim feature vector
│
├── projection: Sequential(
│   ├── nn.Linear(576, 512)
│   └── nn.BatchNorm1d(512)
│   )
│
└── forward():
    ├── backbone(x) → 576-dim features
    ├── projection → 512-dim embedding
    ├── L2-normalise embedding
    └── return (embedding, norm) or embedding
```

**Student parameters:** ~21M (backbone) + ~0.3M (projection) ≈ **21.3M**

**Embedding dimension:** 512 (L2-normalised, identical to teacher — enabling direct MSE distillation)

**Input resolution:** 224 × 224 (student's native resolution)

### 4.5 Resolution Handling

The teacher (SwinV2-Base) requires 256 × 256. The student (TinyViT-21M) uses 224 × 224. Images are loaded at 224 × 224. Inside the training loop, teacher inputs are dynamically upscaled:

```python
images_teacher = F.interpolate(images, size=(256, 256), mode='bicubic', align_corners=False)
teacher_embeddings = teacher_model(images_teacher).detach()

student_embeddings, student_norms = student_model(images, return_norms=True)
# images remains 224×224 for the student
```

Bicubic was used for upscaling (higher quality than bilinear for super-resolution-style upscaling; ringing artefacts are less of a concern here since the teacher is robust to minor resizing effects).

### 4.6 Distillation Loss

The loss combines two terms:

```
loss = (1 − α) × loss_hard  +  α × loss_soft
```

**`loss_hard`** — AdaFace loss on the student's embeddings against ground-truth identity labels. The student still learns to discriminate identities directly, not just to mimic the teacher.

**`loss_soft`** — MSE loss between the student's 512-dim L2-normalised embedding and the teacher's 512-dim L2-normalised embedding. This pulls the student's embedding space into alignment with the teacher's, transferring the teacher's learned identity geometry.

```python
loss_hard = adaface_head(student_embeddings, student_norms, labels)
loss_soft  = F.mse_loss(student_embeddings, teacher_embeddings)
loss       = (1.0 - alpha) * loss_hard + alpha * loss_soft
```

**Dynamic alpha annealing:**

```
α(epoch) = α_start − (epoch−1)/(EPOCHS−1) × (α_start − α_end)

α_start = 0.90,   α_end = 0.20
```

At epoch 1, the loss is 90% soft (teacher imitation) and 10% hard (label supervision). By the final epoch it is 20% soft and 80% hard. This curriculum ensures the student first aligns its embedding space with the teacher's before progressively developing independent label-discriminating capability.

### 4.7 Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 40 |
| Batch size | 256 |
| Hardware | Google Colab A100 |
| Precision | bfloat16 AMP (no GradScaler) |
| Gradient clipping | `clip_grad_norm_` at 1.0 (student + head separately) |
| Student LR | 1e-4 |
| AdaFace head LR | 5e-4 |
| Optimizer | AdamW, `weight_decay=1e-4`, `fused=True` |
| LR schedule | LinearLR warmup (2 epochs, 0.1× → 1×) then CosineAnnealingLR (`T_max=38`, `eta_min=1e-6`) via SequentialLR |
| α_start | 0.90 |
| α_end | 0.20 |
| Early stopping patience | 8 epochs (on VER@0.1%) |
| Memory format | `torch.channels_last` |
| torch.compile | Student and teacher both compiled |
| Eval interval | Every epoch |
| Checkpoint criterion | Best VER@0.1%FAR on validation pairs |

**bfloat16 instead of float16:** On A100 (Ampere architecture), bfloat16 is the correct choice. bfloat16 has the same 8-bit exponent as float32, giving it the same dynamic range. Gradients do not underflow, so **GradScaler is not used and must not be used** — using GradScaler with bfloat16 overscales gradients by the scale factor (typically 2^16), producing NaN weights and eventual DataLoader worker segfaults.

**`fused=True` on AdamW:** Uses a custom CUDA kernel that fuses the optimizer's weight update into a single GPU operation, significantly faster than the standard loop-based implementation on A100.

**Warmup rationale:** Starting at full LR from epoch 1 with the dynamic alpha weighting and a new student model (randomly initialised head) would cause large early gradient steps. Two epochs of linear warmup from 0.1× LR lets the student find a stable initialisation before full-speed training.

---

## 5. Stage 3 — Distillation: TinyViT-21M → TinyViT-5M

**File:** `5mtinyvit.ipynb`

### 5.1 Goal

Compress the 21M-parameter student from Stage 2 into a ~5M-parameter TinyViT-5M — the final deployable model targeting Raspberry Pi inference.

### 5.2 Why Two-Hop Distillation (Swin → 21M → 5M)

The parameter gap between SwinV2-Base (88M) and TinyViT-5M (5M) is approximately 17×. Direct distillation across this gap is significantly harder because:

- The 5M student has far less representational capacity to absorb the teacher's full embedding space geometry.
- The MSE soft loss on a 512-dim embedding becomes an extremely constrained optimisation problem when the student backbone is so much weaker.

The TinyViT-21M model from Stage 2 — having already been distilled from SwinV2 — acts as a **capacity bridge**: it knows what SwinV2 knows but at a scale much closer to 5M parameters. Its embedding space is a distilled, compressed version of SwinV2's, making it a more learnable target for the 5M student than the original 88M teacher.

### 5.3 Teacher Model (Frozen)

The best TinyViT-21M checkpoint from Stage 2 (`best_student_tinyvit21m.pt`) was loaded directly with standard `load_state_dict` (no key remapping needed — this checkpoint was saved cleanly without `torch.compile` artifacts). Compiled with `torch.compile()` for speed, fully frozen.

### 5.4 Student Model Architecture (Final Deployed Model)

```
TinyViT5MStudent
│
├── backbone: tiny_vit_5m_224 (timm, pretrained dist_in22k_ft_in1k)
│   ├── PatchEmbed: 4×4 conv stem → 64-dim tokens
│   ├── Stage 1: MBConv blocks (local CNN-style), dim 64
│   ├── Stage 2: MBConv blocks, dim 128
│   ├── Stage 3: TinyViT attention blocks, dim 160, window 7×7
│   └── Stage 4: TinyViT attention blocks, dim 320, window 7×7
│       └── Global average pool → 320-dim feature vector
│
├── projection: Sequential(
│   ├── nn.Linear(320, 512)
│   └── nn.BatchNorm1d(512)
│   )
│
└── forward():
    ├── backbone(x) → 320-dim features
    ├── projection → 512-dim embedding
    ├── L2-normalise embedding
    └── return (embedding, norm) or embedding
```

**Student parameters:** ~5M (backbone) + ~0.16M (projection) ≈ **5.16M**

**Backbone output dimension:** 320 (vs 576 for TinyViT-21M, vs 1024 for SwinV2-Base)

**Final embedding dimension:** 512 (L2-normalised) — identical across all three stages, enabling consistent cosine similarity evaluation.

**Input resolution:** 224 × 224

### 5.5 Key Difference from Stage 2: No Resolution Mismatch

Since both teacher (TinyViT-21M) and student (TinyViT-5M) use 224 × 224 natively, there is no interpolation inside the training loop. Both receive the same `images` tensor:

```python
teacher_embeddings = teacher_model(images).detach()
student_embeddings, student_norms = student_model(images, return_norms=True)
```

### 5.6 Distillation Loss

Identical formulation to Stage 2:

```
loss = (1 − α) × loss_hard  +  α × loss_soft
```

Same MSE soft loss + AdaFace hard loss. Same dynamic alpha annealing (0.90 → 0.20).

### 5.7 Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 60 |
| Batch size | 512 |
| Hardware | Google Colab A100 |
| Precision | bfloat16 AMP (no GradScaler) |
| Gradient clipping | `clip_grad_norm_` at 1.0 (student + head separately) |
| Student LR | 1e-4 |
| AdaFace head LR | 5e-4 |
| Optimizer | AdamW, `weight_decay=1e-4`, `fused=True` |
| LR schedule | LinearLR warmup (2 epochs) + CosineAnnealingLR (`T_max=58`, `eta_min=1e-6`) |
| α_start | 0.90 |
| α_end | 0.20 |
| Early stopping | Not used in this stage |
| Eval interval | Every 2 epochs |
| Memory format | `torch.channels_last` |
| torch.compile | Student and teacher both compiled |
| Checkpoint criterion | Best VER@0.1%FAR on validation pairs |

**Batch size increased to 512:** The 5M student is much smaller than the 21M model; its forward pass consumes substantially less VRAM, allowing a larger batch. Larger batches improve AdaFace's batch normalisation of norms (more samples = better EMA of `batch_mean` and `batch_std`) and reduce gradient noise.

**60 epochs (vs 40):** The smaller student has lower capacity and converges more slowly from the teacher signal. More epochs allow it to fully absorb what the 21M teacher has to offer.

---

## 6. Final Model Architecture (TinyViT-5M) In Depth

This section describes the complete forward-pass architecture of the deployed model.

### 6.1 Patch Embedding

Input: `[B, 3, 224, 224]`

A convolutional stem with two sequential 3×3 convolutions (stride 2 each) maps the image to patch tokens:
- Conv2d(3, 64, 3×3, stride 2, padding 1) → [B, 64, 112, 112]
- Conv2d(64, 64, 3×3, stride 2, padding 1) → [B, 64, 56, 56]

Tokens are then laid out as sequence: [B, 56×56, 64] = [B, 3136, 64]

### 6.2 Stage 1 — MBConv Blocks (Resolution 56×56, dim 64)

Uses MBConv (Mobile Inverted Bottleneck Convolution) blocks — the same building block as EfficientNet. No attention. Structure per block:

```
LayerNorm → Linear(expand) → GELU → DepthwiseConv 3×3 → Squeeze-Excite → Linear(project) → Dropout → residual
```

Operates at high spatial resolution, capturing low-level texture and local edge patterns. Output: [B, 3136, 64]

Patch merging at end: 2×2 spatial downsampling → [B, 784, 128]

### 6.3 Stage 2 — MBConv Blocks (Resolution 28×28, dim 128)

Same MBConv structure. Higher channel dimension captures richer mid-level features (curved contours, structural ridges). Output: [B, 784, 128]

Patch merging: → [B, 196, 160]

### 6.4 Stage 3 — TinyViT Attention Blocks (Resolution 14×14, dim 160, window 7×7)

Switches from MBConv to transformer attention. TinyViT's attention is a linear approximation:

```
Q = X · W_Q     (full sequence)
K = pool(X) · W_K   (key computed on pooled tokens — reduces key complexity)
V = pool(X) · W_V

Attention = softmax(Q · K^T / √d_k) · V
```

Keys and values are computed on a pooled (downsampled) version of the token sequence, reducing attention complexity from O(n²) to O(n·√n). Queries remain full-resolution.

Within a window of 7×7 = 49 tokens at 14×14 spatial resolution (4 windows), this is very efficient.

**Relative position biases** are learned per window, providing spatial structure without absolute position encodings.

Output: [B, 196, 160]

Patch merging: → [B, 49, 320]

### 6.5 Stage 4 — TinyViT Attention Blocks (Resolution 7×7, dim 320, window 7×7)

At 7×7 spatial resolution, there is exactly one window containing all 49 tokens. The attention in this stage effectively becomes global (all tokens attend to each other within the single window). This is where high-level semantic identity features are assembled.

Output: [B, 49, 320]

### 6.6 Global Average Pooling

```
x = mean(x, dim=1)   # [B, 49, 320] → [B, 320]
```

### 6.7 Projection Head

```
nn.Linear(320, 512)
nn.BatchNorm1d(512)
```

BatchNorm here serves two roles: it normalises the pre-embedding distribution for training stability, and its learned scale/shift parameters act as a calibration layer between the backbone's feature space and the AdaFace loss's expected input range.

### 6.8 L2 Normalisation

```
norms = ||embedding||_2
embedding = embedding / norms
```

The output embedding lies on the unit hypersphere. Cosine similarity between two embeddings is then simply their dot product:

```
similarity = embedding_1 · embedding_2   (since both are unit vectors)
```

### 6.9 Final Output

A 512-dimensional L2-normalised vector on the unit hypersphere. Two ear images are matched by computing cosine similarity between their embeddings and comparing against a threshold tuned on the validation set.

### 6.10 Summary Table

| Component | Output Shape | Notes |
|---|---|---|
| Input | [B, 3, 224, 224] | WebP, ImageNet-normalised |
| PatchEmbed | [B, 3136, 64] | 2× conv stem |
| Stage 1 (MBConv) | [B, 3136, 64] | Local CNN blocks |
| Patch Merge 1 | [B, 784, 128] | 2× spatial downsample |
| Stage 2 (MBConv) | [B, 784, 128] | Local CNN blocks |
| Patch Merge 2 | [B, 196, 160] | 2× spatial downsample |
| Stage 3 (Attention) | [B, 196, 160] | Linear attn, window 7×7 |
| Patch Merge 3 | [B, 49, 320] | 2× spatial downsample |
| Stage 4 (Attention) | [B, 49, 320] | Global (1 window) |
| Global Avg Pool | [B, 320] | |
| Linear(320→512) | [B, 512] | |
| BatchNorm1d(512) | [B, 512] | |
| L2 Normalise | [B, 512] | Unit hypersphere |

---

## 7. Evaluation Protocol

### 7.1 Task: Pairwise Verification

Each evaluation sample is a pair of ear images `(img1, img2)` with a binary label: `1` (same identity) or `0` (different identity). The model produces cosine similarity scores; the ROC curve is swept over all possible thresholds.

### 7.2 Metrics

**VER@0.1%FAR (Verification Rate at 0.1% False Acceptance Rate):** The primary metric. At the operating point where 1 in 1000 impostors are incorrectly accepted, what fraction of genuine pairs are correctly verified? Operationally meaningful for deployment — a biometric system must hold a strict false acceptance ceiling.

**VER@1%FAR:** Same metric at a more relaxed 1% false acceptance ceiling.

**EER (Equal Error Rate):** The error rate at the threshold where False Acceptance Rate = False Rejection Rate. Lower is better. A threshold-independent summary metric.

**AUC (Area Under the ROC Curve):** Overall ranking quality of the similarity scores. Higher is better.

### 7.3 EER Computation

Linear interpolation between the two ROC curve points straddling the FPR = FNR crossing:

```python
t = (fnr[idx] - fpr[idx]) / ((fpr[idx+1] - fpr[idx]) + (fnr[idx] - fnr[idx+1]))
eer = fpr[idx] + t * (fpr[idx+1] - fpr[idx])
```

This avoids the quantisation error of simply reading the closest point.

### 7.4 Checkpoint Criterion

**All three stages saved checkpoints on best VER@0.1%FAR**, not AUC. AUC measures global ranking quality but is insensitive to the operating region that matters for deployment (very low FAR). VER@0.1% directly measures performance at the strictest operating point.

---

## 8. Final Results

Results on the held-out **test set** (`pairs_test.csv`) using the final TinyViT-5M model:

| Metric | Value |
|---|---|
| VER @ 0.1% FAR | **44.09%** |
| EER | **10.60%** |
| AUC | **95.83%** |

The high AUC (95.83%) indicates the model's cosine similarity scores rank genuine pairs far above impostor pairs across the full ROC curve. The VER@0.1% (44.09%) reflects the difficulty of ear recognition at very strict operating points — unconstrained ear images have high intra-class variance from pose, lighting, and occlusion (hair). The EER of 10.60% places the model's balanced operating point at approximately 89.4% accuracy.

---

## 9. Problems Encountered and Solutions

### 9.1 CPU Bottleneck — GPU Starvation

**Problem:** Training was slow not because the GPU was underperforming, but because the CPU DataLoader pipeline could not supply batches fast enough. The GPU finished a batch in ~50 ms; the CPU took ~300 ms to prepare the next one. GPU utilisation dropped to ~15%.

**Root causes:** `cv2.imread` disk I/O for thousands of small files; PIL↔cv2 format conversions; CLAHE computation; on-the-fly augmentations — all sequential and CPU-bound.

**Solution:** Moved to WebP (faster decode), used `persistent_workers=True` (no worker respawn overhead), `pin_memory=True` (no intermediate CPU copy before GPU transfer), `prefetch_factor=4` (workers stay ahead of the GPU), and `non_blocking=True` on `.to(device)`. A profiling timing block at epoch 1 was used to measure whether the bottleneck was data loading or GPU transfer, guiding which knobs to turn.

### 9.2 AMP Precision Mismatch — GradScaler with bfloat16

**Problem:** The Stage 1 training used float16 + GradScaler (correct). When moving to A100 for Stages 2 and 3 and switching to bfloat16, GradScaler was initially left in. This caused NaN gradients, which propagated into BatchNorm, corrupted worker memory, and triggered DataLoader worker **segfaults** on the next epoch reset.

**Root cause:** GradScaler exists because float16 gradients underflow to zero (float16 minimum representable value: ~6×10⁻⁵). It scales the loss by a large factor (e.g. 2^16) before `.backward()`. bfloat16 has the same 8-bit exponent as float32 — its gradients do not underflow. Applying GradScaler anyway overscales gradients 65536×, producing NaN weights.

**Solution:** Removed GradScaler entirely for bfloat16 training. `torch.amp.autocast('cuda', dtype=torch.bfloat16)` is used; `.backward()` is called directly on the loss.

### 9.3 torch.compile State Dict Key Corruption

**Problem:** The Stage 1 checkpoint was saved after training with `torch.compile()`. When loaded in Stage 2, `model.load_state_dict()` failed or produced shape mismatches because `torch.compile` prepends `_orig_mod.` to all state_dict keys.

**Solution:** A remapping loop stripped the prefix before loading:
```python
new_state_dict = {}
for k, v in ckpt['model_state_dict'].items():
    name = k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict, strict=False)
```
`strict=False` additionally handled minor layer naming differences between timm versions.

### 9.4 Image Size Mismatch Between Teacher and Student

**Problem:** SwinV2-Base requires 256×256 input; TinyViT-21M and TinyViT-5M use 224×224. All DataLoader images are 224×224. Feeding 224×224 directly to SwinV2-Base during the Stage 2 distillation loop produced silent accuracy degradation (wrong positional bias interpolation inside the backbone) before it was caught.

**Solution:** Dynamic per-batch upscaling inside the training loop using `F.interpolate(..., size=(256,256), mode='bicubic')` for the teacher pass only, with the student receiving the original 224×224 tensors. In Stage 3 there is no size mismatch — both teacher and student use 224×224.

### 9.5 Double Normalisation — Silent Pipeline Bug

**Problem:** An inference pipeline applied ImageNet normalisation before passing images to a model whose `forward()` method also applied normalisation. No exception was raised; the model simply received inputs from a completely different distribution and produced confidently wrong embeddings.

**Solution:** All preprocessing was encapsulated inside the model's `forward()` method or a clearly named `preprocess()` wrapper. No external caller applies normalisation separately. This makes the model self-contained and immune to double-application by any pipeline.

### 9.6 Sobel Filter — Added Overhead, No Benefit

**Problem:** Sobel edge filtering was added as a preprocessing step under the hypothesis that ear structure is edge-defined. It added 20–50 ms of CPU time per image and degraded model performance.

**Root cause:** Modern backbones (SwinV2, TinyViT) learn their own edge detectors in early layers — Gabor-like and Sobel-like filters emerge naturally. Hand-crafted Sobel preprocessing: (a) discards colour and texture information the model could have used, (b) duplicates what the first convolutional stage already learns but with a fixed, non-task-adapted kernel, (c) is not differentiable so no gradient signal flows through it.

**Solution:** Removed entirely. The model learns better edge representations end-to-end.

### 9.7 DataLoader Worker Spawn Overhead

**Problem:** Between epochs, PyTorch's DataLoader was killing all worker processes and respawning them. Each spawn involves forking the process, reimporting all Python modules, and re-initialising the Dataset object — adding several seconds of dead time per epoch.

**Solution:** `persistent_workers=True` — workers stay alive between epochs and simply wait for the next iterator to be created.

### 9.8 PNG/JPEG Compression Artefacts Corrupting Learned Features

**Problem:** JPEG compression introduces blocking artefacts and ringing at edges. In biometric contexts, the model can inadvertently learn compression artefact patterns as identity-correlated features, reducing generalisation.

**Solution:** All images converted to WebP lossless before training. Lossless WebP is identical at the pixel level to the original, while being 25–35% smaller than PNG for faster I/O.

---

## 10. Design Decisions and Rationale

### 10.1 Why AdaFace Over ArcFace

Ear images have extreme quality variance — blurry, occluded by hair, severe pose variation. ArcFace uses a fixed angular margin; it penalises low-quality image embeddings equally to high-quality ones, causing the model to learn noisy gradients from inherently difficult samples. AdaFace's norm-based quality estimation provides a principled way to downweight bad samples automatically without manual sample weighting.

### 10.2 Why MSE Loss for Soft Distillation

The embeddings are L2-normalised 512-dim vectors on the unit hypersphere. MSE between two normalised vectors is monotonically related to their cosine distance: `MSE = 2 − 2·cos(θ)`. Minimising MSE directly minimises the angular distance in embedding space — exactly what is needed to align the student's embedding geometry with the teacher's. Cosine embedding loss would have been equivalent; MSE was used for simplicity.

### 10.3 Why Dynamic Alpha Annealing

Starting with α=0.90 (heavy teacher imitation) prevents the randomly initialised student from receiving large, noisy AdaFace gradients before it has established a reasonable embedding space. The student first converges onto the teacher's geometry, then progressively takes ownership of its own identity-discrimination capability as α decays toward 0.20. If α were fixed at 0.50 throughout, early training would be unstable and the student would not fully internalise the teacher's embedding structure.

### 10.4 Why Cosine Similarity for Verification

The embedding space is trained with angular margin losses (AdaFace/ArcFace), which explicitly optimise inter-class angular distance. The decision boundary is angular by construction. Cosine similarity measures the cosine of the angle between two L2-normalised embedding vectors — it is the natural distance metric for this space. Euclidean distance in the same space would be a monotonic function of cosine distance for unit vectors, so either would work, but cosine similarity is the canonical choice.

### 10.5 Why bfloat16 on A100 Over float16

A100 (Ampere architecture) natively accelerates bfloat16. bfloat16 retains float32's full dynamic range (8 exponent bits) while reducing storage and bandwidth (16-bit). This eliminates the need for GradScaler, simplifying the training loop and removing a class of numerical instability bugs. On Turing-architecture GPUs (T4, V100), bfloat16 is not hardware-accelerated; float16 + GradScaler is the correct choice there.

### 10.6 Why No Quantization or Pruning in the Final Pipeline

QAT (Quantization Aware Training) and Taylor pruning were investigated but not applied to the final pipeline for the following reasons:

- `torch.compile` (used for training speed) is fundamentally incompatible with QAT: compile fuses Conv+BN kernels, bypassing QAT's fake-quantization observers that need to fire after each individual operation.
- `channels_last` memory format (used for GPU efficiency) silently corrupts QAT observer calibration by presenting NHWC-strided tensors to observers that assume NCHW layout.
- `fused=True` AdamW bypasses the Python-level hooks that QAT observers rely on.
- The TinyViT-5M model at 5M parameters was already sufficiently compact for the intended deployment profile. Further compression at the cost of accuracy and training complexity was not warranted.

---

*Pipeline developed for UERC26 competition — 2026. No external data used at any stage.*
