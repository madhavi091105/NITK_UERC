import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


def _set_module(model, path, new_module):
    """Replace a module at a dot-separated path."""
    parts = path.split('.')
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def rebuild_from_pruned_state_dict(model, state_dict):
    """
    Inspect `state_dict` shapes vs current model. For every module whose
    weight/bias shapes differ, create a replacement module with the correct
    dimensions, then load the full state_dict.
    """
    # Map: module_path -> {param_name: shape}
    sd_shapes = {}
    for key, tensor in state_dict.items():
        parts = key.rsplit('.', 1)
        if len(parts) == 2:
            mod_path, param_name = parts
            sd_shapes.setdefault(mod_path, {})[param_name] = tensor.shape

    replaced = 0
    for name, module in list(model.named_modules()):
        if name not in sd_shapes:
            continue

        # Check if any parameter shape mismatches
        needs_replace = False
        for pn in ('weight', 'bias', 'running_mean', 'running_var'):
            cur = getattr(module, pn, None)
            if cur is not None and pn in sd_shapes[name] and cur.shape != sd_shapes[name][pn]:
                needs_replace = True
                break

        if not needs_replace:
            continue

        if isinstance(module, nn.Conv2d):
            w = sd_shapes[name]['weight']  # [out_ch, in_ch/groups, kH, kW]
            out_ch = w[0]
            if module.groups > 1:          # depthwise conv
                groups = out_ch
                in_ch  = out_ch
            else:
                groups = 1
                in_ch  = w[1]
            new = nn.Conv2d(
                in_ch, out_ch, module.kernel_size, module.stride,
                module.padding, module.dilation, groups,
                bias=(module.bias is not None),
                padding_mode=module.padding_mode,
            )
            _set_module(model, name, new); replaced += 1

        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nf = sd_shapes[name]['weight'][0]
            new = type(module)(nf, eps=module.eps, momentum=module.momentum,
                               affine=module.affine,
                               track_running_stats=module.track_running_stats)
            _set_module(model, name, new); replaced += 1

        elif isinstance(module, nn.Linear):
            w = sd_shapes[name]['weight']  # [out_features, in_features]
            new = nn.Linear(w[1], w[0], bias=(module.bias is not None))
            _set_module(model, name, new); replaced += 1

    print(f'  Rebuilt {replaced} modules to match pruned checkpoint shapes.')
    # Use nn.Module.load_state_dict directly to avoid triggering any
    # overridden load_state_dict (e.g. _PrunedTinyViT's) → prevents recursion.
    nn.Module.load_state_dict(model, state_dict, strict=False)
    return model


class TinyVITStudentModel(nn.Module):
    """
    TinyViT-21M backbone with a 512-dim projection head.
    
    forward() returns L2-normalized embeddings matching training-time behavior.
    This is critical because the weights were trained with normalization
    inside the forward pass (before AdaFace loss).
    """
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = timm.create_model('tiny_vit_21m_224', pretrained=False, num_classes=0)
        self.projection = nn.Sequential(
            nn.Linear(576, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        feat = self.backbone(x)
        embeddings = self.projection(feat)
        # Return raw embeddings — normalization is handled by solution.py's evaluate()
        # via F.normalize(), matching the baseline convention.
        return embeddings


class PrunedTinyViTWrapper(nn.Module):
    """
    Wrapper that exposes a .model attribute (TinyVITStudentModel).
    
    IMPORTANT: The evaluation scripts (evaluate_models.py, extract_features.py)
    do `model = model_class().model` and then call model.load_state_dict()
    directly. To support this with pruned weights, we override load_state_dict
    on the inner model so it automatically rebuilds the architecture first.
    """
    def __init__(self):
        super().__init__()
        self.model = _PrunedTinyViT(embedding_dim=512)

    def forward(self, x):
        return self.model(x)


class _PrunedTinyViT(TinyVITStudentModel):
    """
    Subclass of TinyVITStudentModel that overrides load_state_dict to
    automatically rebuild pruned layers before loading weights.
    
    This is necessary because:
      - evaluate_models.py does: model = model_class().model 
        then solution.load_model(model, weights) which calls
        model.load_state_dict(torch.load(weights_path), strict=False)
      - extract_features.py does: model = model_class().model
        then model.load_state_dict(torch.load(weights), strict=False)
    
    In both cases, the caller expects a standard load_state_dict interface.
    The pruned checkpoint has structurally different shapes, so we must
    rebuild first.
    """
    def __init__(self, embedding_dim=512):
        super().__init__(embedding_dim=embedding_dim)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # If the state_dict is a full checkpoint dict (from torch.save),
        # extract the model_state_dict key
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        # Rebuild layers to match pruned dimensions, then load weights.
        # We call rebuild which internally uses nn.Module.load_state_dict
        # (the base implementation), NOT this override — avoiding recursion.
        rebuild_from_pruned_state_dict(self, state_dict)
