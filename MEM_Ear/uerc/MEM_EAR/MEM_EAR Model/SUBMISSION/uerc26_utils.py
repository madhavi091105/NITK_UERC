import os
import random
import numpy as np
import torch
from itertools import combinations


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_pairs(df, n_genuine=5):
    genuine, impostor = [], []
    labels = df['label'].unique()

    for lbl in labels:
        paths = df[df['label'] == lbl]['path'].tolist()
        if len(paths) >= 2:
            combs = list(combinations(paths, 2))
            sampled = random.sample(combs, min(len(combs), n_genuine))
            genuine.extend([(p1, p2, 1) for p1, p2 in sampled])

    while len(impostor) < len(genuine):
        l1, l2 = random.sample(list(labels), 2)
        p1 = random.choice(df[df['label'] == l1]['path'].tolist())
        p2 = random.choice(df[df['label'] == l2]['path'].tolist())
        impostor.append((p1, p2, 0))

    all_pairs = genuine + impostor
    random.shuffle(all_pairs)
    return all_pairs


def contrastive_loss(e1, e2, labels, margin=0.5):
    import torch.nn.functional as F
    dist = F.pairwise_distance(e1, e2)
    loss = torch.mean(
        labels * torch.pow(dist, 2) +
        (1 - labels) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    )
    return loss, dist
