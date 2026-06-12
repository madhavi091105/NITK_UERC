import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import EarSiameseModel
from uerc26_dataset import UERC26Dataset, val_transform
from score import compute_metrics, print_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',     required=True)
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--weights', default='best_pruned_tiny.pth')
    parser.add_argument('--batch',   type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    model = EarSiameseModel(embed_dim=256)
    state = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f'Model  : {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')

    dataset = UERC26Dataset(args.csv, args.img_dir, val_transform)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f'Pairs  : {len(dataset)}')

    all_dists, all_labels = [], []
    with torch.no_grad():
        for img1, img2, label, _, _ in loader:
            img1, img2 = img1.to(device), img2.to(device)
            e1, e2 = model(img1, img2)
            dist = F.pairwise_distance(e1, e2)
            all_dists.extend(dist.cpu().numpy().tolist())
            all_labels.extend(label.numpy().tolist())

    distances = np.array(all_dists)
    labels    = np.array(all_labels)

    m = compute_metrics(distances, labels)
    print_metrics(m)

    # Sample predictions
    print('\nSample predictions (first 10):')
    for i in range(min(10, len(distances))):
        pred  = 'MATCH'     if distances[i] < m['opt_thresh'] else 'NON-MATCH'
        truth = 'MATCH'     if labels[i] == 1                 else 'NON-MATCH'
        ok    = '✓' if pred == truth else '✗'
        print(f'  Pair {i+1:2d} | dist={distances[i]:.4f} | pred={pred:<9} | truth={truth:<9} | {ok}')


if __name__ == '__main__':
    main()
