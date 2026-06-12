import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import EarSiameseModel
from uerc26_dataset import UERC26Dataset, val_transform


def extract_features(model, loader, device):
    model.eval()
    embeddings1, embeddings2, labels_all = [], [], []

    with torch.no_grad():
        for img1, img2, label, _, _ in loader:
            img1, img2 = img1.to(device), img2.to(device)
            e1 = model.forward_one(img1)
            e2 = model.forward_one(img2)
            embeddings1.append(e1.cpu().numpy())
            embeddings2.append(e2.cpu().numpy())
            labels_all.append(label.numpy())

    emb1   = np.concatenate(embeddings1, axis=0)
    emb2   = np.concatenate(embeddings2, axis=0)
    labels = np.concatenate(labels_all,  axis=0)
    return emb1, emb2, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',        required=True)
    parser.add_argument('--img_dir',    required=True)
    parser.add_argument('--weights',    default='best_pruned_tiny.pth')
    parser.add_argument('--output_dir', default='.')
    parser.add_argument('--batch',      type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EarSiameseModel(embed_dim=256)
    state = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    print(f'Loaded weights from {args.weights}')

    dataset = UERC26Dataset(args.csv, args.img_dir, val_transform)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    emb1, emb2, labels = extract_features(model, loader, device)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'emb1.npy'),   emb1)
    np.save(os.path.join(args.output_dir, 'emb2.npy'),   emb2)
    np.save(os.path.join(args.output_dir, 'labels.npy'), labels)
    print(f'Features saved to {args.output_dir}  shape={emb1.shape}')


if __name__ == '__main__':
    main()
