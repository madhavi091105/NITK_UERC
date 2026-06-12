import os
import argparse
import numpy as np
from score import compute_metrics, print_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dir', default='.', help='Directory with emb1.npy, emb2.npy, labels.npy')
    args = parser.parse_args()

    emb1   = np.load(os.path.join(args.emb_dir, 'emb1.npy'))
    emb2   = np.load(os.path.join(args.emb_dir, 'emb2.npy'))
    labels = np.load(os.path.join(args.emb_dir, 'labels.npy'))

    # L2 distance between embedding pairs
    distances = np.linalg.norm(emb1 - emb2, axis=1)

    m = compute_metrics(distances, labels)
    print_metrics(m)


if __name__ == '__main__':
    main()
