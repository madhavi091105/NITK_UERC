import numpy as np
from sklearn.metrics import roc_curve, auc as sk_auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_metrics(distances, labels):
    scores = -distances
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr     = 1.0 - tpr
    auc_val = sk_auc(fpr, tpr)

    eer     = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    fif     = float(fnr[np.argmin(np.abs(fpr - 0.01))])
    g_val   = 2.0 * auc_val - 1.0
    ver_01  = float(tpr[np.argmin(np.abs(fpr - 0.001))])
    ver_1   = float(tpr[np.argmin(np.abs(fpr - 0.01))])

    opt_idx    = int(np.argmin(np.abs(fpr - fnr)))
    opt_thresh = float(-thresholds[opt_idx])

    return {
        'AUC'         : float(auc_val),
        'EER'         : float(eer),
        'FIF'         : fif,
        'G'           : float(g_val),
        'VER@0.1%FAR' : ver_01,
        'VER@1%FAR'   : ver_1,
        'opt_thresh'  : opt_thresh,
    }


def print_metrics(m):
    print('\n' + '=' * 52)
    print('        Siamese Network — Test Results')
    print('=' * 52)
    print(f'                         AUC : {m["AUC"]:.4f}')
    print(f'                         EER : {m["EER"]:.4f}')
    print(f'                         FIF : {m["FIF"]:.4f}')
    print(f'                           G : {m["G"]:.4f}')
    print(f'               VER@0.1%%FAR : {m["VER@0.1%FAR"]:.4f}')
    print(f'                 VER@1%%FAR : {m["VER@1%FAR"]:.4f}')
    print(f'  optimal_distance_threshold : {m["opt_thresh"]:.4f}')
    print('=' * 52)
