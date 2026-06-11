import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score


def compute_metrics(labels, scores, times=None):
    # Compute the evaluation metrics based on the labels, scores, and inference times
    fpr, tpr, _ = roc_curve(labels, scores)
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    interp = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))

    metrics = {}
    # VER @ FAR=0.1%
    metrics["VER@0.1%"] = float(interp(0.001))
    # VER @ FAR=1%
    metrics["VER@1%"] = float(interp(0.01))
    # EER
    fnr = 1 - tpr
    metrics["EER"] = float(fpr[np.argmin(np.abs(fnr - fpr))])
    # AUC
    metrics["AUC"] = float(roc_auc_score(labels, scores))

    if times is not None:
        # Get average inference time and standard deviation
        metrics["inference_time_mean"] = float(np.mean(times))
        metrics["inference_time_std"] = float(np.std(times))
        metrics["inference_time"] = float(np.sum(times))
    else:
        metrics["inference_time_mean"] = 0
        metrics["inference_time_std"] = 0
        metrics["inference_time"] = 0

    return metrics


def normalize(x, xmin, xmax):
    if xmax - xmin == 0:
        return 0.0

    return (x - xmin) / (xmax - xmin)


def get_min_max_stats(metrics, baseline_metrics=None):
    stats = {
        "ver_min": min(m["VER@0.1%"] for m in metrics.values()),
        "ver_max": max(m["VER@0.1%"] for m in metrics.values()),
        "p_min": min(np.log(m.get("num_parameters", 1)) for m in metrics.values()),
        "p_max": max(np.log(m.get("num_parameters", 1)) for m in metrics.values()),
        "s_min": min(np.log(m.get("model_size", 1)) for m in metrics.values()),
        "s_max": max(np.log(m.get("model_size", 1)) for m in metrics.values()),
        "t_min": min(m.get("inference_time", float('inf')) for m in metrics.values()),
        "t_max": max(m.get("inference_time", float('-inf')) for m in metrics.values())
    }

    if baseline_metrics is not None:
        print("Incorporating baseline metrics into min-max stats")
        print(f"Baseline VER@0.1%: {baseline_metrics.get('VER@0.1%', 'N/A')}")
        print(f"Baseline num_parameters: {baseline_metrics.get('num_parameters', 'N/A')}")
        print(f"Baseline model_size: {baseline_metrics.get('model_size', 'N/A')}")
        print(f"Baseline inference_time: {baseline_metrics.get('inference_time', 'N/A')}")
        print("---------------------------------------")

        stats["ver_min"] = max(stats["ver_min"], baseline_metrics.get("VER@0.1%", stats["ver_min"]))
        stats["p_max"] = min(stats["p_max"], np.log(baseline_metrics.get("num_parameters", 1)))
        stats["s_max"] = min(stats["s_max"], np.log(baseline_metrics.get("model_size", 1)))
        stats["t_max"] = min(stats["t_max"], baseline_metrics.get("inference_time", float('-inf')))

    return stats


def compute_rt1(metrics, stats):
    ver = metrics["VER@0.1%"]
    params = metrics.get("num_parameters", 1)
    size = metrics.get("model_size", 1)

    v = normalize(ver, stats["ver_min"], stats["ver_max"])
    p = 1 - normalize(np.log(params), stats["p_min"], stats["p_max"])  # Penalize larger models
    s = 1 - normalize(np.log(size), stats["s_min"], stats["s_max"])  # Penalize larger sizes

    return 0.5 * v + 0.2 * p + 0.3 * s, v, p, s


def compute_rt2(metrics, stats):
    ver = metrics["VER@0.1%"]
    latency = metrics["inference_time"]

    v = normalize(ver, stats["ver_min"], stats["ver_max"])
    t = 1 - normalize(latency, stats["t_min"], stats["t_max"])  # Penalize slower inference times

    return 0.7 * v + 0.3 * t, v, t
