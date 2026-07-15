"""Metric 2 (quality guard): Laplacian sharpness + LPIPS-to-reference. GT-free and fully
unblocked -- unlike geometry.py's metric 1a, this needs no camera intrinsics.
"""

import numpy as np
import torch


def laplacian_var(frame_rgb):
    """Variance of a 4-neighbour discrete Laplacian on luma (blur/fog detector). Same formula
    as scripts/qc_cfg_compare.py -- keep the QC convention identical across the project."""
    g = np.asarray(frame_rgb).astype(np.float32)
    g = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
    lap = (-4.0 * g[1:-1, 1:-1]
           + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def video_sharpness(frames):
    """Mean Laplacian-variance over a list/array of RGB uint8 frames."""
    vals = [laplacian_var(f) for f in frames]
    return float(np.mean(vals)) if vals else None


_LPIPS_NET = {}


def get_lpips(net="alex", device="cuda"):
    """Cached LPIPS network -- loading it is not free; one instance per (net, device)."""
    key = (net, device)
    if key not in _LPIPS_NET:
        import lpips
        m = lpips.LPIPS(net=net).to(device)
        m.eval()
        _LPIPS_NET[key] = m
    return _LPIPS_NET[key]


def to_lpips_tensor(frames, device):
    """(N,H,W,C) uint8 -> (N,C,H,W) float in [-1,1] (same convention as compute_metrics_camxtime.py)."""
    t = torch.from_numpy(np.asarray(frames)).float() / 127.5 - 1.0
    return t.permute(0, 3, 1, 2).to(device)


def lpips_distance(frames_a, frames_b, device="cuda", net="alex", batch=8):
    """Mean LPIPS over matched frames."""
    lpips_net = get_lpips(net, device)
    a = to_lpips_tensor(frames_a, device)
    b = to_lpips_tensor(frames_b, device)
    T = min(a.shape[0], b.shape[0])
    vals = []
    with torch.no_grad():
        for i in range(0, T, batch):
            d = lpips_net(a[i:i + batch], b[i:i + batch]).squeeze().cpu()
            vals.extend(d.tolist() if d.ndim > 0 else [float(d)])
    return float(np.mean(vals)) if vals else None
