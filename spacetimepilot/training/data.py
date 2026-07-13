"""Training data layer for the DMD loop (PLAN_DMD_TRAINING.md §7 items 2 + A5).

This module is the *sampler*, not the tensor builder — it deterministically turns a global
training step into a ``TrainingItem`` (which v0, which K middles, which target action), and
``batch.build_batch`` later materializes that into the tensor dict ``steps.dmd_step_k`` eats.

Design invariants (A5):
  * v0 drawn uniformly from the train split;
  * K_step middles drawn WITHOUT replacement from the v0's cached bank (K_bank >= K_step);
  * target action a2 = a pool trajectory *minus* the middles' trajectories, forward time;
  * ALL randomness is keyed by the global step, so a resumed run reproduces the exact stream
    (``sample(s)`` depends only on ``(base_seed, s)`` — never on call history).

Everything here is pure Python + torch RNG + cheap bank-metadata reads (no VAE, no GPU), so
it is CPU/login-node runnable and unit-tested in tests/test_data.py. Camera geometry (the
overlap score of ledger #7) is intentionally NOT invented here — it is injected via an
optional callable so this module never bakes a guessed metric.
"""

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from . import middle_bank as mbk

# Demo camera pool exposes cam01..cam10 (see middle_bank.DEFAULT_CROSSING_CAM_TYPES).
DEFAULT_CAMERA_POOL = tuple(range(1, 11))
_SEED_STRIDE = 1_000_003  # large prime: decorrelate consecutive steps' generators


# --------------------------------------------------------------------------------------- #
# Split                                                                                     #
# --------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class V0Entry:
    """One source video in a split: a stable id, its pixel path, and its caption."""
    v0_id: str
    video_path: str
    caption: str


def build_split_from_metadata(metadata_csv, video_dir, seed=0, n_heldout=8):
    """Deterministic train/held-out split over a ``file_name,text`` metadata csv (PLAN B1).

    Returns ``(train, heldout)`` lists of ``V0Entry``. The permutation is seeded so the
    split is reproducible and committable (config/data/pilot_split.json).
    """
    entries = []
    with open(metadata_csv, newline="") as f:
        for row in csv.DictReader(f):
            fname = row["file_name"]
            v0_id = os.path.splitext(os.path.basename(fname))[0]
            entries.append(V0Entry(v0_id=v0_id,
                                   video_path=os.path.join(video_dir, fname),
                                   caption=row["text"]))
    if n_heldout >= len(entries):
        raise ValueError(f"n_heldout={n_heldout} >= corpus size {len(entries)}")
    g = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(entries), generator=g).tolist()
    heldout_pos = set(order[:n_heldout])
    heldout = [entries[i] for i in sorted(heldout_pos)]
    train = [entries[i] for i in range(len(entries)) if i not in heldout_pos]
    return train, heldout


def write_split(path, train, heldout):
    """Serialize a split to json (order preserved; the committed source of truth)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "schema": 1,
        "train": [e.__dict__ for e in train],
        "heldout": [e.__dict__ for e in heldout],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(path, which="train"):
    """Load one side ("train"/"heldout") of a committed split json as ``V0Entry`` list."""
    with open(path) as f:
        payload = json.load(f)
    return [V0Entry(**e) for e in payload[which]]


# --------------------------------------------------------------------------------------- #
# Step-keyed sampling                                                                       #
# --------------------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MiddleRef:
    """A cached middle chosen for this step: its bank idx, trajectory, time, and mp4 path."""
    idx: int
    cam_idx: int
    time_pattern: str
    video_path: str


@dataclass
class TrainingItem:
    """One resolved training example: v0 + its K middles + the target action a2."""
    v0_id: str
    video_path: str
    caption: str
    target_cam_idx: int
    target_time_pattern: str
    src_time_pattern: str
    middles: list  # list[MiddleRef]
    overlap_score: Optional[float] = None
    global_step: int = -1


def step_generator(base_seed, global_step):
    """A CPU ``torch.Generator`` determined solely by ``(base_seed, global_step)``.

    Keying on the step (not on call order) is what makes resume bit-exact: sample(s) is a
    pure function of s. The prime stride keeps neighbouring steps' streams decorrelated.
    """
    seed = (int(base_seed) * _SEED_STRIDE + int(global_step)) % (2**63)
    return torch.Generator().manual_seed(seed)


def sample_without_replacement(n, k, generator):
    """Return ``k`` distinct indices in ``[0, n)`` via a seeded permutation (deterministic)."""
    if k > n:
        raise ValueError(f"cannot draw {k} without replacement from {n}")
    return torch.randperm(n, generator=generator)[:k].tolist()


class V0Sampler:
    """Deterministic per-step sampler over a train split and its cached middle bank.

    ``sample(global_step)`` reads the bank metadata for the drawn v0 (cheap json) and returns
    a fully-resolved ``TrainingItem``. It does NOT load pixels or touch a GPU — that is
    ``batch.build_batch``'s job.
    """

    def __init__(self, entries, bank_dir, k_step, camera_pool=DEFAULT_CAMERA_POOL,
                 base_seed=0, target_time_pattern="forward", src_time_pattern="forward",
                 overlap_fn: Optional[Callable] = None):
        if not entries:
            raise ValueError("V0Sampler got an empty split")
        if k_step < 1:
            raise ValueError("k_step must be >= 1")
        self.entries = list(entries)
        self.bank_dir = bank_dir
        self.k_step = int(k_step)
        self.camera_pool = tuple(camera_pool)
        self.base_seed = int(base_seed)
        self.target_time_pattern = target_time_pattern
        self.src_time_pattern = src_time_pattern
        # overlap_fn(target_cam_idx, [middle_cam_idx, ...]) -> float; injected, never guessed.
        self.overlap_fn = overlap_fn

    def sample(self, global_step):
        g = step_generator(self.base_seed, global_step)

        v0 = self.entries[int(torch.randint(len(self.entries), (1,), generator=g).item())]

        cached = mbk.list_middles(self.bank_dir, v0.v0_id)
        if len(cached) < self.k_step:
            raise RuntimeError(
                f"v0_id={v0.v0_id} has {len(cached)} cached middles but k_step={self.k_step} "
                f"(build the bank with >= k_step middles first)")
        positions = sample_without_replacement(len(cached), self.k_step, g)
        chosen = [cached[p] for p in positions]

        middles, used_cams = [], set()
        for idx in chosen:
            v1_path, meta_path = mbk.middle_paths(self.bank_dir, v0.v0_id, idx)
            meta = mbk.load_meta(meta_path)
            cam_idx = int(meta["cam_idx"])
            used_cams.add(cam_idx)
            middles.append(MiddleRef(
                idx=idx, cam_idx=cam_idx,
                time_pattern=meta.get("time_pattern", "forward"), video_path=v1_path))

        # a2 = pool trajectory MINUS the middles' trajectories (A5).
        target_pool = [c for c in self.camera_pool if c not in used_cams]
        if not target_pool:
            raise RuntimeError(
                f"no target camera left after excluding middles {sorted(used_cams)} from pool "
                f"{self.camera_pool}; shrink k_step or widen the pool")
        target_cam_idx = target_pool[int(torch.randint(len(target_pool), (1,), generator=g).item())]

        overlap = None
        if self.overlap_fn is not None:
            overlap = float(self.overlap_fn(target_cam_idx, [m.cam_idx for m in middles]))

        return TrainingItem(
            v0_id=v0.v0_id, video_path=v0.video_path, caption=v0.caption,
            target_cam_idx=target_cam_idx, target_time_pattern=self.target_time_pattern,
            src_time_pattern=self.src_time_pattern, middles=middles,
            overlap_score=overlap, global_step=int(global_step))
