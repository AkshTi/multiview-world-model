"""Middle bank (Rung 4).

The direct-teacher DMD branch marginalizes the student joint over the middle view v1:

    p'(v2|v0) = ∫ q_θ(v2|v0,v1) p(v1|v0) dv1  ≈  (1/K) Σ_k q_θ(v2|v0, v1_k),
    with v1_k ~ p(v1|v0).

Here ``p`` = the FROZEN released SPT ("what you have"). The middles v1_k are therefore
independent of the student parameters, so we pre-generate K of them per source v0 ONCE
and cache to scratch — turning a K×(N-step generation) per training step into a cheap
cache read + one VAE encode (the same encode v0 already pays every step).

Each middle is a genuine counterfactual of v0 under a *crossing* camera trajectory (one
whose path overlaps v0's; the crossing set is chosen heuristically for now — an open item
with Hyunwoo). We cache the decoded mp4 (portable, human-inspectable, re-encoded at train
time exactly like v0) plus a small json of the generation settings.

Split: the path / metadata / crossing-selection helpers here are pure and CPU-tested
(tests/test_middle_bank.py). ``generate_middle`` drives frozen-SPT inference and needs a
GPU (see scripts/build_middle_bank.py).

Camera representation note (defer to Rung 5): a middle is generated as a *target* under a
``camera_extrinsics.json`` trajectory ``cam_idx``. When v1 later conditions v2 it becomes a
*source*, and source vs. target cameras use different embeddings in SPT. We store the raw
``cam_idx`` (+ camera_file) so Rung 5 can build whichever representation the model needs;
the bank stays representation-agnostic.
"""

import json
import os

# The demo camera file exposes cam01..cam10. Default crossing pool = all of them.
DEFAULT_CROSSING_CAM_TYPES = tuple(range(1, 11))

META_SUFFIX = ".json"
VIDEO_SUFFIX = ".mp4"


def middle_stem(v0_id, idx):
    """Filesystem-safe stem for middle ``idx`` of source ``v0_id``."""
    safe = str(v0_id).replace(os.sep, "_").replace(" ", "_")
    return f"{safe}__mid{idx:03d}"


def middle_paths(bank_dir, v0_id, idx):
    """Return (video_path, meta_path) for a middle. Does not touch the filesystem."""
    stem = middle_stem(v0_id, idx)
    base = os.path.join(bank_dir, stem)
    return base + VIDEO_SUFFIX, base + META_SUFFIX


def crossing_cam_types(k, pool=DEFAULT_CROSSING_CAM_TYPES):
    """Pick K crossing camera indices deterministically by cycling the pool.

    Deterministic (not random) so a rebuilt bank reuses the same trajectories and a
    resumed run lines up with what is already cached. ``pool`` is the heuristic crossing
    set; refine which trajectories actually cross v0 with Hyunwoo.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(pool) == 0:
        raise ValueError("empty crossing pool")
    return [pool[i % len(pool)] for i in range(k)]


def build_meta(v0_id, idx, cam_idx, camera_file, time_pattern, src_time_pattern,
               seed, num_inference_steps, cfg_scale, caption, source_cam_kind):
    """Assemble the metadata dict cached alongside a middle video.

    Everything needed to (a) reproduce the generation and (b) re-condition the student on
    this middle in Rung 5: the crossing trajectory id, the time patterns, and how v0's
    source camera was set when generating.
    """
    return {
        "v0_id": str(v0_id),
        "idx": int(idx),
        "cam_idx": int(cam_idx),
        "camera_file": str(camera_file),
        "time_pattern": str(time_pattern),
        "src_time_pattern": str(src_time_pattern),
        "seed": int(seed),
        "num_inference_steps": int(num_inference_steps),
        "cfg_scale": float(cfg_scale),
        "caption": str(caption),
        "source_cam_kind": str(source_cam_kind),  # e.g. "identity" — how v0's src cam was built
        "schema": 1,
    }


def save_meta(meta_path, meta):
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def load_meta(meta_path):
    with open(meta_path) as f:
        return json.load(f)


def list_middles(bank_dir, v0_id):
    """Return sorted indices of cached, COMPLETE middles (both mp4 + json present)."""
    if not os.path.isdir(bank_dir):
        return []
    prefix = middle_stem(v0_id, 0).rsplit("mid", 1)[0] + "mid"  # "<safe>__mid"
    idxs = []
    for name in os.listdir(bank_dir):
        if name.startswith(prefix) and name.endswith(META_SUFFIX):
            stem = name[: -len(META_SUFFIX)]
            if os.path.exists(os.path.join(bank_dir, stem + VIDEO_SUFFIX)):
                try:
                    idxs.append(int(stem.rsplit("mid", 1)[1]))
                except ValueError:
                    continue
    return sorted(idxs)


def is_cached(bank_dir, v0_id, idx):
    video_path, meta_path = middle_paths(bank_dir, v0_id, idx)
    return os.path.exists(video_path) and os.path.exists(meta_path)


# --- GPU generation (needs the pipeline + a GPU) ---------------------------------------

def generate_middle(pipe, source_video, source_camera, middle_camera,
                    src_time, mid_time, caption, negative_prompt,
                    seed, num_inference_steps=50, cfg_scale=5.0, tiled=True):
    """Run frozen-SPT inference to produce ONE middle v1 (list of PIL frames).

    Mirrors single_video_test.run_inference's pipe() call exactly: v0 is the source, the
    crossing trajectory is the target camera, the released @torch.no_grad __call__ denoises
    and decodes. The DiT must hold the released weights (no student edits) — this is p, not
    q_θ. Returns the decoded pixel frames; the caller saves them + metadata.
    """
    return pipe(
        prompt=caption,
        negative_prompt=negative_prompt,
        source_video=source_video,
        target_camera=middle_camera,
        source_camera=source_camera,
        src_time_embedding=src_time,
        tgt_time_embedding=mid_time,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        tiled=tiled,
    )
