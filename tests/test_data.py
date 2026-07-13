"""CPU tests for the step-keyed data sampler (spacetimepilot/training/data.py).

Covers the load-bearing determinism guarantees:
  * step_generator + sample_without_replacement are pure functions of (seed, step);
  * V0Sampler.sample(s) is reproducible (resume-safety) and depends only on s, not call order;
  * K middles are drawn without replacement and the target camera excludes the middles' cams;
  * split builder is deterministic and partitions cleanly.

Runs on CPU with no GPU / VAE. Uses a stub bank (empty mp4 + minimal json) on a temp dir.
Runnable via pytest OR directly: `python tests/test_data.py`.
"""

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spacetimepilot.training import data, middle_bank as mbk


def _make_stub_bank(bank_dir, v0_id, cam_idxs):
    """Create `len(cam_idxs)` cached middles (empty mp4 + meta json) for one v0."""
    os.makedirs(bank_dir, exist_ok=True)
    for idx, cam in enumerate(cam_idxs):
        vpath, mpath = mbk.middle_paths(bank_dir, v0_id, idx)
        open(vpath, "wb").close()  # stub mp4; sampler never reads pixels
        meta = mbk.build_meta(
            v0_id=v0_id, idx=idx, cam_idx=cam, camera_file="x.json",
            time_pattern="forward", src_time_pattern="forward", seed=idx,
            num_inference_steps=20, cfg_scale=1.0, caption="c", source_cam_kind="identity")
        mbk.save_meta(mpath, meta)


def test_step_generator_is_pure():
    a = data.step_generator(0, 100)
    b = data.step_generator(0, 100)
    import torch
    ra = torch.randint(0, 1_000_000, (5,), generator=a)
    rb = torch.randint(0, 1_000_000, (5,), generator=b)
    assert torch.equal(ra, rb), "same (seed, step) must give the same stream"
    c = data.step_generator(0, 101)
    rc = torch.randint(0, 1_000_000, (5,), generator=c)
    assert not torch.equal(ra, rc), "different steps should (almost surely) differ"


def test_sample_without_replacement():
    g = data.step_generator(0, 7)
    idxs = data.sample_without_replacement(8, 3, g)
    assert len(idxs) == 3 and len(set(idxs)) == 3
    assert all(0 <= i < 8 for i in idxs)
    # deterministic for a fresh generator at the same key
    g2 = data.step_generator(0, 7)
    assert data.sample_without_replacement(8, 3, g2) == idxs
    try:
        data.sample_without_replacement(2, 5, data.step_generator(0, 0))
        raise AssertionError("expected ValueError for k > n")
    except ValueError:
        pass


def test_sampler_resume_and_constraints():
    with tempfile.TemporaryDirectory() as d:
        bank = os.path.join(d, "bank")
        # two v0s, each with a K_bank=8 stub bank spanning cams 1..8
        entries = [
            data.V0Entry("video_0", "video_0.mp4", "cap0"),
            data.V0Entry("video_1", "video_1.mp4", "cap1"),
        ]
        for e in entries:
            _make_stub_bank(bank, e.v0_id, cam_idxs=list(range(1, 9)))

        s = data.V0Sampler(entries, bank_dir=bank, k_step=3, base_seed=0)

        # resume-safety: same step -> identical item, regardless of call order
        i_a = s.sample(500)
        _ = s.sample(3)  # intervening call must not perturb step 500
        i_b = s.sample(500)
        assert i_a == i_b, "sample(step) must be a pure function of step"

        # constraints
        assert len(i_a.middles) == 3
        mid_idxs = [m.idx for m in i_a.middles]
        assert len(set(mid_idxs)) == 3, "middles drawn without replacement"
        mid_cams = {m.cam_idx for m in i_a.middles}
        assert i_a.target_cam_idx not in mid_cams, "a2 must exclude the middles' trajectories (A5)"
        assert i_a.global_step == 500


def test_sampler_requires_enough_middles():
    with tempfile.TemporaryDirectory() as d:
        bank = os.path.join(d, "bank")
        _make_stub_bank(bank, "video_0", cam_idxs=[1, 2])  # only 2 middles
        s = data.V0Sampler([data.V0Entry("video_0", "v.mp4", "c")],
                           bank_dir=bank, k_step=4, base_seed=0)
        try:
            s.sample(0)
            raise AssertionError("expected RuntimeError for too-few middles")
        except RuntimeError:
            pass


def test_split_builder_deterministic():
    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "metadata.csv")
        with open(csv_path, "w") as f:
            f.write("file_name,text\n")
            for i in range(20):
                f.write(f"video_{i}.mp4,caption {i}\n")
        tr1, ho1 = data.build_split_from_metadata(csv_path, "vids", seed=0, n_heldout=8)
        tr2, ho2 = data.build_split_from_metadata(csv_path, "vids", seed=0, n_heldout=8)
        assert [e.v0_id for e in tr1] == [e.v0_id for e in tr2]
        assert [e.v0_id for e in ho1] == [e.v0_id for e in ho2]
        assert len(tr1) == 12 and len(ho1) == 8
        ids = {e.v0_id for e in tr1} | {e.v0_id for e in ho1}
        assert len(ids) == 20, "train/heldout must partition the corpus with no overlap"
        # round-trip through json
        sp = os.path.join(d, "split.json")
        data.write_split(sp, tr1, ho1)
        assert [e.v0_id for e in data.load_split(sp, "train")] == [e.v0_id for e in tr1]


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in _TESTS:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(_TESTS)} data-sampler tests passed")
