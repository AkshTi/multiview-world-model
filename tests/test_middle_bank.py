"""CPU tests for the pure middle-bank helpers (Rung 4).

Covers path scheme, metadata json roundtrip, deterministic crossing selection, and
cache listing (only complete mp4+json pairs count) — no GPU, no pipeline.

Run:  python tests/test_middle_bank.py   (or pytest tests/test_middle_bank.py)
"""

import importlib.util
import os
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_HERE, os.pardir, "spacetimepilot", "training", "middle_bank.py")
_spec = importlib.util.spec_from_file_location("spt_middle_bank", _PATH)
mb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mb)


def test_stem_and_paths_are_zero_padded_and_paired():
    v, m = mb.middle_paths("/bank", "video_0", 7)
    assert v.endswith("video_0__mid007.mp4")
    assert m.endswith("video_0__mid007.json")
    assert os.path.dirname(v) == os.path.dirname(m) == "/bank"


def test_stem_is_filesystem_safe():
    # slashes and spaces in a v0 id must not create subdirectories
    stem = mb.middle_stem("a/b c", 0)
    assert "/" not in stem and " " not in stem


def test_crossing_selection_is_deterministic_and_cycles():
    a = mb.crossing_cam_types(5, pool=(1, 2, 3))
    b = mb.crossing_cam_types(5, pool=(1, 2, 3))
    assert a == b == [1, 2, 3, 1, 2]


def test_crossing_rejects_bad_args():
    for bad in (lambda: mb.crossing_cam_types(0),
                lambda: mb.crossing_cam_types(3, pool=())):
        raised = False
        try:
            bad()
        except ValueError:
            raised = True
        assert raised


def test_meta_roundtrip():
    meta = mb.build_meta(
        v0_id="video_0", idx=2, cam_idx=3, camera_file="cams.json",
        time_pattern="forward", src_time_pattern="forward", seed=0,
        num_inference_steps=50, cfg_scale=5.0, caption="a scene",
        source_cam_kind="identity")
    with tempfile.TemporaryDirectory() as d:
        _, mp = mb.middle_paths(d, "video_0", 2)
        mb.save_meta(mp, meta)
        assert mb.load_meta(mp) == meta


def test_list_middles_counts_only_complete_pairs():
    with tempfile.TemporaryDirectory() as d:
        # idx 0: complete (mp4 + json).  idx 1: json only (incomplete -> excluded).
        v0, m0 = mb.middle_paths(d, "video_0", 0)
        open(v0, "w").close()
        mb.save_meta(m0, mb.build_meta("video_0", 0, 1, "c", "forward", "forward",
                                       0, 50, 5.0, "x", "identity"))
        _, m1 = mb.middle_paths(d, "video_0", 1)
        mb.save_meta(m1, mb.build_meta("video_0", 1, 2, "c", "forward", "forward",
                                       0, 50, 5.0, "x", "identity"))
        assert mb.list_middles(d, "video_0") == [0]
        assert mb.is_cached(d, "video_0", 0)
        assert not mb.is_cached(d, "video_0", 1)


def test_list_middles_isolates_by_v0_id():
    with tempfile.TemporaryDirectory() as d:
        for vid in ("video_0", "video_1"):
            v, m = mb.middle_paths(d, vid, 0)
            open(v, "w").close()
            mb.save_meta(m, mb.build_meta(vid, 0, 1, "c", "forward", "forward",
                                          0, 50, 5.0, "x", "identity"))
        assert mb.list_middles(d, "video_0") == [0]
        assert mb.list_middles(d, "video_1") == [0]


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} middle-bank tests passed.")


if __name__ == "__main__":
    _run_all()
