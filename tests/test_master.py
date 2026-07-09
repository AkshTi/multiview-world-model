"""CPU tests for fp32 master weights (Rung 5/6 hardening).

Pins the actual failure mode and the fix:
  * a sub-ulp update rounds away when written straight to a bf16 param;
  * MasterAdamW accumulates such updates in fp32 and the bf16 param eventually moves;
  * the grad-bridge / write-back helpers are correct.
"""

import importlib.util
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_HERE, os.pardir, "spacetimepilot", "training", "master.py")
_spec = importlib.util.spec_from_file_location("spt_master", _PATH)
mw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mw)


def test_bf16_write_rounds_away_a_sub_ulp_update():
    """Baseline: adding a sub-ulp value straight to a bf16 weight is a no-op (the problem)."""
    p = torch.tensor(0.05, dtype=torch.bfloat16)
    moved = (p + torch.tensor(1e-5, dtype=torch.bfloat16)).to(torch.bfloat16)
    assert moved.item() == p.item()  # 1e-5 << bf16 ulp at 0.05 -> rounds away


def test_master_accumulates_and_moves_the_bf16_param():
    """MasterAdamW: a small lr that would round away each step DOES move the param."""
    p = torch.nn.Parameter(torch.tensor([0.05], dtype=torch.bfloat16))
    start = p.item()
    opt = mw.MasterAdamW([p], lr=1e-4)
    for _ in range(5):
        opt.zero_grad()
        p.grad = torch.ones_like(p)  # constant grad; each AdamW step ~ -lr
        opt.step()
    assert p.item() != start                  # bf16 param actually moved
    assert p.item() < start                   # in the descent direction
    # model and master stay in sync (model is the master cast to bf16)
    assert p.item() == opt.master_params[0].to(torch.bfloat16).item()


def test_grad_bridge_copies_bf16_grad_to_fp32_master():
    model = [torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))]
    master = [model[0].detach().clone().float().requires_grad_(True)]
    model[0].grad = torch.tensor([1.0, -2.0, 0.5], dtype=torch.bfloat16)
    mw.sync_grads_to_master(model, master)
    assert master[0].grad.dtype == torch.float32
    assert torch.allclose(master[0].grad, model[0].grad.float())


def test_write_back_casts_master_into_model():
    model = [torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))]
    master = [torch.tensor([1.25, -0.5], dtype=torch.float32, requires_grad=True)]
    mw.sync_master_to_model(model, master)
    assert model[0].dtype == torch.bfloat16
    assert torch.allclose(model[0].float(), master[0].detach())


def test_rejects_empty_trainable_set():
    frozen = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
    raised = False
    try:
        mw.MasterAdamW([frozen], lr=1e-4)
    except ValueError:
        raised = True
    assert raised


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nAll {len(tests)} master-weight tests passed.")


if __name__ == "__main__":
    _run_all()
