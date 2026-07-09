"""fp32 master weights for bf16-parameter training (Rung 5/6 hardening).

The SPT DiT trains in bf16. A small AdamW update (~lr) on a bf16 weight of magnitude ~0.05
is below the bf16 ulp there (~1e-4) and **rounds away on write** — the weight never moves,
so at a realistic lr (1e-5..1e-4) the fake-score net never separates from the teacher and the
DMD arrow stays 0. (This is why the K=1/K>1 smokes had to inflate ``fake_lr`` to 1e-2.)

The standard mixed-precision fix: keep an fp32 **master** copy of each trainable param. The
forward still uses the bf16 params (and receives bf16 grads); the optimizer steps on the fp32
masters (so sub-ulp updates accumulate instead of rounding away), then casts the masters back
into the bf16 params. A single sub-ulp update still quantizes on write-back, but the master
*retains* it, so after enough steps the accumulation crosses a bf16 boundary and the model
param ticks — real small-lr training works.

``MasterAdamW`` is a drop-in for the ``{zero_grad(set_to_none=), step()}`` calls that
``steps.dmd_step_k1`` / ``dmd_step_k`` make on their optimizers, so nothing else changes.

The grad-bridge / write-back logic is pure and CPU-tested (tests/test_master.py).
"""

import torch


def sync_grads_to_master(model_params, master_params):
    """Copy bf16 model-param grads into the fp32 master grads (before ``opt.step()``)."""
    for mp, p in zip(master_params, model_params):
        if p.grad is None:
            mp.grad = None
        else:
            g = p.grad.detach().float()
            if mp.grad is None:
                mp.grad = g
            else:
                mp.grad.copy_(g)


@torch.no_grad()
def sync_master_to_model(model_params, master_params):
    """Cast fp32 master weights back into the bf16 model params (after ``opt.step()``)."""
    for mp, p in zip(master_params, model_params):
        p.data.copy_(mp.data)


class MasterAdamW:
    """AdamW that steps on fp32 masters of bf16 params, then writes them back to bf16.

    ``params`` is the trainable-parameter list (e.g. from ``freeze.set_trainable``). Only
    params with ``requires_grad`` are tracked. Costs one fp32 copy of the trainable subset
    plus fp32 AdamW state (fine on H200; the trainable subset is a fraction of the 1.3B DiT).
    """

    def __init__(self, params, **adamw_kwargs):
        self.model_params = [p for p in params if p.requires_grad]
        if not self.model_params:
            raise ValueError("MasterAdamW got no trainable params")
        self.master_params = [
            p.detach().clone().float().requires_grad_(True) for p in self.model_params]
        self.opt = torch.optim.AdamW(self.master_params, **adamw_kwargs)

    def zero_grad(self, set_to_none=True):
        self.opt.zero_grad(set_to_none=set_to_none)
        for p in self.model_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self):
        sync_grads_to_master(self.model_params, self.master_params)
        self.opt.step()
        sync_master_to_model(self.model_params, self.master_params)
