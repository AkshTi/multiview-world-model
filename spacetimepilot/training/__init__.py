"""Training scaffold for multi-view-consistent SPT via MC-marginalized DMD.

See IMPLEMENTATION_PLAN.md at repo root. This package is built rung by rung:
  score.py       - velocity<->score + DMD math (Rung 1, pure/unit-tested)
  latents.py     - VAE-latent helpers (Rung 2)
  freeze.py      - freeze/unfreeze + grad-mask assertions (Rung 2)
  steps.py       - training steps (Rung 2/3/5)
  middle_bank.py - cached middles v1 ~ p(v1|v0) (Rung 4)

Only `score.py` is imported eagerly here because it depends on torch alone; the other
modules pull in the SPT model and are imported where needed.
"""

from . import score  # noqa: F401
