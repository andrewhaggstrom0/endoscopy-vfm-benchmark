# Failure & Dead-End Ledger

Written as it happens, not reconstructed. Entries stay even when resolved.

| Date | Symptom | Hypothesis | Resolution |
|---|---|---|---|
| 2026-07-19 | — | Project scaffolded, abstract pre-registered. | — |
| 2026-07-19 | `pip install -r requirements.txt` reported dependency conflicts for tensorflow, keras, peft, datasets, accelerate, dm-tree — none of which are in requirements.txt | Assumed PYTHONPATH leak from the openvla-oft environment | Neither. `PYTHONPATH` was empty and `sys.prefix` was correct. Root cause: `~/.local/lib/python3.11/site-packages` (the **user site** dir) is keyed on Python version only, not on conda env, so it is shared by every py3.11 env on this cluster — including openvla-oft. Fixed by exporting `PYTHONNOUSERSITE=1` from a `conda/activate.d` hook scoped to endo-vfm. Note this leaks in **both** directions: a stray `pip install --user` here would also contaminate the VLA env. Diagnostic that settled it: `site.getusersitepackages()`. |
| 2026-07-22 | Pre-registered prediction, BiomedCLIP (before extraction) | — | PREDICT: probes between CLIP and DINOv2; large linear-kNN gap like EndoViT; therefore loses ground under smoothing. Testing whether the linear-kNN-gap diagnostic (§5.1) PREDICTS rather than merely describes. |
| 2026-07-22 | Prediction check: BiomedCLIP | Predicted large linear-kNN gap like EndoViT | WRONG — BiomedCLIP has the SMALLEST gap (0.074) of four. Diagnostic direction survives (largest gap = EndoViT = smallest smoothing gain) but is driven by one point; the other three cluster at +14.4 regardless of gap. State §5.1 as a flag for extreme gaps, not a graded rule. |
