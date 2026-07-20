# Abstract (v0 — scope contract, drafted 2026-07-19)

> **Status:** Pre-registered. Numbers marked `XX` are unfilled by design. This document
> is the scope contract for the project: work that does not serve a sentence below is
> out of scope. Revise only via `abstract_v1.md`, never by editing in place.

Frozen vision foundation model (VFM) features are increasingly proposed as the
perception backbone for surgical and robotic endoscopic systems, and are almost
universally evaluated by linear-probe accuracy on independently sampled video frames.
We argue this protocol is misaligned with deployment: a surgical system consumes a
temporally ordered stream, and a model may achieve high frame-level accuracy while
producing embeddings that are unstable across adjacent frames. We benchmark `XX`
vision foundation models (DINOv2, CLIP, and a surgical-domain encoder) on
laparoscopic video from Cholec80, CholecSeg8k, and AutoLaparo, and introduce a
temporal reliability suite measuring inter-frame feature drift, predicted
phase-transition jitter, and the correlation between embedding velocity and optical
flow, each reported under an accuracy-conditioned criterion that penalizes degenerate
constant-output solutions. Across models, frame-level linear-probe accuracy and
temporal stability rank differently (Spearman rho = `XX`, 95% CI [`XX`, `XX`]); the
best-probing encoder attains `XX`% phase accuracy while emitting `XX` predicted phase
transitions per minute against a ground-truth rate of `XX`. We further evaluate
degradation under a released suite of endoscopy-specific corruptions (smoke,
specular highlight, motion blur, lens fogging, blood occlusion) and report the
accuracy/stability/throughput Pareto frontier at 1080p on `XX` hardware. Our results
indicate that `XX`, and we recommend `XX` as a label-free proxy for temporal
reliability. Code, cached embeddings, corruption suite, and evaluation protocol are
released.

---

## Pre-registered predictions (locked 2026-07-19)

Recorded before any results exist, so that either outcome is a finding rather than a
post-hoc narrative.

| # | Prediction | Confidence |
|---|---|---|
| P1 | Probe-accuracy ranking and temporal-stability ranking will differ by at least one adjacent swap | 0.65 |
| P2 | CLIP will rank materially worse on temporal stability than on probe accuracy | 0.70 |
| P3 | DINOv2 will be the most temporally stable of the general-purpose encoders | 0.60 |
| P4 | The surgical-domain encoder will lead on both axes, but by a smaller margin on stability than on accuracy | 0.55 |
| P5 | All encoders will over-predict phase transitions by more than 5x the ground-truth rate | 0.75 |
| P6 | Rank order will be preserved under corruption (robustness will not reshuffle the leaderboard) | 0.50 |

## Stopping rules (pre-committed)

1. **Week 2 reproduction gate.** If the DINOv2 frozen linear probe on Cholec80 phase
   recognition does not land within 15 points of published frozen-feature baselines,
   all forward work halts until the discrepancy is explained. Numbers above the
   published range are treated as a leakage bug until proven otherwise.
2. **Week 3 metric gate.** Temporal metrics are not run on real embeddings until all
   four synthetic controls pass (static clip, shuffled frames, constant-embedding
   degenerate case, frame-rate doubling).
3. **Week 4 pivot rule.** If no rank inversion is observed, the framing changes to
   "frame-level accuracy is a reliable proxy, with the following exceptions" and the
   project proceeds. No additional encoders are added in an attempt to produce an
   inversion.
4. **No scope additions after Week 5** except those required to fill an `XX` above.

## Explicitly out of scope

- FAISS retrieval indexing (redundant with kNN probe evidence)
- SAM2 as a benchmarked encoder (category mismatch; permitted only as a labeled
  side-experiment on temporal memory)
- LoRA and last-layer fine-tuning (linear probe is the instrument; one adaptation
  method at most, and only if Weeks 1-4 finish early)
- t-SNE gallery figures
- Training any foundation model from scratch
