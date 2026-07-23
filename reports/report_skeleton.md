# Three Protocols, Three Winners: Evaluation Choice Determines Backbone Selection for Surgical Video

**Andrew Haggstrom** — draft skeleton, rev. 2026-07-22
`[HAVE]` = committed results · `[NEED]` = not yet run · `XX` = unfilled
All numbers: `reports/RESULTS.md` (auto-generated)

> **Thesis.** Frozen vision foundation models are selected for surgical video
> by frame-level probe accuracy. We evaluate four encoders spanning the
> pretraining 2×2 (general/medical × self-supervised/language-supervised) under
> three standard protocols and find that **each protocol selects a different
> backbone from the same four candidates**: linear probing selects EndoViT,
> kNN selects BiomedCLIP, and temporally-smoothed accuracy — the deployment
> condition — selects DINOv2. Backbone choice is determined by an evaluation
> decision that is rarely reported and never justified.

---

## 1. Introduction `[HAVE]`

- Frozen VFMs as perception backbones for surgical/robotic systems.
- The standard evaluation menu: linear probe, kNN (k=20), on i.i.d. frames
  (Ramesh et al. MedIA 2023; SurgeNetXL; PL-Stitch).
- **The problem.** These protocols are treated as interchangeable proxies for
  representation quality. They are not: they disagree on which encoder is best,
  and none matches the temporal aggregation a deployed system applies.
- **Contributions:**
  1. **Protocol-determined selection.** Three standard protocols → three
     different winners from four encoders.
  2. **The deployment-relevant ranking differs from both frame-level ones**,
     under offline *and* real-time smoothing, with the crossover at the same
     window in each.
  3. **The pretraining axis that matters is supervision, not domain.** Both
     self-supervised encoders beat both language-supervised ones; medical
     language supervision (BiomedCLIP) buys nothing over general (CLIP).
  4. Magnitude: all four encoders over-segment surgical phase by 80–124×.
  5. A temporal reliability suite with a variance normalization that raw drift
     metrics demonstrably require; 12 synthetic controls; released code.

## 2. Related work `[NEED: lit review pass]`

- Phase recognition: EndoNet → TeCNO → Trans-SVNet → LoViT → SKiT. **Framing:**
  these temporal models exist *because* frozen features are unstable. We show
  that the compensation they provide also changes which backbone is best.
- Surgical/medical foundation models: EndoViT, BiomedCLIP, SurgeNetXL,
  SurgXBench (WACV 2026).
- Frozen-feature protocols and their implicit i.i.d. assumption.

## 3. Method

### 3.1 Encoders — the pretraining 2×2 `[HAVE]`

|  | general | medical / surgical |
|---|---|---|
| **self-supervised** | DINOv2 ViT-S/14 (LVD-142M) | EndoViT ViT-B/16 (Endo700k, MAE) |
| **language-supervised** | CLIP ViT-B/16 (WIT-400M) | BiomedCLIP ViT-B/16 (PMC-15M) |

**Table 1** (`RESULTS.md`). Measured: embed dim, VRAM (489 MB for ViT-S vs
856–1011 MB for ViT-B). **Extraction throughput is I/O-bound on JPEG loading,
not GPU-bound — 101 fps (ViT-S) vs 93–98 fps (ViT-B) despite a 4× parameter
difference. Reported as pipeline throughput, not model throughput.**
`[NEED: synthetic forward-pass benchmark for true FPS]`

Uniform frozen-encoder interface; per-encoder normalization (ImageNet / CLIP /
endoscopic / PMC stats, the last read from the open_clip transform rather than
hardcoded). Config hash partitions the cache; SHA256 manifests verified on read.

### 3.2 Evaluation protocols `[HAVE]`
Three, all standard, all reported in the literature as representation-quality
proxies:
- **P1 Linear probe** — logistic regression on frozen features, class-balanced.
- **P2 kNN (k=20, cosine)** — no learned head; reads raw embedding geometry.
- **P3 Temporally smoothed accuracy** — majority filter over window w, applied
  to the P1 prediction stream. **Centered** (offline/retrospective) and
  **causal/trailing** (real-time; the only deployable variant).
  w = 1 recovers P1 exactly.

### 3.3 Temporal reliability metrics `[HAVE]`
Computed **within a contiguous sequence**, never across video boundaries.
- `normalized_drift` — inter-frame cosine distance ÷ sequence spread.
  Normalization is **necessary, not cosmetic** (§5.2).
- `boundary_jitter` — predicted transitions per 100 frames, as a ratio to the
  ground-truth rate.
- `conditional_stability` — stability × neighbor-label consistency; guards the
  degenerate constant-encoder solution.
- `velocity_flow_coupling` — Spearman(embedding velocity, dense optical flow).
  Separates drift-as-signal from drift-as-noise.

**Validation.** 12 synthetic controls with known answers gate the suite (static
→ zero drift; shuffled → maximal; constant encoder → not rewarded; causal
filter provably cannot see forward; exact transition counts).

### 3.4 Data and statistics `[HAVE]`
Cholec80, 1 fps (184,498 frames / 80 videos), 40/8/32 video-level split.
**Video-level splits and video-level bootstrap**: frames within a surgery are
near-duplicates, so frame-level splitting leaks and frame-level bootstrapping
yields intervals an order of magnitude too tight. Effective n = 40 surgeries.
Macro-F1 reported throughout (phase imbalance up to 31×).

## 4. Results

### 4.1 P1 and P2 disagree `[HAVE]`

| Encoder | Linear acc | Linear F1 | 95% CI | kNN acc | kNN F1 | gap |
|---|---|---|---|---|---|---|
| EndoViT | **0.704** | **0.615** | [0.565, 0.671] | 0.522 | 0.377 | 0.238 |
| DINOv2 | 0.677 | 0.597 | [0.552, 0.637] | 0.611 | 0.463 | 0.134 |
| BiomedCLIP | 0.658 | 0.578 | [0.536, 0.615] | **0.617** | **0.504** | 0.075 |
| CLIP | 0.648 | 0.568 | [0.525, 0.607] | 0.572 | 0.391 | 0.177 |

- **P1 selects EndoViT. P2 selects BiomedCLIP.** Two protocols routinely
  reported side by side as equivalent, disagreeing on the winner.
- EndoViT: best linear separability, **worst** geometric clustering. Its phases
  are separable by a hyperplane, not by proximity — consistent with the
  smallest feature spread of the four (emb_std 0.425).
- Linear-probe CIs overlap substantially; the ordering is suggestive, not
  established. The kNN gaps are larger.
- Per-phase pattern is identical across all four encoders: dissection phases
  strong (0.69–0.77), transitional/rare phases collapse (0.46–0.59).
  **Structural, not model-specific.**

### 4.2 Temporal reliability at w = 1 `[HAVE]`

| Encoder | acc | drift_ratio | jitter (× GT) | cond. stab. | run len |
|---|---|---|---|---|---|
| EndoViT | 0.719 | 0.512 | 79.7 | 0.475 | 5.9 |
| DINOv2 | 0.689 | 0.665 | 110.5 | 0.336 | 4.0 |
| BiomedCLIP | 0.664 | — | 122.4 | 0.328 | 3.5 |
| CLIP | 0.661 | 0.800 | 124.3 | 0.199 | 3.5 |

- **Magnitude.** Every encoder over-segments by 80–124×. The best — a surgical
  foundation model pretrained on this distribution — still announces 80× too
  many transitions at 0.72 accuracy.
- Spearman(acc, stability) = 1.000, p < 0.001 — concordant **at w = 1 only**;
  §4.3 shows this does not survive smoothing.
- **Figure 1** (`figures/phase_ribbon.png`): ground-truth bands vs. per-frame
  predictions, median-jitter test video. Motivation, not result.

### 4.3 P3 selects a third winner `[HAVE — headline]`

Accuracy, **causal** (deployment) smoothing:

| Window | EndoViT | DINOv2 | BiomedCLIP | CLIP |
|---|---|---|---|---|
| 1 | **0.719** | 0.689 | 0.664 | 0.661 |
| 15 | **0.771** | 0.767 | 0.755 | 0.752 |
| 31 | 0.777 | **0.782** | 0.766 | 0.764 |
| 61 | 0.765 | **0.779** | 0.760 | 0.758 |
| 121 | 0.727 | **0.747** | 0.730 | 0.718 |

Centered (offline) smoothing shows the same crossover: DINOv2 0.833 vs EndoViT
0.801 at w = 121.

- **EndoViT falls from 1st to 3rd.** Crossover between w = 15 and w = 31 in
  *both* regimes — the inversion does not depend on access to future frames.
- Jitter drops 38–54× across the sweep. **Accuracy rises while jitter
  collapses**: the phase information was in the frozen features throughout, and
  P1 gave no signal about how much temporal repair each encoder would need.
- **Causal smoothing reveals a true optimum that centered smoothing hides.**
  Under causal, accuracy peaks at w ≈ 31 and declines (DINOv2 0.782 → 0.747)
  because a trailing window lags every transition. Under centered it rises
  monotonically — an artifact of hindsight.
- Macro-F1 collapses *below* the unsmoothed baseline by w = 121
  (0.550 / 0.499 / 0.507 / 0.500): over-smoothing erases short rare phases.
  **Honest operating point: w ≈ 31**, where DINOv2 leads on both accuracy
  (0.782) and macro-F1 (0.689).
- **Figure 2** (`figures/smoothing_inversion.png`): accuracy and jitter vs.
  window, centered and causal panels, crossover annotated. **The result figure.**

### 4.4 Supervision beats domain `[HAVE]`
BiomedCLIP ≈ CLIP at every window (0.664 vs 0.661 at w = 1; 0.806 vs 0.806 at
w = 121 centered; jitter 122.4 vs 124.3). Medical-domain pretraining delivered
through *language supervision* transfers essentially nothing to surgical video,
while medical-domain *self-supervised* pretraining (EndoViT) yields a clear
frame-level lead.

Both SSL encoders (0.689, 0.719) beat both language-supervised encoders
(0.661, 0.664) at w = 1. **The axis that matters is the supervision signal, not
the domain label.** Plausible mechanism: PMC-15M is static biomedical figures —
histology, radiology, diagrams — whose visual statistics differ sharply from
endoscopic video. Nominal domain proximity does not imply transfer.

### 4.5 Is drift signal or noise? `[HAVE, 3 encoders]`
Spearman(embedding velocity, optical-flow magnitude), 10 test videos, all
p < 0.05: EndoViT 0.499, DINOv2 0.471, CLIP 0.388. Moderate — drift partly
tracks scene motion, but ~half the variance is unexplained by pixel motion.
Ordering matches the stability ranking (internal consistency check).
`[NEED: re-run including BiomedCLIP]`

### 4.6 Remaining `[NEED]`
- Synthetic forward-pass FPS benchmark (current numbers are I/O-bound).
- Flow coupling for BiomedCLIP.
- Cross-procedure transfer (AutoLaparo) — blocked on dataset access.
- Endoscopic corruption suite.

## 5. Analysis

### 5.1 What predicts the P1 → P3 reordering? `[HAVE — weak, report as observation]`
EndoViT has the largest linear-probe/kNN gap (0.238) and the smallest smoothing
gain (+8.2 vs +14.3–14.4 for the others). The mechanism is plausible: temporal
smoothing aggregates over local neighborhoods, and EndoViT's separability
depends on a learned hyperplane rather than neighborhood geometry, so it gains
least from aggregation.

**But the relationship is not graded.** CLIP has the second-largest gap (0.177)
with an entirely typical gain (+14.4); BiomedCLIP has the smallest gap (0.075)
and the same gain. **The pattern rests on EndoViT alone (n = 1).** We report it
as an observation warranting further study, not a usable heuristic. A larger
encoder panel is required to test whether an extreme linear-kNN gap flags
encoders that lose ground under temporal aggregation.

### 5.2 Raw drift is variance-gameable `[HAVE — methodological]`
The first cross-encoder run produced an apparent inversion: CLIP scored *more*
stable than DINOv2 (raw drift 0.042 vs 0.124). Diagnosis: CLIP's feature spread
was 3.5× smaller and emb_std 2.7× smaller — low drift from low variance, not
stability. The internal contradiction that flagged it: CLIP simultaneously
jittered *more* (124× vs 110×).

Normalizing drift by sequence spread removes the artifact. Critically, EndoViT
has the **lowest** raw variance of the four (emb_std 0.425) yet is genuinely
the most stable (drift_ratio 0.512) — low variance done *right*. Raw drift and
raw variance each mislead in isolation; only the normalized metric separates a
bland encoder from a well-organized one.

### 5.3 Limitations `[HAVE]`
- **Domain overlap.** Endo700k includes Cholec80, so EndoViT saw these videos
  in self-supervised pretraining. This inflates its *frame-level* lead — and
  cuts against its already-losing smoothed result, so §4.3 is conservative.
- **Single dataset, single procedure.** Cross-procedure replication (AutoLaparo)
  is the outstanding generalization check.
- **n = 4 encoders.** Sufficient to demonstrate protocol disagreement;
  insufficient for rank correlations or for §5.1.
- **No learned temporal head.** The claim concerns frozen features under
  standard protocols, not end-to-end systems. A TCN would upper-bound what
  majority smoothing approximates.
- **Efficiency numbers are pipeline-level**, not model-level.

## 6. Conclusion `[HAVE]`
Given four frozen encoders and three standard evaluation protocols, each
protocol selects a different backbone. The encoder that wins the linear probe
is not the one that wins kNN, and neither is the one that wins under the
temporal smoothing a deployed surgical system applies — a reordering that holds
for real-time causal smoothing, not merely retrospective analysis. Meanwhile
all four over-segment surgical phase by 80–124× at frame accuracies that read
as respectable.

Practitioners should evaluate frozen backbones under the aggregation regime
they intend to deploy, and report which protocol drove the selection. We
release the suite to make the temporal evaluation cheap.

---

## Appendix: what would strengthen this
- 6–8 encoders → rank correlations across protocols; a real test of §5.1.
- Cross-procedure replication (AutoLaparo, HeiChole).
- Learned temporal head as an upper bound on smoothing.
- Per-phase temporal breakdown: is jitter concentrated at boundaries?
