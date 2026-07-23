# Temporal Aggregation Erases the Advantage of Surgical-Domain Pretraining

**Andrew Haggstrom** — draft skeleton, rev. 4 (2026-07-22)
`[HAVE]` = committed results · `[NEED]` = not yet run
Primary evidence: 5-fold CV over all 80 Cholec80 videos
(`reports/kfold_paired_causal.json`). All tables auto-generated into
`reports/RESULTS.md`.

> **Thesis.** Frozen vision foundation models for surgical video are selected
> by frame-level linear-probe accuracy. On that metric, surgical-domain
> pretraining looks decisive: EndoViT leads general-purpose DINOv2 by 6.0
> accuracy points (p < 0.001, 80 paired videos). Apply the temporal
> aggregation any deployed system performs and **85% of that advantage
> disappears** — the residual 0.9-point gap is statistically indistinguishable
> from zero. The effect is specific to the domain-pretrained encoder: gaps
> between general-purpose encoders are unchanged by smoothing. Frame-level
> benchmarks systematically overstate what domain pretraining buys, because
> much of what it buys is single-frame discriminability that temporal
> smoothing recovers for free.

---

## 1. Introduction `[HAVE]`

- Frozen VFMs as perception backbones for surgical/robotic systems; the
  practical question is which backbone to pay for.
- Standard evaluation: linear probe / kNN on i.i.d. sampled frames
  (Ramesh et al. MedIA 2023; SurgeNetXL; PL-Stitch).
- **The gap.** A deployed surgical system consumes an ordered stream and
  aggregates over it. Frame-level protocols measure the one condition the
  system never operates in.
- **Contributions:**
  1. **The domain-pretraining advantage is largely temporal-recoverable.**
     EndoViT's frame-level lead over DINOv2 shrinks 85% under causal
     smoothing to non-significance; its lead over BiomedCLIP shrinks 64%.
  2. **The effect is specific to the domain-pretrained encoder.** The
     DINOv2–CLIP gap is unchanged by smoothing (+0.033 → +0.032, both
     p < 0.001), ruling out a generic compression-toward-the-mean explanation.
  3. **Supervision beats domain.** BiomedCLIP ≈ CLIP at every window; medical
     pretraining delivered by language supervision transfers nothing.
  4. **Magnitude.** All four encoders over-segment surgical phase by 80–124×
     at frame accuracies that read as respectable.
  5. A temporal reliability suite with a variance normalization that raw drift
     metrics demonstrably require; 12 synthetic controls; released code.
  6. **A retracted result.** An apparent rank inversion at n = 32 did not
     replicate at n = 80 (§5.3). We report the retraction because it bears on
     how sub-2-point margins in this literature should be read.

## 2. Related work `[NEED: lit review pass]`
- Phase recognition: EndoNet → TeCNO → Trans-SVNet → LoViT → SKiT. **Framing:**
  these temporal models exist because frozen features are unstable; we quantify
  what their aggregation recovers, and show it partly substitutes for domain
  pretraining.
- Surgical/medical foundation models: EndoViT, BiomedCLIP, SurgeNetXL,
  SurgXBench (WACV 2026).
- Frozen-feature protocols and their implicit i.i.d. assumption.

## 3. Method

### 3.1 Encoders — the pretraining 2×2 `[HAVE]`

|  | general | medical / surgical |
|---|---|---|
| **self-supervised** | DINOv2 ViT-S/14 (LVD-142M) | EndoViT ViT-B/16 (Endo700k, MAE) |
| **language-supervised** | CLIP ViT-B/16 (WIT-400M) | BiomedCLIP ViT-B/16 (PMC-15M) |

Measured: 384-dim (ViT-S) vs 768-dim (ViT-B); **VRAM 489 MB vs 856–1011 MB**.
Extraction throughput (101 vs 93–98 fps) is **I/O-bound on JPEG loading, not
GPU-bound**, and is reported as pipeline throughput only.
`[NEED: synthetic forward-pass benchmark]`

Uniform frozen-encoder interface; per-encoder normalization (ImageNet / CLIP /
endoscopic / PMC, the last read from the open_clip transform rather than
hardcoded). Config hash partitions the embedding cache; SHA256 manifests
verified on read.

### 3.2 Protocols `[HAVE]`
- **P1 Linear probe** — class-balanced logistic regression on frozen features.
- **P2 kNN (k = 20, cosine)** — no learned head; reads raw embedding geometry.
- **P3 Temporally smoothed accuracy** — majority filter of window *w* over the
  P1 prediction stream. **Causal/trailing** (real-time; the only deployable
  variant) is primary; **centered** (offline) reported for comparison.
  *w* = 1 recovers P1 exactly.

### 3.3 Temporal reliability metrics `[HAVE]`
Computed **within a contiguous sequence**, never across video boundaries.
`normalized_drift` (inter-frame cosine distance ÷ sequence spread — necessary,
not cosmetic, §5.2), `boundary_jitter` (predicted transitions ÷ ground-truth
rate), `conditional_stability` (guards the degenerate constant encoder),
`velocity_flow_coupling` (Spearman vs dense optical flow).
**Validation:** 12 synthetic controls with known answers (static → zero drift;
shuffled → maximal; constant encoder → not rewarded; causal filter provably
cannot see forward; exact transition counts).

### 3.4 Data and statistics `[HAVE]`
Cholec80, 1 fps (184,498 frames / 80 videos).

- **Primary analysis: 5-fold CV over all 80 videos**, shared fold assignment
  across encoders so every comparison is paired. 64 train / 16 test per fold;
  every video scored exactly once.
- **Paired video-level bootstrap** (10,000 resamples) on per-video differences.
  Pairing cancels the large between-video variation in length and difficulty;
  overlapping marginal CIs do *not* imply a non-significant paired difference.
- Frames within a surgery are near-duplicates, so **all splitting and
  resampling is at the video level**. Frame-level bootstrapping would give
  intervals an order of magnitude too tight.
- Macro-F1 reported alongside accuracy (phase imbalance up to 31×).
- 6 pairs × 5 windows = 30 tests; **24 significant at p < 0.05 vs ~1.5 expected
  by chance**, so the panel is well-powered and the null results below are
  informative rather than merely underpowered.

## 4. Results

### 4.1 Frame-level protocols disagree `[HAVE]`
Fixed 40/8/32 split, for comparability with published protocols:

| Encoder | P1 acc | P1 F1 | 95% CI | P2 acc | P2 F1 | P1−P2 gap |
|---|---|---|---|---|---|---|
| EndoViT | **0.704** | **0.615** | [0.565, 0.671] | 0.522 | 0.377 | 0.238 |
| DINOv2 | 0.677 | 0.597 | [0.552, 0.637] | 0.611 | 0.463 | 0.134 |
| BiomedCLIP | 0.658 | 0.578 | [0.536, 0.615] | **0.617** | **0.504** | 0.075 |
| CLIP | 0.648 | 0.568 | [0.525, 0.607] | 0.572 | 0.391 | 0.177 |

**P1 selects EndoViT; P2 selects BiomedCLIP.** Two protocols routinely reported
side by side disagree on the winner. EndoViT has the best linear separability
and the *worst* geometric clustering — its phases are separable by a hyperplane,
not by proximity (smallest feature spread of the four, emb_std 0.425).

Per-phase pattern is identical across all four: dissection phases strong
(0.69–0.77), transitional/rare phases collapse (0.46–0.59). **Structural, not
model-specific** — these are the states a single frame cannot disambiguate.

### 4.2 Temporal reliability at w = 1 `[HAVE]`

| Encoder | acc | drift_ratio | jitter (× GT) | cond. stab. | run len |
|---|---|---|---|---|---|
| EndoViT | 0.719 | 0.512 | 79.7 | 0.475 | 5.9 |
| DINOv2 | 0.689 | 0.665 | 110.5 | 0.336 | 4.0 |
| BiomedCLIP | 0.664 | — | 122.4 | 0.328 | 3.5 |
| CLIP | 0.661 | 0.800 | 124.3 | 0.199 | 3.5 |

**Magnitude finding.** Every encoder over-segments phase by 80–124×. The best —
a surgical foundation model pretrained on this distribution — still announces
80× too many transitions at 0.72 frame accuracy. This is a within-encoder
measurement against ground truth, two orders of magnitude in size, and does not
depend on any cross-encoder comparison.

**Figure 1** (`figures/phase_ribbon.png`): ground-truth phase bands vs. per-frame
predictions, median-jitter test video. Motivation figure.

### 4.3 The domain advantage converges away `[HAVE — headline]`

**Mean accuracy, 5-fold CV over all 80 videos, causal smoothing:**

| Encoder | w=1 | w=15 | w=31 | w=61 | w=121 |
|---|---|---|---|---|---|
| EndoViT | **0.756** | **0.805** | **0.810** | **0.795** | **0.751** |
| DINOv2 | 0.697 | 0.779 | 0.790 | 0.784 | 0.742 |
| BiomedCLIP | 0.676 | 0.760 | 0.770 | 0.762 | 0.723 |
| CLIP | 0.664 | 0.753 | 0.764 | 0.752 | 0.710 |

**Mean macro-F1:**

| Encoder | w=1 | w=15 | w=31 | w=61 | w=121 |
|---|---|---|---|---|---|
| EndoViT | **0.644** | **0.709** | **0.707** | 0.648 | 0.530 |
| DINOv2 | 0.599 | 0.695 | 0.700 | **0.652** | **0.533** |
| BiomedCLIP | 0.576 | 0.674 | 0.675 | 0.626 | 0.513 |
| CLIP | 0.573 | 0.679 | 0.683 | 0.636 | 0.514 |

**Ordering is preserved at every window — there is no rank inversion (§5.3).**
What changes is the *size* of the gaps.

**Paired differences, accuracy, causal (n = 80):**

| Pair | w=1 | w=31 | w=121 | shrinkage |
|---|---|---|---|---|
| EndoViT − DINOv2 | +0.060*** | +0.019* | +0.009 n.s. | **−85%** |
| EndoViT − CLIP | +0.092*** | +0.046*** | +0.041*** | −55% |
| EndoViT − BiomedCLIP | +0.080*** | +0.040*** | +0.029** | −64% |
| DINOv2 − CLIP | +0.033*** | +0.026*** | +0.032*** | **−2%** |
| DINOv2 − BiomedCLIP | +0.021** | +0.020* | +0.020* | −5% |
| CLIP − BiomedCLIP | −0.012* | −0.006 n.s. | −0.013 n.s. | — |

`*** p<0.001, ** p<0.01, * p<0.05`

- **Every EndoViT gap shrinks by 55–85%. No general-encoder gap shrinks at
  all.** DINOv2−CLIP is +0.033 at w=1 and +0.032 at w=121 — flat. This rules
  out generic regression-to-the-mean from smoothing and localizes the effect
  to the domain-pretrained encoder.
- **EndoViT vs DINOv2 crosses into non-significance** between w=31 (p = 0.017)
  and w=61 (p = 0.192). At w=121, EndoViT wins 44/80 videos — a coin flip.
- **Interpretation.** Surgical pretraining buys single-frame discriminability.
  Temporal aggregation recovers most of the same information from context, so
  the two paths to the same signal substantially overlap. A practitioner who
  smooths — i.e. any deployed system — pays for domain pretraining and receives
  ~1 non-significant point over a model one-third the size using half the VRAM.
- **Operating point.** Accuracy peaks at w ≈ 31 for all encoders and declines
  thereafter (trailing windows lag transitions); macro-F1 peaks at w ≈ 15–31
  and collapses by w = 121 as short rare phases are erased. **w ≈ 31 is the
  honest operating point.**
- Jitter falls 38–54× across the sweep while accuracy *rises*: the phase
  information was in the frozen features throughout, and P1 gave no signal
  about how much temporal repair each encoder would need.

**Figure 2** `[NEED: regenerate from k-fold]`: paired difference vs. window,
one line per pair, with the significance boundary marked. The EndoViT lines
converging to zero while the general-encoder lines stay flat **is the result**.

### 4.4 Supervision beats domain `[HAVE]`
BiomedCLIP − CLIP is non-significant at every window w ≥ 15 (p = 0.17–0.47,
differences ≤ 0.013, splits 33–42 of 80). Medical-domain pretraining delivered
through *language supervision* transfers essentially nothing to surgical video,
while medical-domain *self-supervised* pretraining (EndoViT) yields a large
frame-level lead.

Both SSL encoders beat both language-supervised encoders at every window.
**The supervision signal matters more than the domain label.** Plausible
mechanism: PMC-15M is static biomedical figures — histology, radiology,
diagrams — whose visual statistics differ sharply from endoscopic video.
Nominal domain proximity does not imply transfer.

Note this is a **well-powered null**: in the same test panel, 24 of 30
comparisons reached significance.

### 4.5 Is drift signal or noise? `[HAVE, 3 encoders]`
Spearman(embedding velocity, optical-flow magnitude), 10 test videos, all
p < 0.05: EndoViT 0.499, DINOv2 0.471, CLIP 0.388. Moderate — drift partly
tracks scene motion, but roughly half the variance is unexplained by pixel
motion. Ordering matches the stability ranking.
`[NEED: re-run including BiomedCLIP]`

### 4.6 Remaining `[NEED]`
- Regenerate Figure 2 from k-fold paired differences.
- Centered-smoothing k-fold run (for the offline/online contrast).
- Synthetic forward-pass FPS benchmark.
- Flow coupling for BiomedCLIP.
- Cross-procedure transfer (AutoLaparo) — blocked on dataset access.

## 5. Analysis

### 5.1 Why domain pretraining is temporal-recoverable `[HAVE]`
EndoViT has the largest P1−P2 gap of the four (0.238): high linear separability,
low geometric clustering. Temporal smoothing aggregates over local
neighbourhoods — the structure EndoViT relies on least. Its frame-level
advantage therefore rests on a property that aggregation substitutes for, while
DINOv2's neighbourhood structure and smoothing are complementary.

**Stated as an observation, not a rule.** CLIP has the second-largest gap
(0.177) with a stable, non-shrinking margin; BiomedCLIP has the smallest gap
(0.075) and likewise stable margins. The pattern is driven by EndoViT alone
(n = 1 among four encoders) and requires a larger panel to test.

### 5.2 Raw drift is variance-gameable `[HAVE — methodological]`
An early cross-encoder run showed CLIP as *more* temporally stable than DINOv2
(raw drift 0.042 vs 0.124). Diagnosis: CLIP's feature spread was 3.5× smaller
and emb_std 2.7× smaller — low drift from low variance, not stability. The
internal contradiction that flagged it: CLIP simultaneously jittered *more*
(124× vs 110×).

Normalizing drift by sequence spread removes the artifact. Critically, EndoViT
has the **lowest** raw variance of the four (emb_std 0.425) yet is genuinely
the most stable (drift_ratio 0.512) — low variance done *right*. Raw drift and
raw variance each mislead in isolation; only the normalized metric distinguishes
a bland encoder from a well-organized one.

### 5.3 A retracted result, and what it implies `[HAVE]`
On the fixed 40/8/32 split (32 test videos, 40 training), causal smoothing
appeared to **invert** the DINOv2/EndoViT ranking: EndoViT +0.030 at w = 1,
DINOv2 +0.021 at w = 121. We reported this internally as the project's
headline.

It did not replicate. Under 5-fold CV (80 test videos, 64 training), the sign
never flips; EndoViT leads at every window. The apparent inversion was an
artifact of a smaller evaluation set and a smaller training set — at n = 32 no
pairwise comparison in the panel reached significance, including EndoViT's
frame-level lead (p = 0.076).

**This has a methodological implication beyond our own error.** Sub-2-point
margins between frozen encoders on a 32-video Cholec80 test split are not
resolvable, and the surgical-CV literature routinely reports such rankings
without paired error bars. We recommend paired video-level bootstrapping and
cross-validation over fixed splits whenever the reported margin is small.

### 5.4 Limitations `[HAVE]`
- **Domain overlap.** Endo700k includes Cholec80, so EndoViT saw these videos
  during self-supervised pretraining. This inflates its frame-level lead —
  which makes the convergence finding *conservative*: the advantage that
  smoothing erases is, if anything, overstated at w = 1.
- **Single dataset, single procedure.** Cross-procedure replication (AutoLaparo)
  is the outstanding generalization check.
- **n = 4 encoders.** Sufficient for pairwise claims at n = 80 videos;
  insufficient for §5.1 or for rank correlations.
- **No learned temporal head.** Majority smoothing is a floor; a TCN would
  upper-bound what aggregation recovers, and would likely erase *more* of the
  domain advantage, not less.
- **Efficiency numbers are pipeline-level**, not model-level.
- k-fold uses 64 training videos vs 40 in the fixed split, so §4.3 accuracies
  are not directly comparable to §4.1.

## 6. Conclusion `[HAVE]`
Surgical-domain pretraining delivers a large, unambiguous advantage on the
frame-level probe that the field uses to select frozen backbones — 6.0 accuracy
points for EndoViT over DINOv2 across 80 paired videos. Under the temporal
aggregation a deployed system performs, 85% of that advantage disappears and
the remainder is indistinguishable from zero, while gaps between
general-purpose encoders are untouched. Meanwhile every encoder tested
over-segments surgical phase by 80–124× at frame accuracies that read as
respectable.

Practitioners should evaluate frozen backbones under the aggregation regime
they intend to deploy, report paired error bars on any margin below a few
points, and treat frame-level gains from domain pretraining as an upper bound
on deployed benefit. We release the evaluation suite to make this cheap.

---

## Appendix: what would strengthen this
- 6–8 encoders → a real test of §5.1's mechanism.
- Cross-procedure replication (AutoLaparo, HeiChole).
- Learned temporal head (TCN) as the aggregation upper bound.
- Per-phase temporal breakdown: is jitter concentrated at phase boundaries?
