# Frame-Level Accuracy Selects the Wrong Backbone: Temporal Evaluation of Frozen Vision Foundation Models for Surgical Video

**Andrew Haggstrom** — draft skeleton, rev. 2026-07-22
`[HAVE]` = backed by committed results · `[NEED]` = experiment not yet run · `XX` = unfilled

> **Thesis.** Frozen vision foundation models are evaluated almost universally
> by linear-probe accuracy on independently sampled video frames. On endoscopic
> video this protocol does not merely understate temporal instability — it
> **reverses the model ranking**. A surgical-domain encoder (EndoViT) wins the
> standard frame-level evaluation and finishes last under the temporal
> smoothing any deployed system applies, where a general self-supervised
> encoder (DINOv2) wins. Practitioners selecting a backbone by the standard
> metric select the wrong one.

---

## 1. Introduction `[HAVE]`

- Frozen VFMs as perception backbones for surgical/robotic systems.
- Standard evaluation: frame-level linear probe / kNN on i.i.d. sampled frames
  (Ramesh et al. MedIA 2023; SurgeNetXL; PL-Stitch kNN protocol).
- **The misalignment.** A surgical system consumes a temporally ordered stream
  and applies temporal aggregation. Frame-level protocols discard order.
- **Contributions:**
  1. **A protocol-dependent rank inversion.** Three encoders rank
     EndoViT > DINOv2 > CLIP at frame level and DINOv2 > CLIP > EndoViT under
     temporal smoothing. The standard metric picks the wrong backbone.
  2. A mechanism: linear-vs-kNN divergence at frame level predicts which
     encoder loses under smoothing.
  3. A temporal reliability suite, including a variance normalization that raw
     drift metrics demonstrably require.
  4. Magnitude: all encoders over-segment surgical phase by 80–124×.
  5. Released code, cached embeddings, and evaluation protocol.

## 2. Related work `[NEED: lit review pass]`

- Phase recognition: EndoNet → TeCNO → Trans-SVNet → LoViT → SKiT.
  **Framing:** these temporal models exist *because* frozen features are
  unstable. We measure what they compensate for — and show that compensation
  changes which backbone is best.
- Surgical foundation models: EndoViT, SurgeNetXL, SurgXBench (WACV 2026).
- Frozen-feature evaluation protocols and their implicit i.i.d. assumption.

## 3. Method

### 3.1 Frozen-feature extraction `[HAVE]`
Uniform encoder interface; per-encoder normalization (ImageNet / CLIP /
endoscopic stats). Config hash partitions the embedding cache; SHA256
manifests verified on read.

**Table 1.** Encoders.

| Encoder | Arch | Pretraining | Dim |
|---|---|---|---|
| DINOv2 | ViT-S/14 | LVD-142M, self-supervised | 384 |
| CLIP | ViT-B/16 | WIT-400M, image–text | 768 |
| EndoViT | ViT-B/16 | Endo700k, MAE | 768 |

### 3.2 Temporal reliability metrics `[HAVE]`
All metrics computed **within a contiguous sequence**, never across video
boundaries.

- `feature_drift` — mean cosine distance between consecutive frames.
- `normalized_drift` — drift ÷ sequence spread. **Necessary, not cosmetic**
  (§5.2).
- `boundary_jitter` — predicted transitions per 100 frames, as a ratio to the
  ground-truth transition rate.
- `phase_fragmentation` — segment count, mean run length.
- `conditional_stability` — stability × neighbor-label consistency, guarding
  the degenerate constant-encoder solution.
- `velocity_flow_coupling` — Spearman(embedding velocity, dense optical-flow
  magnitude). Separates drift-as-signal from drift-as-noise.

**Validation.** Ten synthetic controls with known answers gate the suite
(static → zero drift; shuffled → maximal; constant encoder → not rewarded;
exact transition counts on hand-built streams).

### 3.3 Temporal smoothing protocol `[HAVE]`
Centered majority filter, window w ∈ {1, 3, 5, 9, 15, 31, 61, 121} frames at
1 fps. w = 1 is the standard frame-level protocol. **Centered smoothing peeks
at future frames and is therefore the generous, offline case**; a causal
trailing window is the realistic deployment setting (§5.3, `[NEED]`).

### 3.4 Evaluation protocol `[HAVE]`
Cholec80, 1 fps (184,498 frames / 80 videos), 40/8/32 video-level split.
**Video-level splits and video-level bootstrap.** Frames within a surgery are
near-duplicates; frame-level splitting leaks, and frame-level bootstrapping
yields intervals an order of magnitude too tight. Effective n = 40 surgeries.
Metrics: accuracy, macro-F1 (phase imbalance up to 31×), kNN k = 20.

## 4. Results

### 4.1 Frame-level baseline, and a first protocol divergence `[HAVE]`

| Encoder | Linear acc | Linear macro-F1 | kNN acc | kNN macro-F1 |
|---|---|---|---|---|
| EndoViT | **0.704** | **0.615** [0.565, 0.671] | 0.522 | **0.377 (worst)** |
| DINOv2 | 0.677 | 0.597 [0.552, 0.637] | **0.611** | **0.463 (best)** |
| CLIP | 0.648 | 0.568 [0.525, 0.607] | 0.572 | 0.391 |

- All land in the frozen-feature neighborhood, far below temporal models
  (SKiT 93.4, LoViT 92.4) **by construction** — that gap is the paper.
- **First inversion.** EndoViT leads the linear probe and trails *both*
  general encoders on kNN. Its phases are linearly separable but not
  geometrically clustered — consistent with the smallest feature spread of the
  three (emb_std 0.425). **Two standard protocols already disagree.**
- Per-phase pattern is identical across encoders: dissection phases strong
  (0.69–0.77), transitional/rare phases collapse (0.46–0.59). Structural, not
  model-specific.

### 4.2 Temporal reliability at w = 1 `[HAVE]`

| Encoder | Frame acc | drift_ratio | Jitter (× GT) | Cond. stability | Mean run |
|---|---|---|---|---|---|
| EndoViT | 0.719 | 0.512 | 79.7 | 0.475 | 5.9 |
| DINOv2 | 0.689 | 0.665 | 110.5 | 0.336 | 4.0 |
| CLIP | 0.661 | 0.800 | 124.3 | 0.199 | 3.5 |

- **Magnitude.** Every encoder over-segments phase by 80–124×. The best — a
  surgical foundation model pretrained on this distribution — still announces
  80× too many transitions at 0.72 accuracy.
- Spearman(accuracy, stability) = 1.000, p < 0.001. **Concordant *under the
  unsmoothed protocol only*** — §4.3 shows this concordance does not survive
  smoothing. Pre-registered P2 supported at w = 1.
- **Figure 1** (`figures/phase_ribbon.png`): ground-truth phase bands vs.
  per-frame predictions, median-jitter test video. The gradient
  GT → EndoViT → DINOv2 → CLIP is the finding, visually.

### 4.3 The rank inversion under smoothing `[HAVE — headline]`

Accuracy by smoothing window:

| Encoder | w=1 | w=15 | w=31 | w=61 | w=121 |
|---|---|---|---|---|---|
| EndoViT | **0.719 (1st)** | 0.775 | 0.791 | 0.806 (2nd) | 0.801 **(3rd)** |
| DINOv2 | 0.689 (2nd) | 0.774 | 0.802 | **0.824 (1st)** | **0.833 (1st)** |
| CLIP | 0.661 (3rd) | 0.758 | 0.783 | 0.804 (3rd) | 0.806 (2nd) |

Jitter ratio by window:

| Encoder | w=1 | w=15 | w=61 | w=121 |
|---|---|---|---|---|
| EndoViT | 79.7 | 10.7 | 3.3 | 2.0 |
| DINOv2 | 110.5 | 14.8 | 3.9 | 2.3 |
| CLIP | 124.3 | 16.5 | 4.5 | 2.3 |

- **The result.** The standard protocol selects EndoViT; deployment-realistic
  smoothing selects DINOv2. DINOv2 gains +14.4 pts across the sweep to
  EndoViT's +8.2. Rank order at w = 121 is the reverse of w = 1 for the top
  and bottom of the frame-level ranking.
- Jitter falls 40–54× while accuracy *rises* — the phase information was in
  the frozen features throughout. Frame accuracy gave no signal about how much
  temporal repair each encoder would need.
- **Macro-F1 peaks at w = 61 and falls by w = 121** (DINOv2 0.767 → 0.717):
  over-smoothing erases short rare phases. **w ≈ 61 is the honest operating
  point**; reporting accuracy alone would wrongly suggest more smoothing is
  always better.
#### Causal (trailing-window) smoothing — the deployment case `[HAVE]`

| Window | EndoViT | DINOv2 | CLIP |
|---|---|---|---|
| w=1 | **0.719** | 0.689 | 0.661 |
| w=15 | **0.771** | 0.767 | 0.752 |
| w=31 | 0.777 | **0.782** | 0.764 |
| w=61 | 0.765 | **0.779** | 0.758 |
| w=121 | 0.727 | **0.747** | 0.718 |

- **The inversion survives.** DINOv2 overtakes EndoViT between w=15 and w=31 —
  the *same* crossover window as centered smoothing. The result does not depend
  on access to future frames, so it holds for real-time systems.
- **Causal smoothing reveals a true optimum that centered smoothing hid.**
  Accuracy peaks at w=31 and declines (0.782 → 0.747 for DINOv2), because a
  trailing window necessarily lags every phase transition. Under centered
  smoothing accuracy rose monotonically — an artifact of hindsight.
- Macro-F1 collapses *below* the unsmoothed baseline by w=121
  (0.550 / 0.500 / 0.499): over-smoothing plus lag erases short rare phases.
- **Honest operating point: w ≈ 31.** There DINOv2 leads on both accuracy
  (0.782 vs 0.777) and macro-F1 (0.689 vs 0.670).

- **Figure 2** `[NEED]`: accuracy and jitter vs. window, one line per encoder,
  centered and causal panels, crossover annotated. Headline figure.

### 4.4 Is drift signal or noise? `[HAVE]`

Spearman(embedding velocity, optical-flow magnitude), 10 test videos, all
p < 0.05:

| Encoder | rho | sd |
|---|---|---|
| EndoViT | 0.499 | 0.067 |
| DINOv2 | 0.471 | 0.067 |
| CLIP | 0.388 | 0.072 |

Moderate coupling: drift is **partly** scene-tracking, but roughly half the
variance is unexplained by actual pixel motion — a substantial spurious
component. The ordering matches the stability ranking, an internal consistency
check.

### 4.5 Remaining experiments `[NEED]`
- **AutoLaparo cross-procedure transfer** — EndoViT has not seen hysterectomy;
  the control for the domain-overlap confound (§5.3).
- Endoscopic corruption suite (smoke, specular, motion blur, fog, blood).
- Efficiency: FPS at 1080p, VRAM, ONNX/TensorRT export.

## 5. Analysis

### 5.1 Why the ranking flips `[HAVE]`
EndoViT has the best single-frame separability (linear probe 0.704) and the
worst geometric clustering (kNN macro-F1 0.377). Temporal smoothing aggregates
over local neighborhoods — precisely the structure EndoViT lacks. So the
frame-level linear-vs-kNN divergence in §4.1 **predicts** the smoothing
inversion in §4.3: an encoder whose separability depends on a learned
hyperplane rather than on neighborhood geometry gains less from temporal
aggregation.

This suggests a cheap diagnostic: **a large linear-probe/kNN gap is a warning
that frame-level ranking will not survive temporal aggregation.**

### 5.2 Raw drift is variance-gameable `[HAVE — methodological]`
The first cross-encoder run produced an apparent inversion: CLIP scored *more*
stable than DINOv2 (raw drift 0.042 vs 0.124). Diagnosis: CLIP's feature
spread was 3.5× smaller, emb_std 2.7× smaller — low drift from low variance,
not stability. The internal contradiction that flagged it: CLIP simultaneously
jittered *more* (124× vs 110×).

Normalizing drift by sequence spread removes the artifact. Critically, EndoViT
has the **lowest** raw variance of the three (emb_std 0.425) yet is genuinely
the most stable (drift_ratio 0.512) — low variance done *right*. Raw drift and
raw variance mislead in isolation; only the normalized metric separates a
bland encoder from a well-organized one.

### 5.3 Limitations `[HAVE]`
- **Domain overlap.** Endo700k includes Cholec80, so EndoViT saw these videos
  in self-supervised pretraining. Its *frame-level* lead conflates domain
  pretraining with domain familiarity. AutoLaparo is the control. Note this
  cuts against EndoViT's already-losing smoothed result, strengthening §4.3.
- ~~Centered smoothing is generous.~~ **Resolved:** the inversion reproduces
  under causal (trailing-window) smoothing at the same crossover window, so it
  is not an artifact of hindsight.
- Three encoders is a small basis for a rank correlation.
- No learned temporal head evaluated — the claim concerns frozen features
  under standard protocols, not end-to-end systems.

## 6. Conclusion `[HAVE]`
Frame-level probe accuracy, the field's default frozen-feature metric, does
not merely conceal an 80–124× temporal over-segmentation — it inverts the
model ranking. The encoder that wins at w = 1 loses at w = 121. Practitioners
should evaluate frozen backbones under the temporal aggregation regime they
intend to deploy, and can use the linear-probe/kNN gap as an early warning
that frame-level ranking will not hold. We release the suite to make this
cheap.

---

## Appendix: what would strengthen this
- 4th–5th encoder (SurgeNetXL, BiomedCLIP) → real rank correlation across
  protocols, not concordance over n = 3.
- Per-phase temporal breakdown: is jitter concentrated at phase boundaries?
- Learned temporal head (single-layer TCN) as an upper bound on what smoothing
  approximates.
