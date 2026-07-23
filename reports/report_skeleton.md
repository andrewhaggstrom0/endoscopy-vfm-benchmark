# Frame-Level Accuracy Conceals Temporal Instability in Frozen Vision Foundation Models for Surgical Video

**Andrew Haggstrom** — draft skeleton, 2026-07-22
Status: sections marked `[HAVE]` are backed by committed results; `[NEED]`
requires an experiment not yet run; `XX` is an unfilled number.

> **Thesis.** Frozen vision foundation model features are evaluated almost
> universally by linear-probe accuracy on independently sampled video frames.
> On endoscopic video this protocol is misaligned with deployment: it reports
> respectable accuracy (0.66–0.72) for encoders that over-segment surgical
> phase by 80–124×. Domain-specific pretraining reduces but does not close
> this gap.

---

## 1. Introduction  `[HAVE]`

- Frozen VFMs as perception backbones for surgical/robotic systems.
- The standard evaluation: frame-level linear probe / kNN on i.i.d. sampled
  frames (cite Ramesh et al. MedIA 2023; SurgeNetXL; PL-Stitch kNN protocol).
- **The misalignment.** A surgical system consumes a temporally ordered
  stream. Frame-level protocols discard order entirely.
- **Contributions:**
  1. A temporal reliability suite for frozen features, with a
     variance-normalization correction that raw drift metrics require.
  2. A three-encoder benchmark spanning natural-image (DINOv2),
     language-supervised (CLIP), and surgical-domain (EndoViT) pretraining.
  3. The measurement that motivates the paper: 80–124× phase
     over-segmentation at 0.66–0.72 frame accuracy.
  4. Released code, cached embeddings, and evaluation protocol.

## 2. Related work  `[NEED: lit review pass]`

- Surgical phase recognition: EndoNet → TeCNO → Trans-SVNet → LoViT → SKiT.
  **Framing:** these temporal models exist *because* frozen features are
  unstable. This paper measures the instability they paper over.
- Surgical foundation models: EndoViT, SurgeNetXL, SurgXBench (WACV 2026).
- Frozen-feature evaluation protocols and their assumptions.

## 3. Method

### 3.1 Frozen-feature extraction  `[HAVE]`
- Uniform encoder interface; per-encoder normalization (ImageNet / CLIP /
  endoscopic stats). Config hash partitions the embedding cache; SHA256
  manifests verified on read.
- **Table 1.** Encoder specifications.
  | Encoder | Arch | Pretraining | Dim | Params |
  |---|---|---|---|---|
  | DINOv2 | ViT-S/14 | LVD-142M, self-supervised | 384 | 22M |
  | CLIP | ViT-B/16 | WIT-400M, image–text | 768 | 86M |
  | EndoViT | ViT-B/16 | Endo700k, MAE | 768 | 86M |

### 3.2 Temporal reliability metrics  `[HAVE]`
- All metrics computed **within a contiguous sequence**; never across video
  boundaries.
- `feature_drift`: mean cosine distance between consecutive frames.
- `normalized_drift`: drift ÷ sequence spread. **Necessary, not cosmetic** —
  see §5.2.
- `boundary_jitter`: predicted phase transitions per 100 frames, reported as
  a ratio against the ground-truth transition rate.
- `phase_fragmentation`: segment count and mean run length.
- `conditional_stability`: stability × neighbor-label consistency, guarding
  the degenerate constant-encoder solution.
- **Validation.** Ten synthetic controls with known answers gate the suite
  (static clip → zero drift; shuffled → maximal; constant encoder → not
  rewarded; exact transition counts on hand-built streams).

### 3.3 Evaluation protocol  `[HAVE]`
- Cholec80, 1 fps (184,498 frames / 80 videos), 40/8/32 video-level split.
- **Video-level splits and video-level bootstrap.** Frames within a surgery
  are near-duplicates; frame-level splitting leaks and frame-level
  bootstrapping produces intervals an order of magnitude too tight. Effective
  n is 40 surgeries, not 90k frames.
- Metrics: frame accuracy, macro-F1 (phase imbalance up to 31×), kNN k=20.

## 4. Results

### 4.1 Reproduction gate  `[HAVE]`
Frozen frame-level phase recognition, 40/8/32:

| Encoder | Linear acc | Linear macro-F1 | kNN acc | kNN macro-F1 |
|---|---|---|---|---|
| DINOv2 ViT-S | 0.677 | 0.597 [0.552, 0.637] | 0.611 | 0.463 |
| CLIP ViT-B | 0.648 | 0.568 [0.525, 0.607] | 0.572 | 0.391 |
| EndoViT ViT-B | XX | XX | XX | XX  `[NEED: run gate on EndoViT]` |

- Lands in the frozen-feature neighborhood; far below the 90%+ of temporal
  models (SKiT 93.4, LoViT 92.4) **by construction** — that gap is the paper.
- Linear-probe CIs overlap between DINOv2 and CLIP; the kNN gap (~7 pt
  macro-F1) is larger, consistent with CLIP's caption-alignment geometry
  being worse organized for surgical phase.
- **Per-phase pattern is identical across encoders**: dissection phases
  strong (0.69–0.76), transitional/rare phases collapse (0.46–0.56). The
  failure is structural, not model-specific.

### 4.2 Temporal reliability  `[HAVE]`

| Encoder | Frame acc | drift_ratio | Jitter (× GT) | Cond. stability | Mean run |
|---|---|---|---|---|---|
| EndoViT | 0.719 | 0.512 | 79.7 | 0.475 | 5.9 |
| DINOv2 | 0.689 | 0.665 | 110.5 | 0.336 | 4.0 |
| CLIP | 0.661 | 0.800 | 124.3 | 0.199 | 3.5 |

- **Headline.** Every encoder over-segments phase by ~80–124×. The best —
  a surgical foundation model pretrained on this distribution — still
  announces 80× too many transitions while posting 0.72 frame accuracy.
- Spearman(accuracy, stability) = 1.000, p < 0.001. **No rank inversion.**
  Frame accuracy tracks temporal stability across these three encoders.
- Pre-registered prediction P2 (CLIP probes respectably but is temporally
  worse) **supported**.
- **Figure 1** (`reports/figures/phase_ribbon.png`): ground-truth phase bands
  vs. per-frame predictions, median-jitter test video. The gradient
  GT → EndoViT → DINOv2 → CLIP is the finding, visually.

### 4.3 Ablations / additional  `[NEED]`
- Temporal smoothing sweep: how much of the jitter does a majority filter of
  window w recover? Quantifies exactly what the frame metric was hiding.
- AutoLaparo cross-procedure transfer — **the key control**: EndoViT has not
  seen hysterectomy. Does its lead survive outside its pretraining domain?
- Endoscopic corruption suite (smoke, specular, motion blur, fog, blood).
- Efficiency: FPS at 1080p, VRAM, ONNX/TensorRT export.

## 5. Analysis

### 5.1 Why frame accuracy misleads  `[HAVE]`
The phases that collapse (ClippingCutting, GallbladderRetraction) are short
and transitional — exactly the states a single frame cannot disambiguate and
temporal context resolves. Frame-level protocols score a model on precisely
the frames where its weakness is invisible.

### 5.2 Raw drift is variance-gameable  `[HAVE — methodological contribution]`
The first cross-encoder run produced an apparent rank inversion: CLIP scored
*more* temporally stable than DINOv2 (raw drift 0.042 vs 0.124). Diagnosis:
CLIP's feature spread was 3.5× smaller and emb_std 2.7× smaller — low drift
from low variance, not stability. The internal contradiction that flagged it:
CLIP simultaneously jittered *more* (124× vs 110×).

Normalizing drift by sequence spread reverses the ordering and removes the
artifact. Critically, EndoViT has the **lowest** raw variance of all three
(emb_std 0.425) yet is genuinely the most stable (drift_ratio 0.512) — low
variance done *right*. Raw drift and raw variance are each misleading alone;
only the normalized metric separates a bland encoder from a well-organized
one.

### 5.3 Limitations  `[HAVE]`
- **Domain overlap.** Endo700k includes Cholec80, so EndoViT has seen these
  videos in self-supervised pretraining. Its lead conflates domain
  pretraining with domain familiarity. AutoLaparo (§4.3) is the control.
- Three encoders is a small basis for a rank correlation; rho = 1.0 over
  n = 3 is concordance, not a strong estimate.
- No temporal head evaluated — the claim is about *frozen features under the
  standard protocol*, not about deployed systems.
- CholecSeg8k pilot limited to 80-frame clips; Cholec80 carries the analysis.

## 6. Conclusion  `[HAVE]`
Frame-level probe accuracy, the field's default frozen-feature metric, gives
no warning of an 80–124× temporal over-segmentation. Domain pretraining
improves both axes but closes neither gap. Practitioners selecting a frozen
backbone for surgical video should report temporal reliability alongside
probe accuracy; we release the suite to make that cheap.

---

## Appendix: what would strengthen this
- 4th–5th encoder (SurgeNetXL, BiomedCLIP) → real rho, and the last honest
  chance at a rank inversion.
- Optical-flow / embedding-velocity correlation (second pass, needs pixels).
- Per-phase temporal breakdown: is jitter concentrated at boundaries?
