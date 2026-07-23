"""Pull every committed result into one markdown file, so writing the report is
transcription rather than archaeology.

Also measures encoder size/throughput from the extraction manifests, which
already recorded fps and peak VRAM per video.
"""
from __future__ import annotations
import glob, json, os
from pathlib import Path
import numpy as np, pandas as pd

B = os.environ["BIGDIR"]
ENC = ["dinov2_vits14", "clip_vitb16", "endovit_vitb16", "biomedclip_vitb16"]
out = ["# Consolidated results\n", "_auto-generated; do not edit by hand_\n"]

# --- Table 1: encoder specs + measured efficiency ---------------------------
rows = []
for e in ENC:
    d = glob.glob(f"{B}/endoscopy/cache/{e}/*/cholec80")
    if not d:
        continue
    metas = [json.load(open(j)) for j in glob.glob(f"{d[0]}/*.json")]
    fp = metas[0]["fingerprint"]
    rows.append({
        "encoder": e,
        "embed_dim": fp["embed_dim"],
        "image_size": fp["config"]["image_size"],
        "pool": fp["config"]["pool"],
        # store.py spreads `extra` into the manifest at top level, not nested.
        "fps_mean": round(np.mean([m["fps"] for m in metas if m.get("fps")]), 1),
        "peak_vram_mb": round(np.mean(
            [m["peak_vram_mb"] for m in metas if m.get("peak_vram_mb")]), 0),
        "norm_mean": round(np.mean([m["norm_mean"] for m in metas]), 2),
        "n_videos": len(metas),
    })
out += ["\n## Table 1 — encoders (measured)\n",
        pd.DataFrame(rows).to_markdown(index=False), "\n"]

# --- Table 2: reproduction gates -------------------------------------------
gates = {"dinov2_vits14": "reports/reproduction_gate.json",
         "clip_vitb16": "reports/reproduction_gate_clip.json",
         "endovit_vitb16": "reports/reproduction_gate_endovit.json",
         "biomedclip_vitb16": "reports/reproduction_gate_biomedclip.json"}
rows = []
for e, f in gates.items():
    if not Path(f).exists():
        continue
    g = json.load(open(f))
    ci = g["bootstrap_video_level"]["f1_ci"]
    rows.append({"encoder": e,
                 "linear_acc": round(g["linear"]["accuracy"], 3),
                 "linear_f1": round(g["linear"]["macro_f1"], 3),
                 "f1_ci": f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                 "knn_acc": round(g["knn"]["accuracy"], 3),
                 "knn_f1": round(g["knn"]["macro_f1"], 3),
                 "linear_knn_gap": round(g["linear"]["macro_f1"] - g["knn"]["macro_f1"], 3)})
out += ["\n## Table 2 — frame-level gates (40/8/32)\n",
        pd.DataFrame(rows).sort_values("linear_f1", ascending=False)
          .to_markdown(index=False), "\n"]

# --- Table 3: temporal at w=1 ----------------------------------------------
tc = json.load(open("reports/temporal_comparison.json"))
out += ["\n## Table 3 — temporal reliability (w=1)\n",
        pd.DataFrame(tc["aggregates"]).round(4).to_markdown(index=False),
        f"\n\nSpearman(acc, stability) = {tc['spearman_rho']}, "
        f"p = {tc['spearman_p']}\n"]

# --- Table 4/5: smoothing sweeps -------------------------------------------
for tag, f in [("centered", "reports/temporal_smoothing.json"),
               ("causal", "reports/temporal_smoothing_causal.json")]:
    df = pd.DataFrame(json.load(open(f)))
    piv = df.pivot(index="encoder", columns="window", values="accuracy").round(3)
    jit = df.pivot(index="encoder", columns="window", values="jitter_ratio").round(1)
    f1 = df.pivot(index="encoder", columns="window", values="macro_f1").round(3)
    out += [f"\n## Smoothing — {tag}\n", "\n### accuracy\n", piv.to_markdown(),
            "\n\n### macro-F1\n", f1.to_markdown(),
            "\n\n### jitter ratio\n", jit.to_markdown(), "\n"]

# --- flow coupling ----------------------------------------------------------
if Path("reports/flow_coupling.json").exists():
    fc = json.load(open("reports/flow_coupling.json"))
    out += ["\n## Flow coupling\n",
            pd.DataFrame(fc["aggregate"]).round(3).to_markdown(index=False), "\n"]

Path("reports/RESULTS.md").write_text("\n".join(out))
print("wrote reports/RESULTS.md")
