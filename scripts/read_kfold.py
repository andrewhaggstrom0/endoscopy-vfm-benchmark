"""Pretty-print a kfold_paired_*.json into readable tables."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

f = sys.argv[1] if len(sys.argv) > 1 else "reports/kfold_paired_causal.json"
d = json.loads(Path(f).read_text())
print(f"{f}: k={d['k']} seed={d['seed']} n_videos={d['n_videos']}\n")

# mean accuracy per encoder per window, from per_video
pv = d["per_video"]
rows = []
for enc, by_w in pv.items():
    for w, by_v in by_w.items():
        accs = [v[0] for v in by_v.values()]
        f1s = [v[1] for v in by_v.values()]
        rows.append({"encoder": enc, "window": int(w),
                     "acc": sum(accs)/len(accs), "macro_f1": sum(f1s)/len(f1s)})
m = pd.DataFrame(rows)
print("=== mean accuracy (all 80 videos) ===")
print(m.pivot(index="encoder", columns="window", values="acc").round(3).to_string())
print("\n=== mean macro-F1 ===")
print(m.pivot(index="encoder", columns="window", values="macro_f1").round(3).to_string())

# pairwise tests
r = pd.DataFrame(d["results"])
acc = r[r.metric == "accuracy"].copy()
acc["sig"] = acc.p < 0.05
acc["CI"] = acc.apply(lambda x: f"[{x.ci_low:+.4f},{x.ci_high:+.4f}]", axis=1)
acc["wins"] = acc.n_videos_A_wins.astype(str) + "/" + acc.n_videos.astype(str)
print("\n=== pairwise paired bootstrap (accuracy) ===")
for pair in acc.pair.unique():
    s = acc[acc.pair == pair].sort_values("window")
    print(f"\n{pair.replace(':', '  minus  ')}")
    print(s[["window", "mean_diff", "CI", "p", "wins", "sig"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

n_sig = int(acc.sig.sum())
print(f"\n{n_sig} of {len(acc)} accuracy tests significant at p<0.05")
print(f"(expect ~{0.05*len(acc):.1f} by chance alone -- 6 pairs x 5 windows)")
