"""HAI 22.04: build runs_max/ from the max-score arrays saved by confirm_2204.py (run from the hai2204 working root).
Ridge, PCA and Isolation Forest keep their prespecified scores (protocol); persistence and the deep baselines are re-scored by
rescore_max_classical.py --only persistence and rescore_max_deep.py."""
import json, os, glob, shutil, numpy as np
os.makedirs("runs_max/train", exist_ok=True); os.makedirs("runs_max/baselines", exist_ok=True)
for d in sorted(glob.glob("runs/train/*__*")):
    n = os.path.basename(d); o = os.path.join("runs_max/train", n); os.makedirs(o, exist_ok=True)
    sm = json.load(open(os.path.join(d, "summary.json"))); z = np.load(os.path.join(d, "scores.npz"))
    pick = lambda k: {key[4:]: v for key, v in sm[k].items() if key.startswith("max_")}
    json.dump({"config": sm["config"], "variant": "max tail surprise (k=1, 30 s; max fusion at b0.5)", "tau": pick("tau"),
               "val_clean": pick("val_clean"), "dev": pick("dev"), "holdout": {}}, open(os.path.join(o, "summary.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(o, "scores.npz"), **{k[4:]: z[k] for k in z.files if k.startswith("max_")})
shutil.copy("runs/train/selected.json", "runs_max/train/selected.json")
for m in ("iforest", "ridge_a1.0", "pca_k20"): shutil.copytree(os.path.join("runs/baselines", m), os.path.join("runs_max/baselines", m), dirs_exist_ok=True)
shutil.copy("runs/baselines/dev_injections.json", "runs_max/baselines/")
print("runs_max built:", len(glob.glob("runs_max/train/*__*")), "runs")
