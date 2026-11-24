"""
TailGuard | HAI 22.04 confirmation runs (protocol frozen in manifests/HAI2204_PROTOCOL.md; GPU recommended, resumable).
Run from the HAI 22.04 repository root with TAILGUARD_PREFIX=hai2204 after Steps 2-4 of that root:
  TAILGUARD_PREFIX=hai2204 python scripts/confirm_2204.py --part 0 --parts 2 --max-hours 11
Trains TailGuard, the matched GRU and A1 with the hyperparameters frozen on HAI 23.05 (5 seeds each) and saves the
prespecified and max tail-surprise scores. Test labels are never read here.
Outputs: runs/train/<run_id>/... (TailGuard seed 17 is named search__ so that step6_evaluate.py finds it).
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
assert os.environ.get("TAILGUARD_PREFIX") == "hai2204", "set TAILGUARD_PREFIX=hai2204"
import tailguard_train as T
import holdout_a7_runs as RR                      # installs the two-score summariser (prespecified + max)
T.OUT = os.path.join(T.ROOT, "runs", "train"); os.makedirs(T.OUT, exist_ok=True)
FROZEN = {"W": 30, "l_inv": 1.0, "l_phys": 0.1,
          "rule": "frozen on HAI 23.05 (manifests/HAI2204_PROTOCOL.md); no selection on HAI 22.04"}


def plan():
    W, li, lp = FROZEN["W"], FROZEN["l_inv"], FROZEN["l_phys"]; P = []
    for s in T.SEEDS:
        P += [T.mk("search" if s == 17 else "tailguard", W, li, lp, s), T.mk("gru_erm", W, 0.0, 0.0, s, balanced=False),
              T.mk("A1_nophys", W, li, 0.0, s)]
    return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--part", type=int, default=0); ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--max-hours", type=float, default=0.0); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    if a.smoke: T.MAX_EPOCHS = 1
    json.dump(FROZEN, open(os.path.join(T.OUT, "selected.json"), "w"), indent=1)
    P = plan()[a.part::a.parts][: 1 if a.smoke else None]
    D = T.Data(); G = RR.tensors(D)
    for i, cfg in enumerate(P):
        if os.path.exists(os.path.join(T.OUT, cfg["id"], "summary.json")): continue
        if a.max_hours and (time.time() - t0) / 3600 > a.max_hours:
            print("TIME BUDGET REACHED - run the same command again to continue"); return
        print(f"[{i+1}/{len(P)}] {cfg['id']}", flush=True); T.run(cfg, D, G)
    print("HAI 22.04 CONFIRMATION RUNS COMPLETE (this part)")


if __name__ == "__main__":
    main()
