"""
TailGuard | Step 7: deep baselines (USAD, TranAD, GDN) with the authors' architectures and training settings,
under the common protocol (same split, eligibility, score policy, calibration, development injections).
Runs on CPU or GPU. Resumable: checkpoints every epoch, finished seeds are skipped.

  python scripts\\step7_deep_baselines.py --model USAD --smoke     (2-minute check)
  python scripts\\step7_deep_baselines.py --model all               (5 seeds x 3 models)
GDN needs:  pip install torch_geometric
Outputs: runs/deep/<MODEL>/s<seed>/{summary.json, scores.npz}
"""
import argparse, copy, json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from tailguard_lib import (load, ecdf_ref, tail_surprise, causal_smooth, calibrate_threshold, alarm_stats,
                          event_stats, apply_injections)
import deep_models as DM

OUT = os.path.join(ROOT, "runs", "deep_rescore")   # re-run that also keeps per-channel errors
SEEDS = [17, 29, 43, 71, 101]; W_MAX, TOPK, SMOOTH = 120, 3, 30
FRAC = 0.2           # random fraction of fitting windows drawn each epoch (neighbouring 1 s windows are near-duplicates)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 1)

# authors' settings (repositories pinned in deep_models.py)
SPEC = {
    "USAD":   {"K": 12, "past_only": False, "batch": 7919, "epochs": 100, "patience": 10},
    "TranAD": {"K": 10, "past_only": False, "batch": 128, "epochs": 5, "patience": 5},
    "GDN":    {"K": 5, "past_only": True, "batch": 32, "epochs": 30, "patience": 15},
}


class Prep:
    def __init__(self):
        tr, te = load("train"), load("test")
        self.nT = tr["Z"].shape[1]
        X = np.hstack([tr["Z"], tr["C"]]); Xte = np.hstack([te["Z"], te["C"]])     # continuous targets + set points
        ids = tr["df_ids"]; self.ids, self.idte = ids, te["df_ids"]
        self.part = ids["partition"].to_numpy(); self.elig = ids["eligible"].to_numpy()
        self.seg = ids["segment_id"].to_numpy(); self.pos = ids["pos_in_segment"].to_numpy()
        self.pos_te = self.idte["pos_in_segment"].to_numpy()
        fit = self.elig & (self.part == "fit")
        self.mn = X[fit].min(0); rngv = X[fit].max(0) - self.mn; self.rg = np.where(rngv > 0, rngv, 1.0)
        self.Z = tr["Z"]; self.C = tr["C"]
        self.X = self.mm(X); self.Xte = self.mm(Xte)                                # authors' min-max scaling (fit only)
        self.fit_idx = np.where(fit)[0]
        self.val_idx = np.where(self.elig & (self.part == "val"))[0]
        self.dev_events = json.load(open(os.path.join(ROOT, "runs", "baselines", "dev_injections.json")))
        print(f"deep baselines: {X.shape[1]} channels ({self.nT} scored), device {DEV}, threads {torch.get_num_threads()}")

    def mm(self, X):
        return ((X - self.mn) / self.rg).astype(np.float32)


def windows(X, idx, K, past_only):
    off = np.arange(-K, 0) if past_only else np.arange(-K + 1, 1)
    return X[idx[:, None] + off[None, :]]                                          # (B, K, F)


# ------------------------------------------------------------------ model adapters
class USADAdapter:
    def __init__(self, F_, s):
        self.K = s["K"]; self.F = F_; w = self.K * F_
        self.m = DM.UsadModel(w, self.K * 100).to(DEV)
        self.o1 = torch.optim.Adam(list(self.m.encoder.parameters()) + list(self.m.decoder1.parameters()))
        self.o2 = torch.optim.Adam(list(self.m.encoder.parameters()) + list(self.m.decoder2.parameters()))
    def state(self): return {"m": self.m.state_dict(), "o1": self.o1.state_dict(), "o2": self.o2.state_dict()}
    def load(self, s): self.m.load_state_dict(s["m"]); self.o1.load_state_dict(s["o1"]); self.o2.load_state_dict(s["o2"])
    def train_step(self, xb, ep):
        b = torch.as_tensor(xb.reshape(len(xb), -1), device=DEV); n = ep + 1
        l1, l2 = self.m.training_step(b, n); l1.backward(); self.o1.step(); self.o1.zero_grad()
        l1, l2 = self.m.training_step(b, n); l2.backward(); self.o2.step(); self.o2.zero_grad()
        return float(l1)
    @torch.no_grad()
    def val_loss(self, xb, ep):
        b = torch.as_tensor(xb.reshape(len(xb), -1), device=DEV); return float(self.m.validation_step(b, ep + 1)["val_loss1"])
    @torch.no_grad()
    def chan_err(self, xb):
        b = torch.as_tensor(xb.reshape(len(xb), -1), device=DEV)
        w1 = self.m.decoder1(self.m.encoder(b)); w2 = self.m.decoder2(self.m.encoder(w1))
        e = 0.5 * (b - w1) ** 2 + 0.5 * (b - w2) ** 2                                # alpha = beta = 0.5 (authors)
        return e.view(len(xb), self.K, self.F)[:, -1, :].cpu().numpy()
    def eval(self): self.m.eval()
    def train(self): self.m.train()
    def epoch_end(self): pass


class TranADAdapter:
    def __init__(self, F_, s):
        self.K = s["K"]; self.F = F_
        self.m = DM.TranAD(F_).to(DEV)                    # lr = deep_models.lr = 0.008, the authors' SWaT setting (closest ICS set)
        self.opt = torch.optim.AdamW(self.m.parameters(), lr=self.m.lr, weight_decay=1e-5)
        self.sch = torch.optim.lr_scheduler.StepLR(self.opt, 5, 0.9)
        self.l = nn.MSELoss(reduction="none")
    def state(self): return {"m": self.m.state_dict(), "o": self.opt.state_dict(), "s": self.sch.state_dict()}
    def load(self, s): self.m.load_state_dict(s["m"]); self.opt.load_state_dict(s["o"]); self.sch.load_state_dict(s["s"])
    def _fw(self, xb):
        w = torch.as_tensor(xb, device=DEV).permute(1, 0, 2); elem = w[-1, :, :].view(1, w.shape[1], self.F)
        return self.m(w, elem), elem
    def train_step(self, xb, ep):
        (z0, z1), elem = self._fw(xb); n = ep + 1
        loss = torch.mean((1 / n) * self.l(z0, elem) + (1 - 1 / n) * self.l(z1, elem))
        self.opt.zero_grad(); loss.backward(); self.opt.step(); return float(loss)
    @torch.no_grad()
    def val_loss(self, xb, ep):
        (z0, z1), elem = self._fw(xb); return float(torch.mean(self.l(z1, elem)))
    @torch.no_grad()
    def chan_err(self, xb):
        (z0, z1), elem = self._fw(xb); return self.l(z1, elem)[0].cpu().numpy()   # authors' test score uses phase-2 output
    def eval(self): self.m.eval()
    def train(self): self.m.train()
    def epoch_end(self): self.sch.step()


class GDNAdapter:
    def __init__(self, F_, s):
        self.K = s["K"]; self.F = F_
        src = [j for i in range(F_) for j in range(F_) if i != j]; dst = [i for i in range(F_) for j in range(F_) if i != j]
        self.ei = torch.tensor([src, dst], dtype=torch.long, device=DEV)            # fully connected prior (authors' default)
        self.m = DM.GDN([self.ei], F_, dim=64, out_layer_inter_dim=128, input_dim=self.K, out_layer_num=1, topk=5).to(DEV)
        self.opt = torch.optim.Adam(self.m.parameters(), lr=1e-3, weight_decay=0)
    def state(self): return {"m": self.m.state_dict(), "o": self.opt.state_dict()}
    def load(self, s): self.m.load_state_dict(s["m"]); self.opt.load_state_dict(s["o"])
    def _fw(self, xb, yb):
        x = torch.as_tensor(xb, device=DEV).permute(0, 2, 1).contiguous()           # (B, nodes, window)
        return self.m(x, self.ei), torch.as_tensor(yb, device=DEV)
    def train_step(self, xb, ep, yb=None):
        p, y = self._fw(xb, yb); loss = F.mse_loss(p, y)
        self.opt.zero_grad(); loss.backward(); self.opt.step(); return float(loss)
    @torch.no_grad()
    def val_loss(self, xb, ep, yb=None):
        p, y = self._fw(xb, yb); return float(F.mse_loss(p, y))
    @torch.no_grad()
    def chan_err(self, xb, yb=None):
        p, y = self._fw(xb, yb); return (p - y).abs().cpu().numpy()
    def eval(self): self.m.eval()
    def train(self): self.m.train()
    def epoch_end(self): pass


ADAPT = {"USAD": USADAdapter, "TranAD": TranADAdapter, "GDN": GDNAdapter}


# ------------------------------------------------------------------ one model, one seed
def run(name, seed, P, smoke=False):
    d = os.path.join(OUT if not smoke else OUT + "_smoke", name, f"s{seed}"); os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "summary.json")): print(f"  {name} s{seed}: done"); return
    s = dict(SPEC[name]); K, past = s["K"], s["past_only"]
    if smoke: s["epochs"] = 1
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    A = ADAPT[name](P.X.shape[1], s); is_gdn = name == "GDN"
    st = {"epoch": 0, "best": np.inf, "bad": 0, "best_state": None, "history": []}
    ck = os.path.join(d, "ckpt.pt")
    if os.path.exists(ck):
        c = torch.load(ck, map_location=DEV, weights_only=False); A.load(c["A"]); st = c["st"]; rng.bit_generator.state = c["rng"]
        print(f"  {name} s{seed}: resumed at epoch {st['epoch']}")
    vsub = P.val_idx[::5]; t0 = time.time()
    frac = 0.01 if smoke else FRAC
    while st["epoch"] < s["epochs"] and st["bad"] < s["patience"]:
        A.train(); ep = st["epoch"]
        idx = rng.permutation(P.fit_idx)[: int(frac * len(P.fit_idx))]
        for k in range(0, len(idx), s["batch"]):
            b = idx[k:k + s["batch"]]
            if len(b) < 2: continue
            xb = windows(P.X, b, K, past)
            A.train_step(xb, ep, P.X[b]) if is_gdn else A.train_step(xb, ep)
        A.epoch_end(); A.eval(); vl = []
        for k in range(0, len(vsub), 2048):
            b = vsub[k:k + 2048]; xb = windows(P.X, b, K, past)
            vl.append(A.val_loss(xb, ep, P.X[b]) if is_gdn else A.val_loss(xb, ep))
        v = float(np.mean(vl)); st["epoch"] += 1; st["history"].append(v)
        if v < st["best"] - 1e-9:
            st["best"], st["bad"] = v, 0; st["best_state"] = copy.deepcopy(A.state())
        else:
            st["bad"] += 1
        torch.save({"A": A.state(), "st": st, "rng": rng.bit_generator.state}, ck)
        print(f"  {name} s{seed} epoch {st['epoch']:3d} val loss {v:.5f} (best {st['best']:.5f}) {time.time()-t0:.0f}s", flush=True)
    A.load(st["best_state"]); A.eval()
    summ = score(name, A, P, K, past, is_gdn); summ.update(seed=seed, epochs=st["epoch"], history=st["history"],
                                                         settings=s, train_seconds=time.time() - t0)
    arr = summ.pop("_arrays"); np.savez_compressed(os.path.join(d, "chan.npz"), **summ.pop("_chan"))
    np.savez_compressed(os.path.join(d, "scores.npz"), **arr)
    json.dump(summ, open(os.path.join(d, "summary.json"), "w"), indent=1, default=float)
    if os.path.exists(ck): os.remove(ck)
    print(f"  {name} s{seed}: tau {summ['tau']:.3f}  dev recall {summ['dev']['event_recall']:.3f}  "
          f"val FA24 {summ['val_clean']['FA24']:.2f}  ({summ['train_seconds']/60:.1f} min)")


def errors(A, X, rows, K, past, is_gdn, n, nT, bs=4096):
    if is_gdn: bs = 256                                    # GDN builds per-edge tensors: small batches bound memory
    E = np.zeros((n, nT), np.float32)
    for k in range(0, len(rows), bs):
        b = rows[k:k + bs]; xb = windows(X, b, K, past)
        e = A.chan_err(xb, X[b]) if is_gdn else A.chan_err(xb)
        E[b] = e[:, :nT]
    return E


def stream(E, pos, ref, chunk=100_000):
    p = np.concatenate([np.sort(tail_surprise(E[k:k + chunk], ref), 1)[:, -TOPK:].mean(1) for k in range(0, len(E), chunk)])  # chunked: bounded memory
    p[pos < W_MAX] = 0
    return causal_smooth(p, SMOOTH)


def score(name, A, P, K, past, is_gdn):
    import resource; mem = lambda t: print(f'    [{t}] peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB', flush=True)
    n, nT = len(P.X), P.nT
    vc = np.where(P.elig & np.isin(P.part, ["val", "cal"]))[0]
    E = errors(A, P.X, vc, K, past, is_gdn, n, nT)
    rows_te = np.where(P.pos_te >= W_MAX)[0]
    mem('E')
    Ete = errors(A, P.Xte, rows_te, K, past, is_gdn, len(P.Xte), nT)
    valm = P.elig & (P.part == "val"); calm = P.elig & (P.part == "cal")
    mem('Ete')
    ref = ecdf_ref(E[valm])
    s = stream(E, P.pos, ref); ste = stream(Ete, P.pos_te, ref)
    tau, _ = calibrate_threshold(s[calm], P.seg[calm], calm.sum() / 3600)
    mem('streams')
    # development injections: same sensor manipulations as every other method
    Zi, evid = apply_injections(P.Z, P.dev_events)
    Xi = P.mm(np.hstack([Zi, P.C]))
    Ei = errors(A, Xi, np.where(valm)[0], K, past, is_gdn, n, nT)
    del Xi, Zi
    si = stream(Ei, P.pos, ref)
    mem('Ei')
    a = si[valm] > tau; ev = evid[valm]
    return {"tau": float(tau),
            "val_clean": alarm_stats(s[valm] > tau, P.seg[valm], np.ones(valm.sum(), bool)),
            "dev": {**event_stats(a, ev), "FA24": alarm_stats(a, P.seg[valm], ev == "")["FA24"]},
            "_arrays": {"s_train": s.astype(np.float32), "s_test": ste.astype(np.float32)},
            "_chan": {"E": E[np.isin(np.arange(n), vc)].astype(np.float32), "rows": vc, "Ete": Ete, "Ei": Ei[valm].astype(np.float32)}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="all"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); P = Prep()
    models = ["USAD", "TranAD", "GDN"] if a.model == "all" else [a.model]
    for m in models:
        for sd in (SEEDS[:1] if a.smoke else SEEDS):
            print(f"[{m} seed {sd}]"); run(m, sd, P, smoke=a.smoke)
    print("DEEP BASELINES COMPLETE" if not a.smoke else "SMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()
