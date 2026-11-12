"""Recompute per-channel forecast errors and constraint residuals for a trained model:
clean training-file data (fit/val/cal), development-injected copy, and test files. No labels are read."""
import sys, os, numpy as np, torch
ROOT = sys.argv[1]; MODEL = sys.argv[2]; OUTF = sys.argv[3]; W = int(sys.argv[4]) if len(sys.argv) > 4 else 30
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tailguard_train as T
torch.set_num_threads(2)
D = T.Data()
G = {k: torch.as_tensor(v) for k, v in {"X": D.X, "Z": D.Z, "Xte": D.Xte, "Zte": D.Zte}.items()}
m = T.GRUForecaster(D.nF, D.nT); m.load_state_dict(torch.load(MODEL, map_location="cpu")); m.eval()
raw, rawte = D.raw(D.Z), D.raw(D.Zte)
V, Vte = D.resid_obs(raw, D.B_terms(raw)), D.resid_obs(rawte, D.B_terms(rawte))
E = T.predict_errors(m, G["X"], G["Z"], W, len(D.Z)); print("E done", flush=True)
Ete = T.predict_errors(m, G["Xte"], G["Zte"], W, len(D.Zte)); print("Ete done", flush=True)
Zi, evid = T.apply_injections(D.Z, D.dev_events); Xi = D.X.copy(); Xi[:, :D.nT] = Zi
Ei = T.predict_errors(m, torch.as_tensor(Xi), torch.as_tensor(Zi), W, len(D.Z)); print("Ei done", flush=True)
rawi = D.raw(Zi); Vi = D.resid_obs(rawi, D.B_terms(rawi))
np.savez_compressed(OUTF, E=E.astype(np.float32), Ete=Ete.astype(np.float32), Ei=Ei.astype(np.float32),
                    V=V, Vte=Vte, Vi=Vi, evid=evid, part=D.part, reg=D.reg, elig=D.elig, seg=D.seg, pos=D.pos,
                    pos_te=D.pos_te, seg_te=D.seg_te)
print("saved", OUTF)
