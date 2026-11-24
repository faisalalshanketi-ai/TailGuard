"""NumPy re-implementation of the TailGuard forecaster (scripts/tailguard_train.py: GRUForecaster) for CPU analyses that
should not need PyTorch. Loads model_best.pt (a PyTorch zip state_dict) without importing torch and evaluates the
two-layer GRU with the PyTorch gate convention (r, z, n). analysis/scripts/fault_type_selection.py checks that it
reproduces the development-fault detections stored by the training runs."""
import io, pickle, zipfile
import numpy as np

_DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32, "HalfStorage": np.float16}


def load_state(path):
    z = zipfile.ZipFile(path); pre = z.namelist()[0].split("/")[0]

    def rebuild(storage, offset, size, stride, *args):
        n = int(np.prod(size)) if size else 1
        a = storage[offset:offset + n] if tuple(stride) == tuple(np.cumprod((list(size)[1:] + [1])[::-1])[::-1]) or n <= 1 else None
        if a is None:
            a = np.lib.stride_tricks.as_strided(storage[offset:], shape=size, strides=[s * storage.itemsize for s in stride]).copy()
        return a.reshape(size)

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2": return rebuild
            if name.endswith("Storage"): return name
            if mod == "collections" and name == "OrderedDict":
                import collections; return collections.OrderedDict
            if mod == "torch._utils" and name == "_rebuild_parameter": return lambda t, *a: t
            raise pickle.UnpicklingError(f"{mod}.{name}")

        def persistent_load(self, pid):
            _, stype, key, _, _ = pid
            return np.frombuffer(z.read(f"{pre}/data/{key}"), dtype=_DT[stype if isinstance(stype, str) else stype.__name__])

    return {k: np.asarray(v, np.float32) for k, v in U(io.BytesIO(z.read(f"{pre}/data.pkl"))).load().items()}


def _sig(x): return 1.0 / (1.0 + np.exp(-x))


def forecast(S, X, rows, W, layers=2, batch=4096):
    """Predict the targets at `rows` from the W preceding rows of X (same windows as tailguard_train.windows)."""
    H = S["gru.weight_hh_l0"].shape[1]; out = np.empty((len(rows), S["head.weight"].shape[0]), np.float32)
    lo = max(int(rows.min()) - W, 0); hi = int(rows.max())
    G0 = X[lo:hi] @ S["gru.weight_ih_l0"].T + S["gru.bias_ih_l0"]          # layer-1 input projection, once per row
    for b in range(0, len(rows), batch):
        r = rows[b:b + batch]; idx = (r[:, None] - W + np.arange(W)[None, :]) - lo
        gi = G0[idx]                                                       # (B, W, 3H)
        for l in range(layers):
            Whh, bhh = S[f"gru.weight_hh_l{l}"], S[f"gru.bias_hh_l{l}"]
            h = np.zeros((len(r), H), np.float32); hs = np.empty((len(r), W, H), np.float32) if l < layers - 1 else None
            for t in range(W):
                gh = h @ Whh.T + bhh; a = gi[:, t]
                rg = _sig(a[:, :H] + gh[:, :H]); zg = _sig(a[:, H:2 * H] + gh[:, H:2 * H])
                n = np.tanh(a[:, 2 * H:] + rg * gh[:, 2 * H:]); h = (1 - zg) * n + zg * h
                if hs is not None: hs[:, t] = h
            if l < layers - 1:
                gi = hs @ S[f"gru.weight_ih_l{l + 1}"].T + S[f"gru.bias_ih_l{l + 1}"]
        out[b:b + len(r)] = h @ S["head.weight"].T + S["head.bias"]
    return out
