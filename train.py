"""Phase 1: dense.  Phase 2 arms from the same checkpoint on identical batches.
python train.py CONFIG SEED ARM   (ARM = phase1 | dense_continue | moe_drop | clone_hard | clone_bias | clone_noise | clone_both)
Outputs go to $OUT (default runs/<config-name>)/seed<k>/."""
import json, os, sys, time, math, torch, numpy as np
from model import LM, Cfg, param_counts
from upcycle import upcycle
import data as D

ARMS = {  # arm -> (conversion mode, bias balancing on, probabilistic selection on)
    "moe_drop": ("drop", True, True), "clone_hard": ("clone", False, False), "clone_bias": ("clone", True, False),
    "clone_noise": ("clone", False, True), "clone_both": ("clone", True, True)}
DEV = "cuda" if torch.cuda.is_available() else "cpu"

class Ctx:
    def __init__(s, cfg_path, seed):
        s.cfg = json.load(open(cfg_path)); s.name = os.path.splitext(os.path.basename(cfg_path))[0]
        s.A = s.cfg["train"]; s.mc = s.cfg["model"]; s.seed = seed
        s.out = os.path.join(os.environ.get("OUT", f"runs/{s.name}"), f"seed{seed}"); os.makedirs(s.out, exist_ok=True)
        tr, va = D.build(s.cfg["data"], s.mc["vocab"], os.environ.get("DATA", "data"))
        s.tr = torch.from_numpy(tr.astype(np.int64)); s.va = torch.from_numpy(va.astype(np.int64))
        s.VAL = [s.batch(0, "val", i) for i in range(s.A["eval_batches"])]
    def model_cfg(s, moe=False): return Cfg(**{**s.mc, "moe": moe})
    def batch(s, step, split="train", i=0):
        d = s.tr if split == "train" else s.va; T = s.mc["seq"]
        g = np.random.default_rng((s.seed, step, i) if split == "train" else (10**9, i))   # val identical for all seeds
        ix = torch.from_numpy(g.integers(0, len(d) - T - 1, s.A["batch"]))
        off = ix[:, None] + torch.arange(T + 1)[None]
        b = d[off]; return b[:, :-1].to(DEV, non_blocking=True), b[:, 1:].to(DEV, non_blocking=True)
    def amp(s): return torch.autocast(DEV, dtype=torch.bfloat16, enabled=bool(s.A.get("bf16")) and DEV == "cuda")
    @torch.no_grad()
    def evaluate(s, m):
        m.eval(); tot = 0.0
        with s.amp():
            for x, y in s.VAL: tot += m(x, y).item()
        m.train(); return tot / len(s.VAL)
    def lr_at(s, g):
        A = s.A; total = A["p1"] + A["p2"]; ds = int(total * (1 - A["decay_frac"]))
        if g < A["warmup"]: return A["lr"] * (g + 1) / A["warmup"]
        return A["lr"] if g < ds else A["lr"] * max(0.0, (total - g) / (total - ds))

def make_opt(m, A):
    dec = [p for p in m.parameters() if p.dim() >= 2]; nod = [p for p in m.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": dec, "weight_decay": A["wd"]}, {"params": nod, "weight_decay": 0.0}],
                             lr=A["lr"], betas=(0.9, 0.95), fused=DEV == "cuda")

def moe_stats(m):
    out = []
    for f in m.moe_layers():
        c = f.win.clone(); f.win.zero_(); fair = c.sum() / len(c)
        out.append(dict(maxvio=round(((c.max() - fair) / fair).item(), 3), dead=int((c < 0.1 * fair).sum()),
                        frac=[round(v, 4) for v in (c / c.sum()).tolist()], bias=[round(v, 3) for v in f.bias.tolist()]))
    return out

def transfer_adam(old_opt, old_m, new_opt, new_m):
    old = dict(old_m.named_parameters()); n = 0
    for k, p in new_m.named_parameters():
        if k in old and old[k].shape == p.shape and old[k] in old_opt.state:
            new_opt.state[p] = {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in old_opt.state[old[k]].items()}; n += 1
    return n

def run(C, m, opt, start, steps, name, bias_on=False, noise_on=False):
    A = C.A; log = []; moe = bool(m.moe_layers()); t0 = time.time(); ntok = A["batch"] * C.mc["seq"]
    for s in range(start, start + steps):
        for gp in opt.param_groups: gp["lr"] = C.lr_at(s)
        if moe:
            T = A["noise_T0"] * max(0.0, 1 - (s - start) / A["noise_steps"]) if noise_on else 0.0
            for f in m.moe_layers(): f.noise_T = T
        x, y = C.batch(s)
        with C.amp(): loss = m(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if moe and bias_on and (s + 1) % A["bias_every"] == 0:
            for f in m.moe_layers(): f.update_bias(A["gamma"])
        rec = dict(step=s, loss=round(loss.item(), 4))
        if (s + 1) % A["eval_every"] == 0 or s == start:
            rec["val"] = round(C.evaluate(m), 4)
            if moe: rec["moe"] = moe_stats(m)
            el = time.time() - t0; rec["tok_per_s"] = round(ntok * (s - start + 1) / el)
            print(f"[seed{C.seed}] {name} step {s} loss {rec['loss']} val {rec['val']} {rec['tok_per_s']} tok/s", flush=True)
        log.append(rec)
    return log, time.time() - t0

def save_json(obj, path):
    json.dump(obj, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)

def phase1(C):
    torch.manual_seed(C.seed); m = LM(C.model_cfg()).to(DEV); opt = make_opt(m, C.A)
    log, wall = run(C, m, opt, 0, C.A["p1"], "dense_p1")
    torch.save({"model": m.state_dict(), "opt": opt.state_dict()}, f"{C.out}/dense_p1.pt.tmp")
    os.replace(f"{C.out}/dense_p1.pt.tmp", f"{C.out}/dense_p1.pt")
    save_json(dict(params=param_counts(m), wall_s=wall, log=log), f"{C.out}/phase1.json")

def load_dense(C):
    ck = torch.load(f"{C.out}/dense_p1.pt", map_location=DEV)
    d = LM(C.model_cfg()).to(DEV); d.load_state_dict(ck["model"]); o = make_opt(d, C.A); o.load_state_dict(ck["opt"])
    return d, o

def phase2(C, arm):
    torch.manual_seed(C.seed + 1000)
    dense, dopt = load_dense(C); info = {"val_before": C.evaluate(dense)}
    if arm == "dense_continue":
        m, opt, bias_on, noise_on = dense, dopt, False, False
    else:
        mode, bias_on, noise_on = ARMS[arm]
        m = upcycle(dense.cpu(), C.model_cfg(True), mode=mode, r=C.A["r"], seed=C.seed).to(DEV); dense.to(DEV)
        opt = make_opt(m, C.A); info["adam_states_transferred"] = transfer_adam(dopt, dense, opt, m)
        info["val_after_convert"] = C.evaluate(m)
    print(f"[seed{C.seed}] {arm} {info}", flush=True)
    log, wall = run(C, m, opt, C.A["p1"], C.A["p2"], arm, bias_on, noise_on)
    save_json(dict(params=param_counts(m), conversion=info, wall_s=wall, log=log), f"{C.out}/{arm}.json")

if __name__ == "__main__":
    C = Ctx(sys.argv[1], int(sys.argv[2])); arm = sys.argv[3]
    phase1(C) if arm == "phase1" else phase2(C, arm)
