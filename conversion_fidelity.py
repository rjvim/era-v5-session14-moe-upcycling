"""Val loss of the trained dense model vs each conversion variant at step 0 (no training)."""
import json, sys, torch
from train import Ctx, load_dense, DEV
from upcycle import upcycle, partition_identity
def fidelity(C):
    torch.manual_seed(C.seed + 2000); d, _ = load_dense(C); d = d.cpu(); ev = lambda m: round(C.evaluate(m.to(DEV)), 4)
    res = {"partition_all_experts_on": ev(partition_identity(d, C.model_cfg())), "dense": ev(d.cpu())}
    d = d.cpu(); mc = C.model_cfg(True); r = C.A["r"]
    for name, kw in [("drop_r_w2zero", dict(mode="drop", r=r)), ("drop_r_w2normal", dict(mode="drop", r=r, redraw_w2="normal")),
                     ("drop_r0_subset_copy", dict(mode="drop", r=0.0)), ("clone", dict(mode="clone"))]:
        res[name] = ev(upcycle(d, mc, seed=C.seed, **kw))
    json.dump(res, open(f"{C.out}/conversion_fidelity.json", "w"), indent=1); print(f"[seed{C.seed}] fidelity {res}", flush=True)
if __name__ == "__main__": fidelity(Ctx(sys.argv[1], int(sys.argv[2])))
