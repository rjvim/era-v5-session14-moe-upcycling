"""Resumable driver: every (seed, arm) whose JSON exists is skipped, so a Colab disconnect costs at most one arm."""
import json, os, sys, time, gc, torch
from train import Ctx, phase1, phase2
from conversion_fidelity import fidelity
cfg = sys.argv[1]; conf = json.load(open(cfg)); t0 = time.time()
for seed in conf["seeds"]:
    C = Ctx(cfg, seed)
    if not os.path.exists(f"{C.out}/phase1.json"): phase1(C)
    if not os.path.exists(f"{C.out}/conversion_fidelity.json"): fidelity(C)
    for arm in conf["arms"]:
        if not os.path.exists(f"{C.out}/{arm}.json"):
            phase2(C, arm); gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"== done seed{seed} {arm}  elapsed {(time.time()-t0)/60:.1f} min", flush=True)
