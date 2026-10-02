"""Aggregate all seeds -> summary.json + plots. README numbers come only from summary.json."""
import json, os, sys, glob, numpy as np, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cfg = sys.argv[1]; conf = json.load(open(cfg)); name = os.path.splitext(os.path.basename(cfg))[0]
root = os.environ.get("OUT", f"runs/{name}"); arms = conf["arms"]; p1 = conf["train"]["p1"]
# only seeds where every arm finished (runs may still be in progress)
seeds = [s for s in conf["seeds"] if all(os.path.exists(f"{root}/seed{s}/{a}.json") for a in ["phase1", "conversion_fidelity"] + arms)]
assert seeds, "no seed has all arms complete"
L = lambda s, f: json.load(open(f"{root}/seed{s}/{f}.json"))
P1 = {s: L(s, "phase1") for s in seeds}; R = {a: {s: L(s, a) for s in seeds} for a in arms}
FID = {s: L(s, "conversion_fidelity") for s in seeds}
vals = lambda log: [(r["step"], r["val"]) for r in log if "val" in r]
r4 = lambda x: round(float(x), 4)
def ms(xs): return {"mean": r4(np.mean(xs)), "std": r4(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, "per_seed": [r4(x) for x in xs]}
def regain(log, tgt):
    for r in log:
        if "val" in r and r["val"] <= tgt: return r["step"] - p1
    return None
S = {"config": name, "seeds": seeds, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
     "params_dense": P1[seeds[0]]["params"], "params_moe": R[arms[1]][seeds[0]]["params"] if len(arms) > 1 else None,
     "tokens_per_step": conf["train"]["batch"] * conf["model"]["seq"], "steps": [p1, conf["train"]["p2"]],
     "val_phase1_start": ms([vals(P1[s]["log"])[0][1] for s in seeds]),
     "val_phase1_end": ms([vals(P1[s]["log"])[-1][1] for s in seeds]),
     "fidelity": {k: ms([FID[s][k] for s in seeds]) for k in FID[seeds[0]]},
     "wall_min_total": r4(sum(P1[s]["wall_s"] for s in seeds) / 60 + sum(R[a][s]["wall_s"] for a in arms for s in seeds) / 60),
     "arms": {}}
for a in arms:
    d = {"val_first": ms([vals(R[a][s]["log"])[0][1] for s in seeds]),
         "val_final": ms([vals(R[a][s]["log"])[-1][1] for s in seeds]),
         "steps_to_regain": [regain(R[a][s]["log"], vals(P1[s]["log"])[-1][1]) for s in seeds],
         "tok_per_s": ms([[r for r in R[a][s]["log"] if "tok_per_s" in r][-1]["tok_per_s"] for s in seeds])}
    ev = {s: [r for r in R[a][s]["log"] if "moe" in r] for s in seeds}
    if ev[seeds[0]]:
        d["final_maxvio_mean"] = ms([np.mean([l["maxvio"] for l in ev[s][-1]["moe"]]) for s in seeds])
        d["final_dead_total"] = [sum(l["dead"] for l in ev[s][-1]["moe"]) for s in seeds]
        d["max_dead_any_eval"] = [max(sum(l["dead"] for l in r["moe"]) for r in ev[s][1:]) for s in seeds]
        d["n_experts_total"] = conf["model"]["n_exp"] * conf["model"]["n_layer"]
    S["arms"][a] = d
diff = [vals(R["moe_drop"][s]["log"])[-1][1] - vals(R["dense_continue"][s]["log"])[-1][1] for s in seeds]
S["moe_drop_minus_dense"] = {**ms(diff), "seeds_moe_better": int(sum(x < 0 for x in diff))}
# first eval step after which moe_drop stays below dense (seed-mean)
steps = [st for st, _ in vals(R["dense_continue"][seeds[0]]["log"])]
mm = {a: np.mean([[v for _, v in vals(R[a][s]["log"])] for s in seeds], 0) for a in arms}
below = mm["moe_drop"] < mm["dense_continue"]; cross = None
for i in range(len(steps)):
    if below[i:].all(): cross = steps[i]; break
S["moe_drop_stays_below_dense_from_step"] = cross
json.dump(S, open("summary.json", "w"), indent=1)

col = dict(dense_continue="gray", moe_drop="tab:blue", clone_hard="tab:red", clone_bias="tab:orange", clone_noise="tab:purple", clone_both="tab:green")
fig, ax = plt.subplots(1, 3, figsize=(19, 5))
v1 = np.mean([[v for _, v in vals(P1[s]["log"])] for s in seeds], 0); s1 = [st for st, _ in vals(P1[seeds[0]]["log"])]
ax[0].plot(s1, v1, "k.-", label="dense phase 1")
for a in arms: ax[0].plot(steps, mm[a], ".-", color=col.get(a), label=a)
ax[0].axvline(p1, c="k", ls="--", lw=.8); ax[0].set_title(f"val loss, mean of {len(seeds)} seeds (dashed = conversion)")
ax[0].set_ylim(min(mm[a].min() for a in arms) - .05, v1[len(v1) // 4] + .1); ax[0].legend(); ax[0].set_xlabel("step")
for a in arms:
    lo = np.min([[v for _, v in vals(R[a][s]["log"])] for s in seeds], 0) - mm["dense_continue"]
    hi = np.max([[v for _, v in vals(R[a][s]["log"])] for s in seeds], 0) - mm["dense_continue"]
    ax[1].plot(steps, mm[a] - mm["dense_continue"], ".-", color=col.get(a), label=a)
    if a != "dense_continue": ax[1].fill_between(steps, lo, hi, color=col.get(a), alpha=.12)
ax[1].axhline(0, c="k", lw=.8); ax[1].set_ylim(-0.05, 0.2); ax[1].set_title("val loss minus dense_continue (band = seed min/max)")
ax[1].legend(); ax[1].set_xlabel("step")
for a in arms:
    if a == "dense_continue": continue
    xs = [r["step"] for r in R[a][seeds[0]]["log"] if "moe" in r][1:]
    mv = np.mean([[np.mean([l["maxvio"] for l in r["moe"]]) for r in R[a][s]["log"] if "moe" in r][1:] for s in seeds], 0)
    ax[2].plot(xs, mv, ".-", color=col.get(a), label=a)
ax[2].set_title("expert load imbalance MaxVio (0 = even)"); ax[2].set_xlabel("step"); ax[2].legend()
plt.tight_layout(); plt.savefig("loss_curves.png", dpi=110)
moe_arms = [a for a in arms if a != "dense_continue"]; nl = conf["model"]["n_layer"]
fig, ax = plt.subplots(len(moe_arms), nl, figsize=(2.2 * nl, 1.9 * len(moe_arms)), squeeze=False)
for i, a in enumerate(moe_arms):
    last = [r for r in R[a][seeds[0]]["log"] if "moe" in r][-1]["moe"]
    for l, st in enumerate(last):
        ax[i][l].bar(range(len(st["frac"])), st["frac"], color=col.get(a)); ax[i][l].axhline(1 / len(st["frac"]), c="k", ls="--", lw=.7)
        ax[i][l].set_ylim(0, .6); ax[i][l].set_xticks([]); ax[i][l].set_title(f"{a} L{l} dead={st['dead']}", fontsize=7)
plt.tight_layout(); plt.savefig("expert_load.png", dpi=110)
print(json.dumps(S, indent=1))
