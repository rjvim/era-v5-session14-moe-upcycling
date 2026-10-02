"""README is generated from summary.json so prose numbers cannot drift from artifacts."""
import json
S = json.load(open("summary.json")); A = S["arms"]; n = len(S["seeds"])
f = lambda d: f"{d['mean']:.4f} ± {d['std']:.4f}" if len(d["per_seed"]) > 1 else f"{d['mean']:.4f}"
pd, pm = S["params_dense"], S["params_moe"]
diff = S["moe_drop_minus_dense"]
verdict = ("single seed — direction only, not a claim" if n == 1 else
           "MoE better than dense on every seed" if diff["seeds_moe_better"] == n else
           "MoE worse than dense on every seed" if diff["seeds_moe_better"] == 0 else
           f"mixed: MoE better on {diff['seeds_moe_better']} of {n} seeds — no defensible winner")
rows = "\n".join(f"| {a} | {f(d['val_first'])} | **{f(d['val_final'])}** | {d['steps_to_regain']} | "
                 + (f"{d['final_maxvio_mean']['mean']:.3f} | {d['final_dead_total']} / {d['n_experts_total']} | {d['max_dead_any_eval']}" if "final_maxvio_mean" in d else "– | – | –")
                 + " |" for a, d in A.items())
fid = "\n".join(f"| {k} | {f(v)} |" for k, v in S["fidelity"].items())
open("README.md", "w").write(f"""# ERA V5 · Session 14 — Dense ("linear") model → Mixture-of-Experts

Train a dense model, convert it into an MoE, show it **continues to train** and **loss keeps dropping**.
Everything below is generated from `summary.json` (`make_readme.py`) and asserted by `test_readme.py`.
Config `{S['config']}`, {n} seeds ({S['seeds']}), device `{S['device']}`, total wall time {S['wall_min_total']:.1f} min.

![loss curves](loss_curves.png)

## Setup
| | Dense | MoE after conversion |
|---|---|---|
| Params total / active | {pd[0]:,} / {pd[1]:,} | {pm[0]:,} / {pm[1]:,} |

Model: decoder, RMSNorm, RoPE, GQA, SwiGLU, tied embeddings (`configs/{S['config']}.json`). Data: TinyStories with own
byte-level BPE. {S['tokens_per_step']:,} tokens/step; phase 1 dense {S['steps'][0]} steps, then {S['steps'][1]} steps per arm
from the **same checkpoint on identical batches** with one WSD LR schedule over the whole run. {'bf16 autocast' if S['device'] != 'cpu' else 'fp32 (CPU)'}; router always fp32.

## Conversion (`upcycle.py`)
- **Shared expert** (always on) = crop of the first dense neurons (Rohan's method 1).
- **Routed experts** = drop-upcycling r=0.5: each copies a random half-width subset of the remaining dense neurons,
  redraws the other half (`w1/w3` ~ N(0, dense std), `w2` = 0). Copied `w2` scaled by (remaining/copied) so each expert is
  an unbiased estimate of the remaining dense FFN; gates renormalised to 1 over top-k.
- **Router**: sigmoid, fp32, top-2. **Balancing**: aux-loss-free bias (DeepSeek-V3), selection only, γ=0.01, loads
  accumulated over 8-step windows. **Probabilistic top-k**: annealed Gumbel noise on selection for the first steps.
- AdamW moments carried over for every unchanged tensor (attention, norms, embedding).

## Results (validation loss, {'mean ± std over ' + str(n) + ' seeds' if n > 1 else 'single seed'})
Phase 1: dense val {f(S['val_phase1_start'])} → {f(S['val_phase1_end'])}.

| arm | first eval after conversion | final | steps to regain pre-conversion val (per seed) | final MaxVio | dead experts at end (per seed) | max dead any eval (per seed) |
|---|---|---|---|---|---|---|
{rows}

- **Continues training**: moe_drop goes {f(A['moe_drop']['val_first'])} → {f(A['moe_drop']['val_final'])}.
- **MoE vs dense at equal active params**: final difference moe_drop − dense = {diff['mean']:+.4f}{(' ± ' + format(diff['std'], '.4f')) if n > 1 else ''}
  (per seed {diff['per_seed']}); {verdict}. Seed-mean MoE stays below dense from step {S['moe_drop_stays_below_dense_from_step']}.
- **Ablation — which fix keeps cloned experts alive?** clone_hard (no balancing, no noise) vs clone_bias / clone_noise /
  clone_both isolate the bias balancing and probabilistic selection; moe_drop adds diverse (dropped) initialisation on top.
  Final val per arm: {', '.join(f"{a} {A[a]['val_final']['mean']:.4f}" for a in A)}; dead experts at end:
  {', '.join(f"{a} {A[a]['final_dead_total']}" for a in A if 'final_dead_total' in A[a])}.

- **Key finding**: arms WITH bias balancing ({', '.join(a for a in A if a in ('moe_drop','clone_bias','clone_both'))}) end with
  {[A[a]['final_dead_total'] for a in ('moe_drop','clone_bias','clone_both')]} dead experts and finals
  {[A[a]['val_final']['mean'] for a in ('moe_drop','clone_bias','clone_both')]} vs dense {A['dense_continue']['val_final']['mean']};
  arms WITHOUT it (clone_hard, clone_noise) end with {[A[a]['final_dead_total'] for a in ('clone_hard','clone_noise')]} dead and finals
  {[A[a]['val_final']['mean'] for a in ('clone_hard','clone_noise')]}. Probabilistic selection only delays collapse (0 dead while
  noise is on, then experts die). Drop-upcycling's diverse init did not beat plain clones once balancing was on.

![expert load](expert_load.png)

### Conversion fidelity (val at step 0, no training)
| variant | val |
|---|---|
{fid}

`partition_all_experts_on` equals `dense`: the conversion wiring is exact; every other jump comes from routing and redrawing.

## Limitations
- Small model on TinyStories; {n} seeds only. Differences smaller than the seed std are not claims.
- No MoE-from-scratch baseline at equal compute.
- Run on a 1-core CPU sandbox; `session14_moe_upcycling_colab.ipynb` runs the same pipeline at ~20M params × 3 seeds
  (`configs/colab20m.json`) on a Colab L4 — not yet executed.
- Expert compute is a Python loop over experts (no grouped GEMM), so MoE tok/s understates a real kernel.

## Reproduce
```
pip install tokenizers datasets matplotlib pytest
python -m pytest -q test_invariants.py
python run_all.py configs/{S['config']}.json && python report.py configs/{S['config']}.json && python make_readme.py
python -m pytest -q
```
""")
