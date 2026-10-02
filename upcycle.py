"""Dense -> MoE conversion (the 'growing' step from Session 14).
shared expert : crop the FIRST `shared` neurons of the dense FFN (Rohan's method 1).
routed experts: drop-upcycling (Nakamura et al. 2025, r=0.5): each expert copies a random
                subset of the REMAINING dense neurons and redraws the rest.
Scaling: rest-of-FFN output = sum over R remaining neurons. An expert holding m random
copied neurons, with their w2 scaled by R/m, is an unbiased estimate of that sum; gates are
renormalised to sum to 1 over top-k, so the MoE output is an unbiased estimate of the dense
output at step 0. Redrawn neurons: w1/w3 ~ N(0, std of dense w1/w3); w2 = 0 (zero-init, so
they add no noise at conversion but receive gradient immediately).
mode='clone' : ablation from the lecture's failure case - every expert is an IDENTICAL copy
                (same neuron subset, nothing redrawn)."""
import copy, torch
from model import LM, Cfg, MoEFFN, DenseFFN

@torch.no_grad()
def upcycle(dense: LM, moe_cfg: Cfg, mode="drop", r=0.5, seed=0, redraw_w2="zero"):
    g = torch.Generator().manual_seed(seed)
    moe = LM(moe_cfg)
    sd = {k: v for k, v in dense.state_dict().items() if ".ffn." not in k}
    missing, unexpected = moe.load_state_dict(sd, strict=False)
    assert not unexpected and all(".ffn." in k for k in missing), (missing, unexpected)
    c = moe_cfg
    for bd, bm in zip(dense.blocks, moe.blocks):
        f: DenseFFN = bd.ffn; m: MoEFFN = bm.ffn
        H = f.w1.shape[0]; S = c.shared
        if S:
            m.shared.w1.copy_(f.w1[:S]); m.shared.w3.copy_(f.w3[:S]); m.shared.w2.copy_(f.w2[:, :S])
        rest = torch.arange(S, H); R = len(rest)
        s1, s3, s2 = f.w1[S:].std(), f.w3[S:].std(), f.w2[:, S:].std()
        n_copy = int(round(c.exp_w * (1 - r))) if mode == "drop" else c.exp_w
        n_copy = min(n_copy, R)
        clone_pick = rest[torch.randperm(R, generator=g)[:n_copy]]
        for e in range(c.n_exp):
            pick = clone_pick if mode == "clone" else rest[torch.randperm(R, generator=g)[:n_copy]]
            scale = R / n_copy
            m.w1[e, :n_copy] = f.w1[pick]; m.w3[e, :n_copy] = f.w3[pick]
            m.w2[e, :, :n_copy] = f.w2[:, pick] * scale
            nf = c.exp_w - n_copy
            if nf:
                m.w1[e, n_copy:] = torch.randn(nf, c.d, generator=g) * s1
                m.w3[e, n_copy:] = torch.randn(nf, c.d, generator=g) * s3
                m.w2[e, :, n_copy:] = 0.0 if redraw_w2 == "zero" else torch.randn(c.d, nf, generator=g) * s2
    return moe

@torch.no_grad()
def partition_identity(dense: LM, base: Cfg):
    """Exactness check: split the dense FFN into disjoint experts, turn ALL on with gate 1.
    Must reproduce the dense model bit-for-bit (up to float reassociation)."""
    H = base.ffn; E = 4; assert H % E == 0
    c = Cfg(**{**base.__dict__, "moe": True, "shared": 0, "n_exp": E, "exp_w": H // E, "top_k": E})
    moe = LM(c)
    moe.load_state_dict({k: v for k, v in dense.state_dict().items() if ".ffn." not in k}, strict=False)
    for bd, bm in zip(dense.blocks, moe.blocks):
        f, m = bd.ffn, bm.ffn
        for e in range(E):
            sl = slice(e * H // E, (e + 1) * H // E)
            m.w1[e] = f.w1[sl]; m.w3[e] = f.w3[sl]; m.w2[e] = f.w2[:, sl]
        m.router.weight.zero_()   # equal scores -> gates 1/E each
        m.c.route_scale = E       # -> scale back to 1
    return moe
