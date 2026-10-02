"""Tiny decoder LM: RMSNorm, RoPE, GQA attention, SwiGLU FFN (dense) or MoE FFN.
MoE = 1 always-on shared expert + E routed experts, sigmoid router (fp32), top-k,
aux-loss-free bias balancing (DeepSeek-V3 style), optional annealed Gumbel
("probabilistic") selection for the first steps after upcycling."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass, field

@dataclass
class Cfg:
    vocab: int = 2048
    d: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_kv: int = 2          # GQA: 2 KV heads shared by 4 query heads
    seq: int = 128
    ffn: int = 512         # dense SwiGLU hidden
    # MoE (used only when moe=True)
    moe: bool = False
    shared: int = 256      # shared-expert width
    n_exp: int = 8
    exp_w: int = 128
    top_k: int = 2
    route_scale: float = 1.0

class RMSNorm(nn.Module):
    def __init__(s, d): super().__init__(); s.w = nn.Parameter(torch.ones(d))
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * s.w

def rope(T, hd):
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(torch.arange(T).float(), inv)
    return f.cos(), f.sin()

def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)

class Attn(nn.Module):
    def __init__(s, c):
        super().__init__(); s.c = c; s.hd = c.d // c.n_head
        s.q = nn.Linear(c.d, c.d, bias=False)
        s.kv = nn.Linear(c.d, 2 * c.n_kv * s.hd, bias=False)
        s.o = nn.Linear(c.d, c.d, bias=False)
    def forward(s, x, cos, sin):
        B, T, _ = x.shape; c = s.c
        q = s.q(x).view(B, T, c.n_head, s.hd).transpose(1, 2)
        k, v = s.kv(x).view(B, T, 2, c.n_kv, s.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = c.n_head // c.n_kv
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(B, T, c.d))

class DenseFFN(nn.Module):
    """SwiGLU: w2( silu(w1 x) * w3 x ). Weight rows of w1/w3 and cols of w2 = 'neurons'."""
    def __init__(s, d, h):
        super().__init__()
        s.w1 = nn.Parameter(torch.randn(h, d) * 0.02)
        s.w3 = nn.Parameter(torch.randn(h, d) * 0.02)
        s.w2 = nn.Parameter(torch.randn(d, h) * 0.02)
    def forward(s, x): return (F.silu(x @ s.w1.t()) * (x @ s.w3.t())) @ s.w2.t()

class MoEFFN(nn.Module):
    def __init__(s, c):
        super().__init__(); s.c = c
        s.shared = DenseFFN(c.d, c.shared) if c.shared > 0 else None
        E, W, d = c.n_exp, c.exp_w, c.d
        s.w1 = nn.Parameter(torch.randn(E, W, d) * 0.02)
        s.w3 = nn.Parameter(torch.randn(E, W, d) * 0.02)
        s.w2 = nn.Parameter(torch.randn(E, d, W) * 0.02)
        s.router = nn.Linear(d, E, bias=False)          # kept fp32 always
        nn.init.normal_(s.router.weight, std=0.006)
        s.register_buffer("bias", torch.zeros(E))         # aux-loss-free balancing bias (selection only)
        s.register_buffer("load", torch.zeros(E))         # token counts since last bias update
        s.register_buffer("win", torch.zeros(E))          # token counts since last stats read (logging)
        s.noise_T = 0.0                                   # >0 => probabilistic (Gumbel) top-k
        s.last_counts = None
    def forward(s, x):
        B, T, d = x.shape; c = s.c
        xf = x.reshape(-1, d)
        with torch.autocast(device_type=x.device.type, enabled=False):   # router always fp32
            logits = s.router(xf.float())
        score = torch.sigmoid(logits)                              # affinity
        sel = score.detach() + s.bias                              # bias only affects WHO is chosen
        if s.training and s.noise_T > 0:
            g = -torch.log(-torch.log(torch.rand_like(sel).clamp_min(1e-9)))
            sel = sel + s.noise_T * g
        idx = sel.topk(c.top_k, -1).indices                        # [N,k]
        w = score.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True) * c.route_scale            # renormalise over chosen (not over bias)
        out = torch.zeros_like(xf)
        counts = torch.bincount(idx.flatten(), minlength=c.n_exp)
        if s.training: s.load += counts.float(); s.win += counts.float()
        s.last_counts = counts
        for e in range(c.n_exp):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0: continue
            h = xf[tok]
            y = (F.silu(h @ s.w1[e].t()) * (h @ s.w3[e].t())) @ s.w2[e].t()
            out.index_add_(0, tok, (y * w[tok, slot].unsqueeze(-1)).to(out.dtype))
        if s.shared is not None: out = out + s.shared(xf).to(out.dtype)
        return out.view(B, T, d).to(x.dtype)
    @torch.no_grad()
    def update_bias(s, gamma):
        """DeepSeek-V3 aux-loss-free rule, applied over the accumulated window: overloaded -> bias down."""
        if s.load.sum() == 0: return
        f = s.load / s.load.sum()
        s.bias += gamma * torch.sign(f.mean() - f)
        s.load.zero_()

class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.n1, s.n2 = RMSNorm(c.d), RMSNorm(c.d)
        s.attn = Attn(c)
        s.ffn = MoEFFN(c) if c.moe else DenseFFN(c.d, c.ffn)
    def forward(s, x, cos, sin):
        x = x + s.attn(s.n1(x), cos, sin)
        return x + s.ffn(s.n2(x))

class LM(nn.Module):
    def __init__(s, c):
        super().__init__(); s.c = c
        s.emb = nn.Embedding(c.vocab, c.d); nn.init.normal_(s.emb.weight, std=0.02)
        s.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        s.nf = RMSNorm(c.d)
        cos, sin = rope(c.seq, c.d // c.n_head)
        s.register_buffer("cos", cos, persistent=False); s.register_buffer("sin", sin, persistent=False)
    def forward(s, idx, tgt=None):
        x = s.emb(idx); T = idx.shape[1]
        for b in s.blocks: x = b(x, s.cos[:T], s.sin[:T])
        logits = s.nf(x) @ s.emb.weight.t()                 # tied embeddings
        if tgt is None: return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1))
    def moe_layers(s): return [b.ffn for b in s.blocks if isinstance(b.ffn, MoEFFN)]

def param_counts(m):
    """total vs active (per-token) parameters."""
    total = sum(p.numel() for p in m.parameters())
    active = total
    for f in m.moe_layers():
        per_exp = f.w1[0].numel() + f.w2[0].numel() + f.w3[0].numel()
        active -= (f.c.n_exp - f.c.top_k) * per_exp
    return total, active
