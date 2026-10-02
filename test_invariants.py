import torch, pytest
from model import LM, Cfg
from upcycle import upcycle, partition_identity

def _dense():
    torch.manual_seed(1); m = LM(Cfg())
    for p in m.parameters(): p.data += 0.02 * torch.randn_like(p)   # non-trivial weights
    return m.eval()

def test_partition_identity_is_exact():
    d = _dense(); m = partition_identity(d, Cfg()).eval()
    x = torch.randint(0, 2048, (2, 64))
    assert torch.allclose(d(x), m(x), atol=1e-5)

def test_non_ffn_weights_copied():
    d = _dense(); m = upcycle(d, Cfg(moe=True))
    for k, v in d.state_dict().items():
        if ".ffn." not in k: assert torch.equal(v, m.state_dict()[k]), k

def test_shared_is_crop_of_first_neurons():
    d = _dense(); m = upcycle(d, Cfg(moe=True)); S = Cfg().shared
    assert torch.equal(m.blocks[0].ffn.shared.w1, d.blocks[0].ffn.w1[:S])

def test_drop_experts_differ_clone_experts_identical():
    d = _dense()
    drop = upcycle(d, Cfg(moe=True), mode="drop").blocks[0].ffn
    clone = upcycle(d, Cfg(moe=True), mode="clone").blocks[0].ffn
    assert not torch.equal(drop.w1[0], drop.w1[1])
    assert all(torch.equal(clone.w1[0], clone.w1[e]) for e in range(8))

def test_redrawn_half_is_half():
    d = _dense(); f = upcycle(d, Cfg(moe=True)).blocks[0].ffn
    assert (f.w2[:, :, 64:] == 0).all() and (f.w2[:, :, :64] != 0).any()

def test_active_params_match_dense():
    from model import param_counts
    dt, _ = param_counts(LM(Cfg())); mt, ma = param_counts(LM(Cfg(moe=True)))
    assert mt > 1.9 * dt and abs(ma - dt) / dt < 0.01   # 2x total, ~same active (router only)

def test_bias_update_direction():
    f = LM(Cfg(moe=True)).blocks[0].ffn
    f.load = torch.tensor([10., 0, 0, 0, 0, 0, 0, 0]); f.update_bias(0.01)
    assert f.bias[0] < 0 and (f.bias[1:] > 0).all()
