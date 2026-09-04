import torch
from easydict import EasyDict as edict


def _cfg(T=12, V=4, F=3):
    c = edict()
    c.T_h = T; c.T_p = T; c.F = F; c.V = V
    c.device = torch.device("cpu")
    return c


def _inp(B=2, T=12, V=4, F=3):
    return (torch.randn(B, T, V, F), torch.randn(B, T, 1, 1), torch.randn(B, T, 1, 1))


def test_cast_forward():
    from model.cast import Cast
    m = Cast(_cfg(), n_units=[3, 16, 16], n_heads=[4, 1], num_v=4)
    x, pw, pd = _inp()
    assert m(x, pw, pd).shape == (2, 12, 4, 3)


def test_cast_ogat():
    from model.cast import Cast
    m = Cast(_cfg(), n_units=[3, 16, 16], n_heads=[4, 1], num_v=4, use_gat=False)
    x, pw, pd = _inp()
    assert m(x, pw, pd).shape == (2, 12, 4, 3)


def test_cast_otem():
    from model.cast import Cast
    m = Cast(_cfg(), n_units=[3, 16, 16], n_heads=[4, 1], num_v=4, use_temporal=False)
    x, pw, pd = _inp()
    assert m(x, pw, pd).shape == (2, 12, 4, 3)


def test_cast_ospa():
    from model.cast import Cast
    m = Cast(_cfg(), n_units=[3, 16, 16], n_heads=[4, 1], num_v=4, use_spatio=False)
    x, pw, pd = _inp()
    assert m(x, pw, pd).shape == (2, 12, 4, 3)


def test_cast_onet():
    from model.cast import Cast
    m = Cast(_cfg(), n_units=[3, 16, 16], n_heads=[4, 1], num_v=4, use_network=False)
    x, pw, pd = _inp()
    assert m(x, pw, pd).shape == (2, 12, 4, 3)


def test_autoformer():
    from models.Autoformer import Model
    T, V, d = 12, 4, 12
    mc = edict()
    mc.version = "Fourier"; mc.modes = 64
    mc.seq_len = T; mc.label_len = T; mc.pred_len = T
    mc.output_attention = False; mc.moving_avg = [24]
    mc.enc_in = d; mc.d_model = d; mc.embed = "timeF"; mc.freq = "h"
    mc.dropout = 0.0; mc.L = 3; mc.base = "legendre"
    mc.cross_activation = "tanh"; mc.dec_in = d
    mc.e_layers = 2; mc.n_heads = V; mc.d_ff = 2048
    mc.activation = "gelu"; mc.c_out = d; mc.d_layers = 1
    mc.node_num = V; mc.factor = 1.2
    model = Model(mc)
    x = torch.randn(2, T, d)
    out = model(x, None, torch.randn(2, T, d), None)
    assert out.shape == (2, T, d)


def test_itransformer():
    from models.iTransformer import Model
    m = Model(seq_len=12, pred_len=12, d_model=12, embed="timeF",
              n_heads=1, d_ff=12, output_attention=0, e_layers=1,
              use_norm=False, freq="h", activation="gelu",
              dropout=0.5, class_strategy="projection", factor=3)
    x = torch.randn(2, 12, 12)
    out = m(x, None, None, None)
    assert out.shape[-1] == 12


def test_tsamd():
    from models.tsAMD import AMD
    m = AMD(input_shape=(12, 12), pred_len=12, dropout=0.1,
            n_block=1, patch=16, k=3, c=2, alpha=0.0,
            target_slice=slice(0, None), norm=False, layernorm=True)
    x = torch.randn(2, 12, 12)
    out = m(x)
    assert isinstance(out, tuple) or hasattr(out, 'shape')
