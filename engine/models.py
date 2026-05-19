"""8 backbone encoders — uniform contract  forward(x:[B,L,F]) -> z:[B,d].

In-house, faithful to each paper's CORE MECHANISM, adapted for our task
(cross-sectional scalar return; the original models forecast multi-step
multivariate series — adaptation is expected and documented). Correctness
of the defining mechanism is verified by engine/verify_models.py
(channel-independence, variate-tokens, causality, decomposition recon,
global receptive field, variable-selection softmax).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- shared bits ----------
class _MovingAvg(nn.Module):
    def __init__(self, k=25):
        super().__init__(); self.k = k

    def forward(self, x):                       # x [B,F,L]
        pad = (self.k - 1) // 2
        xp = F.pad(x, (pad, self.k - 1 - pad), mode="replicate")
        return F.avg_pool1d(xp, self.k, 1)


class _GRN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a, self.b = nn.Linear(d, d), nn.Linear(d, d)
        self.g, self.n = nn.Linear(d, 2 * d), nn.LayerNorm(d)

    def forward(self, x):
        h = self.b(F.elu(self.a(x)))
        return self.n(x + F.glu(self.g(h), -1))


# ---------- 1. DLinear (Zeng AAAI'23): decomposition + linear ----------
class DLinear(nn.Module):
    # Canonical decomposition+linear KEPT AS-IS: its strength is being the
    # correct strong-SIMPLE baseline (bloating it would make it not-DLinear
    # and defeat the honest deep-vs-linear contrast). Only the output rep
    # width tracks the shared d_model for interface parity.
    def __init__(self, n_feat, seq_len, d_model=256, kernel=25):
        super().__init__()
        self.dec = _MovingAvg(kernel)
        self.lt, self.ls = nn.Linear(seq_len, d_model), nn.Linear(seq_len, d_model)

    def decompose(self, x):                     # [B,L,F] -> trend,seasonal [B,F,L]
        xf = x.transpose(1, 2)
        t = self.dec(xf)
        return t, xf - t

    def forward(self, x):
        t, s = self.decompose(x)
        return (self.lt(t) + self.ls(s)).mean(1)            # [B,d]


# ---------- 2. LSTM / 3. RNN baselines (504-len caveat, honest) ----------
class LSTMEnc(nn.Module):
    def __init__(self, n_feat, seq_len, d_model=256, layers=3):
        super().__init__()
        self.rnn = nn.LSTM(n_feat, d_model, layers, batch_first=True,
                           dropout=0.1)

    def forward(self, x):
        return self.rnn(x)[0][:, -1]


class RNNEnc(nn.Module):
    def __init__(self, n_feat, seq_len, d_model=256, layers=3):
        super().__init__()
        self.rnn = nn.RNN(n_feat, d_model, layers, batch_first=True,
                          nonlinearity="tanh", dropout=0.1)

    def forward(self, x):
        return self.rnn(x)[0][:, -1]


# ---------- 4. TCN (Bai'18): dilated CAUSAL conv (explicit left pad) ----
class TCN(nn.Module):
    def __init__(self, n_feat, seq_len, d_model=256):
        super().__init__()
        self.layers = nn.ModuleList()
        cin = n_feat
        for i in range(10):                      # dil 1..512 -> RF 2047>=504
            self.layers.append(nn.ModuleDict({
                "c": nn.Conv1d(cin, d_model, 3, dilation=2 ** i),
                "d": nn.Dropout(0.1)}))
            cin = d_model
        self.dils = [2 ** i for i in range(10)]

    def forward(self, x):
        h = x.transpose(1, 2)                    # [B,F,L]
        for blk, dl in zip(self.layers, self.dils):
            h = F.pad(h, (2 * dl, 0))            # LEFT-ONLY -> strictly causal
            h = blk["d"](F.relu(blk["c"](h)))
        return h.mean(-1)                        # [B,d]


# ---------- 5. iTransformer (Liu ICLR'24): variate tokens ----------
class iTransformer(nn.Module):
    def __init__(self, n_feat, seq_len, d_model=256, heads=8, layers=4):
        super().__init__()
        self.emb = nn.Linear(seq_len, d_model)   # whole series per variate
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, 0.1,
                                         batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)

    def tokens(self, x):                         # [B,L,F] -> variate tokens
        xv = x.transpose(1, 2)                   # [B,F,L]
        xv = (xv - xv.mean(-1, keepdim=True)) / (xv.std(-1, keepdim=True)
                                                 + 1e-5)     # per-variate norm
        return self.emb(xv)                      # [B,F,d]  (F tokens)

    def forward(self, x):
        return self.tr(self.tokens(x)).mean(1)   # attention ACROSS variates


# ---------- 6. PatchTST (Nie ICLR'23): patch + CHANNEL-INDEPENDENCE ----
class PatchTST(nn.Module):
    def __init__(self, n_feat, seq_len, d_model=256, patch=16, stride=8,
                 heads=8, layers=4):
        super().__init__()
        self.p, self.s = patch, stride
        npatch = (seq_len + (stride - (seq_len - patch) % stride) % stride
                  - patch) // stride + 1
        self.npatch = max(npatch, 1)
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.npatch, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, 0.1,
                                         batch_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.proj = nn.Linear(self.npatch * d_model, d_model)
        # PERMUTATION-INVARIANT channel reduction (re-jury-5 fix): the prior
        # Linear(n_feat*d_model) flatten was channel-ORDER-dependent and
        # defeated PatchTST's channel-independence for the model AS USED.
        # Mean over the F channel axis is symmetric -> forward() is provably
        # invariant to channel permutation (verify_models.py T-CI).
        self.outp = nn.Linear(d_model, d_model)

    def backbone(self, x):                       # CHANNEL-INDEPENDENT [B,F,d]
        B, L, Fd = x.shape
        xc = x.transpose(1, 2).reshape(B * Fd, L)          # each chan alone
        need = self.p + (self.npatch - 1) * self.s
        if need > L:
            xc = F.pad(xc, (0, need - L), mode="replicate")  # end-pad
        pat = xc.unfold(-1, self.p, self.s)                 # [B*F,np,P]
        h = self.tr(self.embed(pat) + self.pos)             # shared weights
        return self.proj(h.reshape(B * Fd, -1)).view(B, Fd, -1)

    def forward(self, x):
        r = self.backbone(x)                                # [B,F,d] chan-indep
        return self.outp(r.mean(1))                         # symmetric over F


# ---------- 7. GCformer (Zhao CIKM'23): STRUCTURED global conv + local ---
class GCformer(nn.Module):
    """Global branch = depthwise conv with a length-L kernel built from K
    decaying bases (params O(d*K), SUBLINEAR in L, length-covering) — a
    faithful structured-global-kernel (not a dense FIR). Local branch =
    Transformer on the recent window. Fuse."""
    def __init__(self, n_feat, seq_len, d_model=256, heads=8, K=24,
                 local=128):
        super().__init__()
        self.L, self.K, self.local = seq_len, K, local
        self.proj = nn.Linear(n_feat, d_model)
        # decays spread incl. NEAR-1 (long memory) so the kernel truly
        # covers the full L (slowest base ~0.9995^L stays appreciable);
        # random init decayed too fast -> not global (verify caught it).
        base = torch.logit(torch.linspace(0.90, 0.9995, K))
        self.decay = nn.Parameter(base.repeat(d_model, 1)
                                  + 0.01 * torch.randn(d_model, K))
        self.coef = nn.Parameter(torch.randn(d_model, K) * 0.1)
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, 0.1,
                                         batch_first=True)
        self.tr = nn.TransformerEncoder(enc, 3)         # deeper local branch

    def global_kernel(self):                     # [d_model, L] structured
        t = torch.arange(self.L, device=self.decay.device).float()
        dec = torch.sigmoid(self.decay).unsqueeze(-1)        # [d,K,1] in(0,1)
        k = (self.coef.unsqueeze(-1) * dec ** t).sum(1)      # [d,L]
        return k.flip(-1)                                    # causal align

    def forward(self, x):
        h = self.proj(x).transpose(1, 2)                     # [B,d,L]
        k = self.global_kernel().unsqueeze(1)                # [d,1,L]
        hp = F.pad(h, (self.L - 1, 0))                       # causal
        g = F.conv1d(hp, k, groups=h.size(1))[..., :x.size(1)]
        g = g[..., -1]                                       # global rep [B,d]
        loc = self.tr(self.proj(x)[:, -self.local:]).mean(1)  # local rep
        return g + loc


# ---------- 8. TFT (Lim'21): per-variable VSN + GRN + LSTM + interp attn -
class TFT(nn.Module):
    """Reduced OBSERVED-ONLY TFT (FNSPID has no static/known-future) with
    PER-VARIABLE embeddings (W,B per feature) + variable-selection softmax
    + GRN + LSTM + gated interpretable attention."""
    def __init__(self, n_feat, seq_len, d_model=256, heads=4):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_feat, d_model) * 0.05)
        self.b = nn.Parameter(torch.zeros(n_feat, d_model))
        self.vgrn = _GRN(d_model)
        self.vs = nn.Sequential(nn.Linear(n_feat * d_model, n_feat))
        self.grn = _GRN(d_model)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.att = nn.MultiheadAttention(d_model, heads, 0.1,
                                         batch_first=True)
        self.glu = nn.Linear(d_model, 2 * d_model)
        self.n = nn.LayerNorm(d_model)

    def vsn_weights(self, x):                    # [B,L,F] softmax over F
        e = x.unsqueeze(-1) * self.W + self.b    # per-variable embed [B,L,F,d]
        B, L, Fd, d = e.shape
        return torch.softmax(self.vs(e.reshape(B, L, Fd * d)), -1), e

    def forward(self, x):
        w, e = self.vsn_weights(x)
        sel = self.vgrn((e * w.unsqueeze(-1)).sum(2))        # [B,L,d]
        h, _ = self.lstm(self.grn(sel))
        a, _ = self.att(h, h, h, need_weights=False)
        h = self.n(h + F.glu(self.glu(a), -1))               # gated residual
        return h[:, -1]


REGISTRY = {"itransformer": iTransformer, "patchtst": PatchTST,
            "tft": TFT, "gcformer": GCformer, "dlinear": DLinear,
            "lstm": LSTMEnc, "rnn": RNNEnc, "cnn": TCN}
