"""Recurrent models (LSTM / GRU) for option pricing — smile-as-sequence.

Why recurrence is legitimate here
---------------------------------
A naive LSTM over randomly ordered option rows is meaningless: a single
chain snapshot is cross-sectional, not a time series. The defensible
reformulation used here treats each expiry's smile as a SEQUENCE ORDERED
BY STRIKE: neighbouring strikes are strongly coupled (smile smoothness,
butterfly no-arbitrage), so a recurrent pass along the strike axis lets
the model exploit local smile structure that pointwise models ignore.

Each expiry = one padded sequence of per-option features; the network
emits one prediction per option (seq2seq). Targets are the BS residuals
(C - BS_atm)/K, consistent with the rest of the project.

Feature scaling uses TRAIN statistics only (no leakage), and sequences
are padded with a mask applied in the loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.manual_seed(42)
DEVICE = "cpu"


# ---------------------------------------------------------------- dataset
def _to_sequences(df: pd.DataFrame, feats: list[str], target: str,
                  mu: np.ndarray, sd: np.ndarray):
    """Group by expiry, sort by strike, pad to max length. Returns
    (X [B,L,F], y [B,L], mask [B,L], index_per_seq)."""
    seqs, tgts, idxs = [], [], []
    for _, g in df.groupby("expiry"):
        g = g.sort_values("strike")
        seqs.append((g[feats].values - mu) / sd)
        tgts.append(g[target].values)
        idxs.append(g.index.values)
    L = max(len(s) for s in seqs)
    B, F = len(seqs), len(feats)
    X = np.zeros((B, L, F), np.float32)
    y = np.zeros((B, L), np.float32)
    m = np.zeros((B, L), np.float32)
    for i, (s, t) in enumerate(zip(seqs, tgts)):
        X[i, :len(s)] = s
        y[i, :len(t)] = t
        m[i, :len(t)] = 1.0
    return (torch.tensor(X), torch.tensor(y), torch.tensor(m), idxs)


# ------------------------------------------------------------------ model
class SmileRNN(nn.Module):
    """Bidirectional LSTM/GRU along the strike axis, seq2seq head."""

    def __init__(self, n_features: int, cell: str = "lstm",
                 hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(n_features, hidden, layers, batch_first=True,
                           bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):                       # x: [B, L, F]
        h, _ = self.rnn(x)
        return self.head(h).squeeze(-1)         # [B, L]


# ------------------------------------------------------------------ train
def fit_predict_rnn(train: pd.DataFrame, test: pd.DataFrame,
                    feats: list[str], cell: str = "lstm",
                    target: str = "resid", epochs: int = 400,
                    lr: float = 1e-3, patience: int = 40,
                    verbose: bool = False) -> np.ndarray:
    """Train on train expiries (with a 1-expiry validation hold-out for
    early stopping), return per-option predictions aligned to test.index."""
    mu = train[feats].values.mean(0)
    sd = train[feats].values.std(0) + 1e-12

    # validation = last expiry of the train set (by T)
    val_exp = train.groupby("expiry")["T"].first().idxmax()
    tr, va = train[train["expiry"] != val_exp], train[train["expiry"] == val_exp]

    Xtr, ytr, mtr, _ = _to_sequences(tr, feats, target, mu, sd)
    Xva, yva, mva, _ = _to_sequences(va, feats, target, mu, sd)
    Xte, _, mte, idx_te = _to_sequences(test, feats, target, mu, sd)

    model = SmileRNN(len(feats), cell=cell).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def masked_mse(pred, y, m):
        return ((pred - y) ** 2 * m).sum() / m.sum()

    best_val, best_state, since = np.inf, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = masked_mse(model(Xtr), ytr, mtr)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = masked_mse(model(Xva), yva, mva).item()
        if vl < best_val - 1e-9:
            best_val, best_state, since = vl, {k: v.clone() for k, v in
                                               model.state_dict().items()}, 0
        else:
            since += 1
            if since >= patience:
                break
        if verbose and ep % 50 == 0:
            print(f"  [{cell}] epoch {ep:3d} train {loss.item():.2e} val {vl:.2e}")
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy()
    out = pd.Series(index=test.index, dtype=float)
    for i, idx in enumerate(idx_te):
        out.loc[idx] = pred[i, :len(idx)]
    return out.values
