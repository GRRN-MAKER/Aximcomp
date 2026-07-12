# SYNAXIM on AXIM

AXIM is SYNAXIM's execution substrate. Every SYNAXIM operation lowers to an
AXIM op, so a whole SYNAXIM model runs on any hardware — Nvidia, AMD, Intel,
Apple — with zero CUDA.

## SYNAXIM ops

| AXIM op | SYNAXIM primitive |
|---------|-------------------|
| `int4_matvec` | Boolean-bit fused INT4 projection (Q,K,V,O,gate,up,down) |
| `lowrank_retrieve` | associative memory read `(q @ U) @ V^T` |
| `rmsnorm` | RMS normalization |
| `silu` / `swiglu` | MLP activation |

## Full layer forward

`synaxim_bridge.py` runs one SYNAXIM transformer-replacement layer:

```python
from axim_compiler.synaxim_bridge import LayerWeightsAXIM, synaxim_layer_forward

h = synaxim_layer_forward(hidden_state, weights, device="auto")
```

The layer performs:

```
h_norm  = rmsnorm(h, norm_attn)
q       = int4_matvec(h_norm, W_q)          # AXIM
attn    = lowrank_retrieve(q, U, V)         # AXIM (O(D×r))
o       = int4_matvec(attn, W_o)            # AXIM
h       = h + o                             # residual
h_norm  = rmsnorm(h, norm_mlp)
mlp     = swiglu(...) → int4_matvec(_, W_down)   # AXIM
h       = h + mlp
```

## Verified (Apple M3)

```
Input hidden state norm:  10.7150
  [ cpu] output norm = 10.7188  finite=True
  [auto] output norm = 10.7188  finite=True
CPU vs auto max diff: 1.56e-08
✅ SYNAXIM layer ran end-to-end through AXIM — CUDA-free.
```

Load real weights with the [.symb loader](loader.md).
