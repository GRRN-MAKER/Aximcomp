# .symb Model Loader

The loader reads SYNAXIM's proprietary `.symb` weight files into AXIM
buffers, so real models (e.g. Magnus / Mistral 7B) run on any hardware.

## Binary layout (INT4)

```
[4B uint32: num_groups]
[4B uint32: group_size]
[4B uint32: numel]
[4B uint32: shape_dim0] [4B uint32: shape_dim1]
[num_groups * 4B: FP32 scale factors]
[ceil(numel/2) bytes: packed INT4 pairs — (hi<<4)|lo]
```

SYNAXIM uses symmetric signed INT4 `[-8, 7]` mapped to `[0, 15]`;
dequant = `(nibble - 8) * scale`, i.e. `zero_point = -8 * scale`.

## Usage

```python
from axim_compiler.loader import load_symb_int4, load_model

# Single weight
w = load_symb_int4("model-symb/layers/layer_00/attn_q.symb")
out = axim.int4_matvec(x, w["packed"], w["scales"], w["zeros"],
                       w["shape"][0], w["shape"][1], w["group_size"],
                       device="gpu")

# Full model
model = load_model("magnus-symb", max_layers=4)
print(model.describe())
```

## Directory layout

```
model-symb/
  config.symb.json        (architecture source of truth)
  embeddings.symb         (FP16)
  lm_head.symb            (FP16)
  final_norm.symb         (FP16)
  layers/layer_00/
    attn_q.symb ... mlp_down.symb   (INT4)
    norm_attn.symb, norm_mlp.symb   (FP16)
```

## Verified

```
loader: shape correct                 ✅
loader: group_size correct            ✅
loader→AXIM: matches dequant reference ✅
loader→AXIM: GPU == CPU               ✅
```
