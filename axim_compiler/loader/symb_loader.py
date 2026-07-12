"""
AXIM .symb Loader
=================
Reads SYNAXIM's proprietary .symb weight files into AXIM buffers so that
real models (e.g. Magnus / Mistral 7B) run on any hardware through AXIM.

.symb INT4 binary layout:
    [4B uint32: num_groups]
    [4B uint32: group_size]
    [4B uint32: numel]
    [4B uint32: shape_dim0] [4B uint32: shape_dim1]
    [num_groups * 4B: FP32 scale factors]
    [ceil(numel/2) bytes: packed INT4 pairs — (hi<<4)|lo]

Directory layout (config.symb.json is the source of truth):
    model-symb/
      config.symb.json
      embeddings.symb        (FP16)
      lm_head.symb           (FP16)
      final_norm.symb        (FP16)
      layers/layer_00/attn_q.symb ... mlp_down.symb, norm_attn.symb, ...

The loader returns plain Python structures the AXIM ops accept directly
(packed uint8 list, scales/zeros lists), so a full SYNAXIM model executes
on CPU (SIMD) or GPU (Metal/Vulkan) with zero CUDA.
"""

from __future__ import annotations

import os
import json
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════
# Low-level .symb readers
# ══════════════════════════════════════════════════════════════

def load_symb_int4(path: str) -> Dict:
    """
    Load an INT4 .symb file.

    Returns a dict:
        {
          "packed": list[int]  (uint8, len = ceil(numel/2)),
          "scales": list[float] (len = num_groups),
          "zeros":  list[float] (SYNAXIM is symmetric → derived zero points),
          "num_groups": int, "group_size": int, "numel": int,
          "shape": (dim0, dim1),
        }
    """
    with open(path, "rb") as f:
        num_groups, group_size, numel = struct.unpack("III", f.read(12))
        dim0, dim1 = struct.unpack("II", f.read(8))
        scales = list(struct.unpack(f"{num_groups}f", f.read(num_groups * 4)))
        packed = list(f.read())  # remaining bytes → uint8 list

    # SYNAXIM INT4 is symmetric signed [-8, 7] mapped to [0,15].
    # Dequant is (nibble - 8) * scale, i.e. zero_point = -8 * scale.
    # AXIM's kernel uses out = nibble * scale + zero, so zero = -8 * scale.
    zeros = [-8.0 * s for s in scales]

    return {
        "packed": packed,
        "scales": scales,
        "zeros": zeros,
        "num_groups": num_groups,
        "group_size": group_size,
        "numel": numel,
        "shape": (dim0, dim1) if dim1 > 1 else (dim0,),
    }


def load_symb_fp16(path: str, shape: Optional[Tuple[int, ...]] = None) -> List[float]:
    """Load a flat FP16 .symb file into a Python float list."""
    import array
    raw = open(path, "rb").read()
    # decode IEEE-754 half via struct 'e' (Python 3.6+)
    n = len(raw) // 2
    vals = list(struct.unpack(f"{n}e", raw[: n * 2]))
    return vals


def load_config(model_dir: str) -> Dict:
    """Load config.symb.json — the model architecture source of truth."""
    path = os.path.join(model_dir, "config.symb.json")
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
# High-level model container
# ══════════════════════════════════════════════════════════════

@dataclass
class SymbLayer:
    """One SYNAXIM layer's weights, ready for AXIM ops."""
    index: int
    weights: Dict[str, Dict] = field(default_factory=dict)   # name → int4 dict
    norms: Dict[str, List[float]] = field(default_factory=dict)  # fp16 vectors

    def matmul_inputs(self, name: str):
        """Return (packed, scales, zeros, out_dim, in_dim, group_size)."""
        w = self.weights[name]
        shape = w["shape"]
        out_dim = shape[0]
        in_dim = shape[1] if len(shape) > 1 else 1
        return (w["packed"], w["scales"], w["zeros"],
                out_dim, in_dim, w["group_size"])


@dataclass
class SymbModel:
    """A loaded SYNAXIM model runnable through AXIM on any hardware."""
    config: Dict
    embeddings: List[float]
    lm_head: List[float]
    final_norm: List[float]
    layers: List[SymbLayer]

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def hidden_size(self) -> int:
        return self.config.get("hidden_size", 0)

    def describe(self) -> str:
        return (
            f"SymbModel: {self.config.get('model_name', '?')} | "
            f"layers={self.num_layers} | D={self.hidden_size} | "
            f"quant={self.config.get('quantization', {}).get('method', '?')}"
        )


# ══════════════════════════════════════════════════════════════
# Full model loader
# ══════════════════════════════════════════════════════════════

_LAYER_WEIGHTS = [
    "attn_q", "attn_k", "attn_v", "attn_o",
    "mlp_gate", "mlp_up", "mlp_down",
]
_LAYER_NORMS = ["norm_attn", "norm_mlp"]


def load_model(model_dir: str, max_layers: Optional[int] = None) -> SymbModel:
    """
    Load a full .symb model directory into AXIM-ready structures.

    Args:
        model_dir: path containing config.symb.json + layers/
        max_layers: optionally load only the first N layers (fast smoke test)
    """
    cfg = load_config(model_dir)
    n_layers = cfg.get("num_layers", 0)
    if max_layers is not None:
        n_layers = min(n_layers, max_layers)

    D = cfg.get("hidden_size", 0)

    # Globals
    emb = _try_fp16(os.path.join(model_dir, cfg.get("embedding_file", "embeddings.symb")))
    lmh = _try_fp16(os.path.join(model_dir, cfg.get("lm_head_file", "lm_head.symb")))
    fnorm = _try_fp16(os.path.join(model_dir, "final_norm.symb")) or [1.0] * D

    layers: List[SymbLayer] = []
    pattern = cfg.get("layer_dir_pattern", "layers/layer_{:02d}")
    for i in range(n_layers):
        ldir = os.path.join(model_dir, pattern.format(i))
        layer = SymbLayer(index=i)
        for wname in _LAYER_WEIGHTS:
            wp = os.path.join(ldir, f"{wname}.symb")
            if os.path.exists(wp):
                layer.weights[wname] = load_symb_int4(wp)
        for nname in _LAYER_NORMS:
            np_ = os.path.join(ldir, f"{nname}.symb")
            if os.path.exists(np_):
                layer.norms[nname] = load_symb_fp16(np_)
        layers.append(layer)

    return SymbModel(
        config=cfg, embeddings=emb or [], lm_head=lmh or [],
        final_norm=fnorm, layers=layers,
    )


def _try_fp16(path: str):
    return load_symb_fp16(path) if os.path.exists(path) else None
