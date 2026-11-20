"""Export the ViTPose+ whole-body checkpoints (onepose) to ONNX.

The graph maps a normalized ``(N, 3, 256, 192)`` float32 RGB crop to ``(N, 133, 64, 48)`` heatmaps;
pre- and post-processing live in ``wholebodypose.models.vitpose.onnx_runtime`` (NumPy only).

The onepose checkpoints are float16. PyTorch runs them in float32 after ``load_state_dict``, so every
float32 initializer whose values are exactly representable in float16 is stored as float16 and cast
back to float32 inside the graph (``Cast`` is folded by ONNX Runtime when the session is created).
This halves the file size without changing a single weight; the computation stays float32.
Initializers that are not float16-exact (e.g. BatchNorm folded into the deconvolutions) stay float32.

Needs torch, onepose and onnx (export-time only)::

    python scripts/export_vitpose_onnx.py --sizes small base large huge --out ~/.cache/wholebodypose/vitpose
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

OPSET = 17
INPUT_NAME, OUTPUT_NAME = "input", "heatmaps"


def build_torch_model(size: str):
    import onepose.models.factory as factory
    import torch
    from onepose.models import vitpose
    from onepose.utils import read_cfg

    from wholebodypose.weights import get_weight

    name = f"ViTPose+_{size}_coco_wholebody"
    cfg_path = Path(factory.__file__).parent / "configs" / factory.model_config[name]["model_cfg"]
    cfg = read_cfg(str(cfg_path))
    model = vitpose.ViTPose(cfg.model)
    state = torch.load(get_weight(f"vitpose_{size}"), map_location="cpu")
    model.load_state_dict(state)
    return model.eval().float()


def fp16_storage(model_proto):
    """Store float16-exact float32 initializers as float16 + Cast. Returns (#converted, #kept)."""
    from onnx import TensorProto, helper, numpy_helper

    graph = model_proto.graph
    new_inits, casts, converted, kept = [], [], 0, 0
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            new_inits.append(init)
            continue
        arr = numpy_helper.to_array(init)
        half = arr.astype(np.float16)
        if arr.size < 16 or not np.array_equal(half.astype(np.float32), arr):
            new_inits.append(init)
            kept += 1
            continue
        stored = numpy_helper.from_array(half, init.name + "__fp16")
        new_inits.append(stored)
        casts.append(helper.make_node("Cast", [stored.name], [init.name], to=TensorProto.FLOAT))
        converted += 1
    del graph.initializer[:]
    graph.initializer.extend(new_inits)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    return converted, kept


def export(size: str, out_dir: Path) -> Path:
    import onnx
    import torch

    model = build_torch_model(size)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(tempfile.mkdtemp(prefix=f"vitpose_{size}_", dir=out_dir))
    raw = raw_dir / "raw.onnx"  # the exporter may write external-data files next to it
    final = out_dir / f"vitpose+_{size}_coco_wholebody.onnx"
    dummy = torch.zeros(1, 3, 256, 192)
    torch.onnx.export(
        model,
        (dummy,),
        str(raw),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET,
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    proto = onnx.load(str(raw), load_external_data=True)
    converted, kept = fp16_storage(proto)
    proto.producer_name = "wholebodypose/scripts/export_vitpose_onnx.py"
    onnx.save(proto, str(final))
    onnx.checker.check_model(str(final))
    shutil.rmtree(raw_dir)
    print(f"{final}: {final.stat().st_size / 2**20:.1f} MiB, fp16-stored {converted}, fp32 kept {kept}")
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", nargs="+", default=["small", "base", "large", "huge"])
    ap.add_argument(
        "--out", type=Path, default=Path(os.environ.get("WHOLEBODYPOSE_HOME", "~/.cache/wholebodypose")) / "vitpose"
    )
    args = ap.parse_args(argv)
    out = args.out.expanduser()
    for size in args.sizes:
        export(size, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
