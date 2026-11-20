"""Measurements behind ``docs/guide/onnx.md``: ViTPose torch-vs-ONNX agreement, CPU latency, quantization.

    python scripts/vitpose_onnx_study.py compare --out compare.json   # torch vs ONNX keypoints
    taskset -c 0-7 python scripts/vitpose_onnx_study.py bench --threads 8 --out bench.json
    taskset -c 0-7 python scripts/vitpose_onnx_study.py quant --threads 8 --work DIR --out quant.json
    taskset -c 0-7 python scripts/vitpose_onnx_study.py quant-latency --threads 8 --work DIR --out qlat.json

``compare`` needs PyTorch and onepose (``[vitpose-torch]`` + manual onepose install, see
docs/guide/installation.md); the ``rtmw`` quantization cases need ``rtmlib``;
``quant`` also needs ``onnx``. Weights come from ``get_weight`` (set ``WHOLEBODYPOSE_HOME``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIZES = ["small", "base", "large", "huge"]
VIDEOS = [ROOT / "data/videos/videoTest.mp4", *sorted((ROOT / "hf/space/examples").glob("*.mp4"))]
IMAGES = [*sorted((ROOT / "data/images").glob("*.jp*g")), *sorted((ROOT / "hf/space/examples").glob("*.jpg"))]
BENCH_IMAGE = ROOT / "data/images/image.jpg"


def read_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def video_frames(path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def inputs(frames_per_video: int) -> list[tuple[str, np.ndarray]]:
    items = [(p.name, read_rgb(p)) for p in IMAGES]
    for v in VIDEOS:
        items += [(f"{v.name}#{i}", f) for i, f in enumerate(video_frames(v, frames_per_video))]
    return items


def stats(d: np.ndarray) -> dict:
    d = np.asarray(d, np.float64).ravel()
    if d.size == 0:
        return {"n": 0}
    return {
        "n": int(d.size),
        "median": float(np.median(d)),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
        "frac_gt_0.5px": float(np.mean(d > 0.5)),
    }


def vitpose(size: str, engine: str, detector: str | None = None):
    from wholebodypose.models.vitpose.model import VITPoseModel

    return VITPoseModel(
        device="cpu", model_name=f"ViTPose+_{size}_coco_wholebody", engine=engine, detector=detector, verbose=False
    )


# ------------------------------------------------------------------------------------------- compare
def cmd_compare(args) -> dict:
    data = inputs(args.frames)
    multi = [(n, im) for n, im in data if "#" in n]
    out = {"n_images_full_frame": len(data), "n_frames_multi": len(multi), "models": {}}
    for size in args.sizes:
        res = {}
        for mode, items, det in (("full_frame", data, None), ("multi_person_yolox_m", multi, "yolox_m")):
            t, o = vitpose(size, "torch", det), vitpose(size, "onnx", det)
            dist, sdiff, people_mismatch = [], [], 0
            for _, im in items:
                kt, st = t.predict(im)
                ko, so = o.predict(im)
                if kt.shape != ko.shape:
                    people_mismatch += 1
                    continue
                dist.append(np.linalg.norm(kt - ko, axis=-1).ravel())
                sdiff.append(np.abs(st - so).ravel())
            dist = np.concatenate(dist) if dist else np.zeros(0)
            sdiff = np.concatenate(sdiff) if sdiff else np.zeros(0)
            res[mode] = {
                "keypoint_px": stats(dist),
                "score_abs_diff": {"median": float(np.median(sdiff)), "max": float(sdiff.max())},
                "frames_with_different_people_count": people_mismatch,
            }
            del t, o
        out["models"][size] = res
        print(size, json.dumps(res))
    return out


# --------------------------------------------------------------------------------------------- bench
def _time(fn, repeats: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return ts


def _ort_session(path: str, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def cmd_bench(args) -> dict:
    import torch

    from wholebodypose.models.vitpose.onnx_runtime import ViTPoseOnnx
    from wholebodypose.weights import get_weight

    torch.set_num_threads(args.threads)
    im = read_rgb(BENCH_IMAGE)
    out = {"image": f"{BENCH_IMAGE.relative_to(ROOT)} {im.shape[1]}x{im.shape[0]}", "threads": args.threads}
    # Every configuration is built first, then timed round-robin (one call each per round), so that
    # all of them see the same machine load. Median and minimum over the rounds are reported.
    runs = {}
    for size in args.sizes:
        t = vitpose(size, "torch")
        o = vitpose(size, "onnx")
        o.onnx = ViTPoseOnnx(str(get_weight(f"vitpose_{size}_onnx")), device="cpu", intra_op_num_threads=args.threads)
        runs[f"vitpose_{size}_torch"] = t
        runs[f"vitpose_{size}_onnx"] = o
    from wholebodypose.models.rtmpose.model import RTMPoseModel

    for mode in ("performance", "lightweight"):
        r = RTMPoseModel(device="cpu", mode=mode, verbose=False)
        for tool in (r.model.det_model, r.model.pose_model):
            tool.session = _ort_session(tool.onnx_model, args.threads)
        runs[f"rtmpose_{mode}"] = r
    times = {k: [] for k in runs}
    for m in runs.values():
        for _ in range(3):
            m.predict(im)
    for _ in range(args.repeats):
        for k, m in runs.items():
            t0 = time.perf_counter()
            m.predict(im)
            times[k].append((time.perf_counter() - t0) * 1000)
    rows = {k: {"median_ms": statistics.median(v), "min_ms": min(v)} for k, v in times.items()}
    for k, v in rows.items():
        print(k, v, flush=True)
    out["rows"] = rows
    return out


# --------------------------------------------------------------------------------------------- quant
def unfold_fp16_storage(src: str, dst: str) -> None:
    """Replace float16 initializer + Cast(float) pairs with plain float32 initializers."""
    import onnx
    from onnx import TensorProto, numpy_helper

    m = onnx.load(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    keep_nodes, new_inits = [], {}
    for node in g.node:
        if (
            node.op_type == "Cast"
            and node.input[0] in inits
            and inits[node.input[0]].data_type == TensorProto.FLOAT16
            and node.attribute[0].i == TensorProto.FLOAT
        ):
            arr = numpy_helper.to_array(inits[node.input[0]]).astype(np.float32)
            new_inits[node.input[0]] = numpy_helper.from_array(arr, node.output[0])
        else:
            keep_nodes.append(node)
    out_inits = [new_inits.get(i.name, i) for i in g.initializer]
    del g.initializer[:]
    g.initializer.extend(out_inits)
    del g.node[:]
    g.node.extend(keep_nodes)
    onnx.save(m, dst)


def make_variants(name: str, fp32: str, work: Path) -> dict[str, str]:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from onnxruntime.transformers.float16 import convert_float_to_float16

    work.mkdir(parents=True, exist_ok=True)
    pre = work / f"{name}.pre.onnx"
    quant_pre_process(fp32, str(pre), skip_symbolic_shape=True)
    variants = {"fp32": fp32}
    int8 = work / f"{name}.int8.onnx"
    quantize_dynamic(str(pre), str(int8), weight_type=QuantType.QUInt8)
    variants["int8_dynamic"] = str(int8)
    fp16 = work / f"{name}.fp16.onnx"
    onnx.save(convert_float_to_float16(onnx.load(fp32), keep_io_types=True), str(fp16))
    variants["fp16"] = str(fp16)
    return variants


def _deviation(ref_k, ref_s, k, s, thr) -> dict:
    d = np.linalg.norm(ref_k - k, axis=-1)
    vis = ref_s > thr
    return {
        "all": stats(d),
        "visible": stats(d[vis]),
        "mask_agreement": float(np.mean((ref_s > thr) == (s > thr))),
        "n_visible_ref": int(vis.sum()),
        "n_visible_variant": int((s > thr).sum()),
    }


def cmd_quant(args) -> dict:
    from rtmlib import RTMPose

    from wholebodypose.models.rtmpose.model import RTMPoseModel
    from wholebodypose.models.vitpose.onnx_runtime import ViTPoseOnnx
    from wholebodypose.weights import get_weight

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    data = inputs(args.frames)
    out = {"threads": args.threads, "n_inputs": len(data), "models": {}}

    # ViTPose-base: unfold the fp16 storage first so that quantizers see float32 weights.
    vb = work / "vitpose_base.fp32.onnx"
    unfold_fp16_storage(str(get_weight("vitpose_base_onnx")), str(vb))
    # RTMW-l 384x288 (paper RTMPose model).
    rx = str(get_weight("rtmw_x_384"))

    dummy_v = np.zeros((1, 3, 256, 192), np.float32)
    dummy_r = np.zeros((1, 3, 384, 288), np.float32)

    for name, fp32, thr in (("vitpose_base", str(vb), 0.5), ("rtmw_x_384", rx, 2.5)):
        variants = make_variants(name, fp32, work)
        if name == "vitpose_base":
            ref = None
            res = {}
            for vname, path in variants.items():
                m = ViTPoseOnnx(path, device="cpu", intra_op_num_threads=args.threads)
                ks, ss = zip(*(m([im]) for _, im in data), strict=True)
                k, s = np.concatenate(ks), np.concatenate(ss)
                sess = m.session
                lat = statistics.median(_time(lambda s_=sess: s_.run(None, {"input": dummy_v}), args.repeats))
                ref = (k, s) if vname == "fp32" else ref
                res[vname] = {"size_mib": Path(path).stat().st_size / 2**20, "latency_ms": lat}
                if vname != "fp32":
                    res[vname]["deviation"] = _deviation(ref[0], ref[1], k, s, thr)
                print(name, vname, json.dumps(res[vname]))
        else:
            base = RTMPoseModel(device="cpu", mode="performance", verbose=False)
            base.model.det_model.session = _ort_session(base.model.det_model.onnx_model, args.threads)
            boxes = [base.model.det_model(cv2.cvtColor(im, cv2.COLOR_RGB2BGR)) for _, im in data]
            ref = None
            res = {}
            for vname, path in variants.items():
                pose = RTMPose(path, model_input_size=(288, 384), backend="onnxruntime", device="cpu")
                pose.session = _ort_session(path, args.threads)
                ks, ss = [], []
                for (_, im), b in zip(data, boxes, strict=True):
                    k, s = pose(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), bboxes=b)
                    ks.append(np.asarray(k, np.float32).reshape(-1, 133, 2))
                    ss.append(np.asarray(s, np.float32).reshape(-1, 133))
                k, s = np.concatenate(ks), np.concatenate(ss)
                iname = pose.session.get_inputs()[0].name
                lat = statistics.median(_time(lambda p=pose, n=iname: p.session.run(None, {n: dummy_r}), args.repeats))
                ref = (k, s) if vname == "fp32" else ref
                res[vname] = {"size_mib": Path(path).stat().st_size / 2**20, "latency_ms": lat, "people": len(k)}
                if vname != "fp32":
                    res[vname]["deviation"] = _deviation(ref[0], ref[1], k, s, thr)
                print(name, vname, json.dumps(res[vname]))
        out["models"][name] = res
    return out


def cmd_quant_latency(args) -> dict:
    """Latency of the variants written by ``quant``, timed round-robin so they share the machine load."""
    from wholebodypose.weights import get_weight

    work = Path(args.work)
    shapes = {"vitpose_base": (1, 3, 256, 192), "rtmw_x_384": (1, 3, 384, 288)}
    files = {
        "vitpose_base": {
            "fp32 (shipped, fp16-stored)": str(get_weight("vitpose_base_onnx")),
            "fp32": str(work / "vitpose_base.fp32.onnx"),
            "int8_dynamic": str(work / "vitpose_base.int8.onnx"),
            "fp16": str(work / "vitpose_base.fp16.onnx"),
        },
        "rtmw_x_384": {
            "fp32": str(get_weight("rtmw_x_384")),
            "int8_dynamic": str(work / "rtmw_x_384.int8.onnx"),
            "fp16": str(work / "rtmw_x_384.fp16.onnx"),
        },
    }
    out = {"threads": args.threads, "rounds": args.repeats, "models": {}}
    for model, variants in files.items():
        x = np.random.default_rng(0).standard_normal(shapes[model]).astype(np.float32)
        sess = {v: _ort_session(p, args.threads) for v, p in variants.items()}
        feeds = {v: {s.get_inputs()[0].name: x} for v, s in sess.items()}
        times = {v: [] for v in sess}
        for v, s in sess.items():  # warm-up
            for _ in range(3):
                s.run(None, feeds[v])
        for _ in range(args.repeats):
            for v, s in sess.items():
                t0 = time.perf_counter()
                s.run(None, feeds[v])
                times[v].append((time.perf_counter() - t0) * 1000)
        out["models"][model] = {
            v: {
                "median_ms": statistics.median(t),
                "min_ms": min(t),
                "size_mib": Path(variants[v]).stat().st_size / 2**20,
            }
            for v, t in times.items()
        }
        print(model, json.dumps(out["models"][model]), flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("compare", "bench", "quant", "quant-latency"):
        p = sub.add_parser(name)
        p.add_argument("--sizes", nargs="+", default=SIZES)
        p.add_argument("--frames", type=int, default=10, help="frames sampled per video")
        p.add_argument("--threads", type=int, default=8)
        p.add_argument("--repeats", type=int, default=30)
        p.add_argument("--work", default="onnx_study_work")
        p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cmds = {"compare": cmd_compare, "bench": cmd_bench, "quant": cmd_quant, "quant-latency": cmd_quant_latency}
    res = cmds[args.cmd](args)
    args.out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
