# Benchmarks

Speed of every backend configuration on one image with one person.

```bash
pip install "wholebodypose[mediapipe]" matplotlib
python benchmarks/speed.py --device cpu --threads 8 -o benchmarks/results/cpu.json
python benchmarks/speed.py --device cuda -o benchmarks/results/gpu.json   # needs the [gpu] extra
python benchmarks/speed.py --device coreml -o benchmarks/results/coreml.json   # macOS
python benchmarks/plot_speed.py benchmarks/results -o website/static/images/speed.png
```

`results/` holds the numbers shown in the README and in [docs/guide/backends.md](../docs/guide/backends.md#speed)
(AMD EPYC 7662 pinned to 8 cores with `taskset -c 0-7`; NVIDIA A100; Apple M4 with CoreML in `coreml.json`
and on its CPU in `m4_cpu.json`, which the chart does not use).
