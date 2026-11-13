"""Reproduce the numeric tables and design-space figure in inference_resource_model.md."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from inference_resources import Hardware, Model, step


CASES = {
    "decode_1": ([(1, 1, 4096)], 1),
    "decode_32": ([(32, 1, 4096)], 32),
    "decode_64": ([(64, 1, 4096)], 64),
    "decode_32_long": ([(32, 1, 16384)], 32),
    "prefill_2048": ([(1, 2048, 0)], 1),
    "mixed_32_plus_256": ([(32, 1, 4096), (1, 256, 2048)], 32),
}
SCANS = {
    "base": {}, "2x_DDR": {"ddr_gbs": 1024}, "2x_Tensor": {"tensor_tflops": 256},
    "half_Tensor": {"tensor_tflops": 64}, "2x_SRAM_BW": {"sram_tbs": 16},
    "4x_capacity": {"sram_kib": 2048}, "slow_EXP": {"exp_tops": 0.015625},
    "slow_reduce": {"reduce_tops": 0.0625},
    "balanced_prefill": {"ddr_gbs": 1024, "sram_kib": 2048, "sram_tbs": 16},
}


def config_run(name, hw):
    groups, rows = CASES[name]
    return step(groups, rows, hw)


def benchmark_comparison():
    # Single-stream service-rate diagnostic with one fitted parameter, independent
    # of the hypothetical accelerator's explicit scratchpad schedule.
    m = Model()
    h, k = m.hidden, m.kv_heads * m.dim
    layer_params = 2 * h * h + 2 * h * k + 3 * h * m.ffn
    weights = m.weight_bytes * (m.layers * layer_params + h * m.vocab)
    unit_kv = 2 * m.layers * k * m.kv_bytes
    outputs, tensor_rate = 2048, 312e12
    steps = outputs - 1
    observations = [(1, 84.28), (6144, 80.70), (14336, 77.69), (30720, 70.33)]
    rows = []
    for prompt, obs in observations:
        prefill_flops = (2 * m.layers * layer_params * prompt + 2 * h * m.vocab
                         + 2 * m.layers * h * prompt * (prompt + 1))
        prefill_bytes = weights + unit_kv * prompt
        # Each decode scans the prior prefix once and appends one entry.
        decode_bytes = (steps * weights
                        + unit_kv * (steps * (prompt + 1) + steps * (steps - 1) // 2))
        if prompt == 1:
            effective_bw = (prefill_bytes + decode_bytes) / (outputs / obs)
        prefill_s = max(prefill_bytes / effective_bw, prefill_flops / tensor_rate)
        prediction = outputs / (prefill_s + decode_bytes / effective_bw)
        rows.append(dict(prompt=prompt, measured_tok_s=obs, model_tok_s=prediction,
                         relative_error=prediction / obs - 1, prefill_s=prefill_s))
    return dict(fitted_B_s=effective_bw, flash_decode_heldout_us=64.4 * 2,
                flash_decode_measured_us=106.6, qwen=rows)


def figure(study, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caps = study["capacities_KiB"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.25), layout="constrained")
    labels = {"prefill_2048": "Prefill 2,048", "decode_32": "Decode B=32, prefix 4k",
              "decode_1": "Decode B=1, prefix 4k", "decode_32_long": "Decode B=32, prefix 16k"}
    colors = ["#9d412f", "#1f6381", "#479376", "#6c618f"]
    for (key, label), color in zip(labels.items(), colors):
        times = [r["ms"] for r in study["capacity_sweep"][key]]
        left.plot(caps, times, marker="o", lw=2, ms=4, color=color, label=label)
    left.set_xscale("log", base=2)
    left.set_xticks([64, 256, 1024, 4096], ["64", "256", "1,024", "4,096"])
    left.set_yscale("log")
    left.set(xlabel="Allocatable shared SRAM (KiB)", ylabel="Model step time (ms)",
             title="Capacity changes tile reuse")
    left.grid(alpha=0.2, which="both")
    left.legend(fontsize=8, frameon=False)
    keys = ["2x_Tensor", "2x_DDR", "2x_SRAM_BW", "4x_capacity", "balanced_prefill"]
    names = ["2× Tensor", "2× DDR BW", "2× SRAM BW", "4× SRAM size", "Joint upgrade"]
    for j, key in enumerate(["decode_32", "prefill_2048"]):
        data = study["sensitivity"][key]
        ratios = [data["base"]["ms"] / data[opt]["ms"] for opt in keys]
        coords = [4 - i + (0.18 if j == 0 else -0.18) for i in range(len(keys))]
        right.barh(coords, ratios, height=0.34, color=[colors[1], colors[0]][j],
                   label=["Decode B=32", "Prefill 2,048"][j])
    right.set_yticks(range(4, -1, -1), names)
    right.axvline(1, ls="--", lw=1, color="#373737")
    right.set(xlabel="Modeled speedup over base (×)", title="Hardware sensitivity")
    right.legend(fontsize=8, frameon=False)
    fig.suptitle("Qwen2.5-7B-Instruct BF16  |  hypothetical accelerator", fontsize=13)
    fig.savefig(folder / "resource_sensitivity.png", dpi=165)
    plt.close(fig)


def summary(v):
    return dict(ms=v["seconds"] * 1000, aggregate_floor_ms=v["aggregate_floor_s"] * 1000,
                ddr_GB=v["work"]["ddr"] / 1e9, sram_GB=v["work"]["sram"] / 1e9,
                limiting_ms={k: t * 1000 for k, t in v["stage_limiting_seconds"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]
                        / "artifacts" / "inference_resources")
    parser.add_argument("--plot", action="store_true", help="render with matplotlib")
    args = parser.parse_args()
    caps = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    base = Hardware()
    study = dict(hardware=asdict(base), model=asdict(Model()), groups=CASES,
                 baseline={name: config_run(name, base) for name in CASES},
                 capacities_KiB=caps, performance_changes=SCANS,
                 comparison=benchmark_comparison())
    study["capacity_sweep"] = {
        name: [summary(config_run(name, replace(base, sram_kib=c))) for c in caps]
        for name in CASES}
    study["sensitivity"] = {
        name: {opt: summary(config_run(name, replace(base, **changes)))
               for opt, changes in SCANS.items()} for name in ["decode_32", "prefill_2048"]}
    study["prefill_512_chunks"] = [
        summary(step([(1, 512, p)], int(p == 1536), base)) for p in range(0, 2048, 512)]
    study["isolated_partial_chunk"] = summary(step([(1, 256, 2048)], 0, base))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "study.json").write_text(json.dumps(study, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    if args.plot:
        figure(study, args.out)


if __name__ == "__main__":
    main()
