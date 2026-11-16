"""在相同可用 SRAM 下搜索各策略的 tile，扫描固定参数与离散参数网格。"""
import argparse
import csv
import json
import math
import textwrap
from itertools import product
from pathlib import Path

from attention_timing import attention_footprint, attention_results
from pingpong_tradeoff import Rates, evaluate
from vector_timing import vector_result


WIDTHS = {"int8": 1, "uint8": 1, "fp8": 1, "bf16": 2, "fp16": 2, "int16": 2,
          "fp32": 4, "int32": 4, "fp64": 8, "int64": 8}
# Logical axes, stored tensors, tile axes in model order, active compute engines.
SPECS = {
    "gemm": (("M", "K", "N"), ("x", "w", "y", "partial"), ("m", "n", "k")),
    "flashattention": (("q_len", "kv_len", "d", "dv"),
                       ("q", "k", "v", "score", "prob", "partial", "out"), ("r", "c")),
    "online_softmax": (("rows", "length"), ("x", "out"), ("v",)),
    "swiglu": (("rows", "length"), ("gate", "up", "out"), ("v",)),
}
POLICIES = {"single": (1, 1), "X_only": (2, 1), "W_only": (1, 2), "double": (2, 2)}
TENSOR = {"tensor_tflops", "tensor_ns"}
PORTS = {f"{port}_{unit}" for port in ("ddr_read", "ddr_write", "sram_read", "sram_write", "dma")
         for unit in ("gbs", "ns")}
VOPS = ("add", "mul", "reduce", "exp", "inv")
TRAFFIC = ("ddr_read_bytes", "ddr_write_bytes", "sram_read_bytes", "sram_write_bytes")


def object_fields(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是 JSON 对象")
    return value


def positive_int(value):
    return type(value) is int and value > 0


def byte_width(value):
    if positive_int(value):
        return value
    if isinstance(value, str) and value.lower() in WIDTHS:
        return WIDTHS[value.lower()]
    raise ValueError(f"不支持的 dtype {value!r}；请给出类型名称或每元素正整数字节")


def validate(cfg):
    object_fields(cfg, "配置")
    extra = cfg.keys() - {"kernel", "fixed", "sweep", "tiles", "plot_at"}
    if extra:
        raise ValueError(f"未知顶层字段 {sorted(extra)}")
    kernel = cfg.get("kernel")
    if not isinstance(kernel, str) or kernel not in SPECS:
        raise ValueError(f"kernel 须选自 {list(SPECS)}")
    dims, tensors, axes = SPECS[kernel]
    fixed = object_fields(cfg.get("fixed", {}), "fixed")
    sweep = object_fields(cfg.get("sweep", {}), "sweep")
    tiles = object_fields(cfg.get("tiles"), "tiles")
    anchor = object_fields(cfg.get("plot_at", {}), "plot_at")
    if tiles.keys() != set(axes):
        raise ValueError(f"tiles 须含且仅含 {axes}")
    for key, values in tiles.items():
        if not isinstance(values, list) or not values or not all(map(positive_int, values)):
            raise ValueError(f"tiles.{key} 须为非空正整数列表")
    if fixed.keys() & sweep.keys():
        raise ValueError(f"fixed 与 sweep 重复定义 {sorted(fixed.keys() & sweep.keys())}")
    types = {name+"_dtype" for name in tensors}
    vector = {f"vec_{op}_{unit}" for op in VOPS for unit in ("gops", "ns")}
    keys = fixed.keys() | sweep.keys()
    required = set(dims) | types | PORTS | {"capacity_kib"}
    if kernel in ("gemm", "flashattention"):
        required |= TENSOR
    if kernel != "gemm":
        required |= {key for key in vector if kernel != "swiglu" or "reduce" not in key}
    unknown = keys - (required | TENSOR | vector)
    if required-keys or unknown:
        raise ValueError(f"缺少字段 {sorted(required-keys)}；未知字段 {sorted(unknown)}")
    if sweep.keys()-required:
        raise ValueError(f"该算子未使用以下扫描参数 {sorted(sweep.keys()-required)}")
    for key in keys:
        values = sweep.get(key, [fixed.get(key)])
        if not isinstance(values, list) or not values:
            raise ValueError(f"{key} 的 sweep 须为非空列表")
        for v in values:
            if key in types:
                byte_width(v)
            elif key in dims:
                if not positive_int(v):
                    raise ValueError(f"{key} 须为正整数")
            elif (type(v) not in (int, float) or not math.isfinite(v)
                  or (v < 0 if key.endswith("_ns") else v <= 0)):
                raise ValueError(f"{key} 须为有限数，延迟 ≥ 0，带宽、算力和容量 > 0")
        if len(set(values)) != len(values):
            raise ValueError(f"{key} 的 sweep 值重复")
    for key, value in anchor.items():
        if key not in sweep or value not in sweep[key]:
            raise ValueError(f"plot_at.{key} 必须选自相应 sweep 列表")


def point(kernel, params, candidates):
    dims, tensors, axes = SPECS[kernel]
    p = dict(params)
    for name in tensors:
        p[name+"_bytes"] = byte_width(p[name+"_dtype"])
    hw = Rates(**{k: p[k] for k in Rates.__dataclass_fields__ if k in p})
    if kernel == "gemm":
        limits = (p["M"], p["N"], p["K"])
    elif kernel == "flashattention":
        limits = (p["q_len"], p["kv_len"])
    else:
        limits = (p["length"],)
    grid = [sorted({min(n, size) for n in candidates[axis]}) for axis, size in zip(axes, limits)]
    policies = POLICIES if kernel in ("gemm", "swiglu") else ("single", "double")
    best = dict.fromkeys(policies)
    capacity = p["capacity_kib"] * 1024
    for tile in product(*grid):
        attention = {}
        for policy in policies:
            if kernel == "gemm":
                slots = POLICIES[policy]
                if hw.footprint(*tile, slots) > capacity:
                    continue
                result = evaluate(tuple(p[k] for k in dims), tile, slots, hw)
            elif kernel == "flashattention":
                if attention_footprint(*tile, POLICIES[policy][0], p) > capacity:
                    continue
                if not attention:
                    attention = attention_results(tile, hw, p)
                result = attention[policy]
            else:
                result = vector_result(kernel, tile[0], policy, hw, p)
            if result["bytes_required"] > capacity:
                continue
            ordering = (result["ms"], result["ddr_read_bytes"]+result["ddr_write_bytes"],
                        result["bytes_required"], tile)
            if best[policy] is None or ordering < best[policy][0]:
                best[policy] = (ordering, result)
    baseline = best["single"][1]["ms"] if best["single"] else None
    rows = []
    for policy, choice in best.items():
        row = dict(policy=policy, status="infeasible", capacity_kib=p["capacity_kib"])
        if choice:
            v = choice[1]
            row.update(status="ok", ms=v["ms"], gain_ms=baseline-v["ms"], speedup=baseline/v["ms"],
                       bytes_required=v["bytes_required"], **{k: v[k] for k in TRAFFIC},
                       **{"tile_"+a: n for a, n in zip(axes, v["tile"])})
        rows.append(row)
    return rows


def scan(cfg):
    validate(cfg)
    sweep = cfg.get("sweep", {})
    for index, values in enumerate(product(*sweep.values()), 1):
        coords = dict(zip(sweep, values))
        for row in point(cfg["kernel"], {**cfg.get("fixed", {}), **coords}, cfg["tiles"]):
            yield {"point": index, **coords, **row}


def save_plot(path, cfg, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = cfg.get("sweep", {})
    anchors = {k: cfg.get("plot_at", {}).get(k, vals[0]) for k, vals in sweep.items()}
    port_map = {"sram_read_gbs", "sram_write_gbs"} <= sweep.keys()
    height = max(1, len(sweep)) + int(port_map)
    fig, plots = plt.subplots(height, 2, squeeze=False,
                              figsize=(12, 3.0*height+0.7), layout="constrained")
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    max_ms = max((r["ms"] for r in rows if r["status"] == "ok"), default=1)
    factor, unit = (1e6, "ns") if max_ms < .001 else (1, "ms")
    for axis, (latency, speedup) in zip(sweep or [None], plots):
        for i, policy in enumerate(policies):
            subset = [r for r in rows if r["policy"] == policy and
                      all(r[k] == v for k, v in anchors.items() if k != axis)]
            if axis and all(type(v) in (int, float) for v in sweep[axis]):
                subset.sort(key=lambda r: r[axis])
                x = [r[axis] for r in subset]
            elif axis:
                subset.sort(key=lambda r: sweep[axis].index(r[axis]))
                x = [sweep[axis].index(r[axis]) for r in subset]
                for ax in (latency, speedup):
                    ax.set_xticks(range(len(sweep[axis])), [str(v) for v in sweep[axis]])
            else:
                x = [i]
                for ax in (latency, speedup):
                    ax.set_xticks(range(len(policies)), policies)
            for ax, field in ((latency, "ms"), (speedup, "speedup")):
                scale = factor if field == "ms" else 1
                ax.plot(x, [scale*r.get(field, math.nan) for r in subset], marker="o", label=policy)
                ax.set_xlabel(axis or "policy")
                ax.grid(True, alpha=.25)
        latency.set_ylabel(f"Latency ({unit})")
        speedup.set_ylabel("Speedup vs single")
        speedup.axhline(1, ls=":", color="gray")
    plots[0][0].legend(fontsize=8)
    caption = ", ".join(f"{k}={v}" for k, v in anchors.items())
    if port_map:
        read, write = "sram_read_gbs", "sram_write_gbs"
        rates_r, rates_w = sorted(sweep[read]), sorted(sweep[write])
        selected = {(r[read], r[write]): r for r in rows if r["policy"] == "double" and
                    all(r[k] == v for k, v in anchors.items() if k not in (read, write))}
        for ax, field, scale, title in zip(plots[-1], ("ms", "speedup"), (factor, 1),
                                          (f"Double latency ({unit})", "Double speedup vs single")):
            grid = [[selected[br, bw].get(field, math.nan)*scale for br in rates_r] for bw in rates_w]
            cells = ax.imshow(grid, origin="lower", aspect="auto")
            ax.set(title=title, xlabel=read, ylabel=write)
            ax.set_xticks(range(len(rates_r)), rates_r)
            ax.set_yticks(range(len(rates_w)), rates_w)
            fig.colorbar(cells, ax=ax)
    fig.suptitle(cfg["kernel"] + (("\nOther axes fixed at " + textwrap.fill(caption, 100)) if sweep else ""))
    fig.savefig(path, dpi=135)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="每个实验使用独立输出目录")
    parser.add_argument("--no-plot", action="store_true", help="仅输出 CSV 与配置，无需 matplotlib")
    args = parser.parse_args()
    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
        rows = list(scan(cfg))
        args.out.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(["point", *cfg.get("sweep", {}), "capacity_kib", "policy", "status",
                                    *("tile_"+a for a in SPECS[cfg["kernel"]][2]),
                                    "ms", "gain_ms", "speedup", "bytes_required", *TRAFFIC]))
        with (args.out/"results.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fields)
            writer.writeheader()
            writer.writerows(rows)
        (args.out/"config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        if not args.no_plot:
            save_plot(args.out/"curves.png", cfg, rows)
        print(f"{rows[-1]['point']} 个配置点，{len(rows)} 条策略记录，"
              f"{sum(r['status']=='infeasible' for r in rows)} 条容量不可行。输出 {args.out}")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"评估失败：{exc}\n")


if __name__ == "__main__":
    main()
