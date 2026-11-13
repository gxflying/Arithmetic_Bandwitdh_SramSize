"""Dense causal GQA inference. Analytical device model, SI rates, binary capacity.

Run with --help. No accelerator or third-party Python packages are required.
"""
import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import lru_cache


@dataclass(frozen=True)
class Model:
    layers: int = 28
    hidden: int = 3584
    ffn: int = 18944
    heads: int = 28
    kv_heads: int = 4
    dim: int = 128
    vocab: int = 152064
    weight_bytes: int = 2
    act_bytes: int = 2
    kv_bytes: int = 2


@dataclass(frozen=True)
class Hardware:
    ddr_gbs: float = 512
    tensor_tflops: float = 128
    add_tops: float = 2
    mul_tops: float = 2
    exp_tops: float = 0.25
    reduce_tops: float = 1
    inv_tops: float = 0.125
    sram_tbs: float = 8
    sram_kib: float = 512

    def rates(self):
        return dict(ddr=self.ddr_gbs * 1e9, sram=self.sram_tbs * 1e12,
                    tensor=self.tensor_tflops * 1e12, add=self.add_tops * 1e12,
                    mul=self.mul_tops * 1e12, exp=self.exp_tops * 1e12,
                    reduce=self.reduce_tops * 1e12, inv=self.inv_tops * 1e12)


def ceildiv(x, y):
    return (x + y - 1) // y


def tiles(size, maximum):
    values = {min(size, maximum)}
    candidate = 1
    while candidate < min(size, maximum):
        values.add(candidate)
        candidate *= 2
    return sorted(values)


def select(candidates, rates, capacity):
    best = None
    for work, footprint, tile in candidates:
        if footprint > capacity:
            continue
        time = {r: value / rates[r] for r, value in work.items()}
        bottleneck = max(time, key=time.get)
        duration = time[bottleneck]
        key = (duration, work["ddr"], work["sram"], footprint)
        if best is None or key < best[0]:
            best = (key, dict(seconds=duration, bottleneck=bottleneck, work=work,
                              tile=tile, footprint_bytes=footprint))
    if best is None:
        raise ValueError("SRAM cannot hold any candidate tile for this stage")
    return best[1]


def gemm_candidates(m, k, n, model, panel_k):
    a, w = model.act_bytes, model.weight_bytes
    panel = min(panel_k, k)
    for rows in tiles(m, 4096):
        for cols in tiles(n, 4096):
            accum = 4 * rows * cols
            x_buffer = 2 * panel * a * rows
            w_buffer = 2 * panel * w * cols
            x = a * m * k
            w_total = w * k * n
            output = a * m * n
            x_reads = x * ceildiv(n, cols)
            w_reads = w_total * ceildiv(m, rows)
            engine = x_reads + w_reads + output
            partials = 8 * m * n * (ceildiv(k, panel) - 1)
            policies = (
                ("stream_panels", x_reads, w_reads, x_buffer + w_buffer),
                ("keep_weight_strip", x_reads, w_total, x_buffer + 2 * k * w * cols),
                ("keep_input_strip", x, w_reads, 2 * k * a * rows + w_buffer),
            )
            for policy, x_ddr, w_ddr, buffers in policies:
                ddr = x_ddr + w_ddr + output
                yield (dict(ddr=ddr, sram=ddr + engine + partials, tensor=2 * m * k * n),
                       buffers + accum, dict(m=rows, n=cols, k=panel, policy=policy,
                                             weight_bytes=w_ddr))


@lru_cache(maxsize=4096)
def geometry(q, prefix, rows, cols):
    """Sum exact query-major tile extents. Predication clips the right edge."""
    key_positions = rectangles = row_visits = 0
    for start in range(0, q, rows):
        count = min(rows, q - start)
        end = prefix + start + count
        key_positions += end
        rectangles += count * end
        row_visits += count * ceildiv(end, cols)
    return key_positions, rectangles, row_visits


def attention_candidates(requests, model):
    h, hk, d = model.heads, model.kv_heads, model.dim
    a, b = model.act_bytes, model.kv_bytes
    group = h // hk
    total_rows = h * sum(count * q for count, q, _ in requests)
    for rows in tiles(max(q for _, q, _ in requests), 512):
        for cols in (16, 32, 64, 128, 256):
            key_positions = rectangles = visits = 0
            for count, q, prefix in requests:
                keys, scores, row_visits = geometry(q, prefix, rows, cols)
                key_positions += count * keys
                rectangles += count * h * scores
                visits += count * h * row_visits
            updates = visits - total_rows
            for heads in tiles(group, group):
                footprint = (rows * heads * ((a + 4) * d + 8 + 4 * cols)
                             + 4 * cols * d * b)
                kv = 2 * b * d * hk * ceildiv(group, heads) * key_positions
                ddr = kv + 2 * a * d * total_rows
                sram = (ddr + kv + a * d * (visits + total_rows)
                        + (8 + 2 * a) * rectangles + 2 * (4 * d + 8) * updates)
                work = dict(ddr=ddr, sram=sram, tensor=4 * d * rectangles,
                            add=rectangles + (d + 2) * updates,
                            mul=rectangles + (d + 1) * updates + d * total_rows,
                            exp=rectangles + updates, reduce=2 * (rectangles - visits) + updates,
                            inv=total_rows)
                yield work, footprint, dict(q=rows, k=cols, q_heads=heads,
                                             kv_read_bytes=kv, executed_scores=rectangles,
                                             row_tile_visits=visits)


def step(requests, logit_rows, hw=Hardware(), model=Model(), *, panel_k=64,
         resident_tiles=1, reserve_kib=0):
    """A group is (request count, fresh tokens per request, prior cached length).

    logit_rows counts requested predictor rows, including zero for partial chunks.
    This returns the selected mapping for one representative layer and its repeat.
    """
    if any(not math.isfinite(x) or x <= 0 for x in asdict(hw).values()):
        raise ValueError("hardware rates and capacity must be finite and positive")
    if any(not isinstance(v, int) or v <= 0 for v in asdict(model).values()):
        raise ValueError("model dimensions and byte widths must be positive integers")
    if model.hidden != model.heads * model.dim or model.heads % model.kv_heads:
        raise ValueError("requires hidden = heads * dim and complete GQA groups")
    if not requests or any(min(count, q) < 1 or p < 0 for count, q, p in requests):
        raise ValueError("request count/query length must be positive and prefix nonnegative")
    m = sum(count * q for count, q, _ in requests)
    if not 0 <= logit_rows <= m:
        raise ValueError("logit rows outside scheduled token count")
    pairs = sum(count * (q * p + q * (q + 1) // 2) for count, q, p in requests)
    h, f, k, a = model.hidden, model.ffn, model.kv_heads * model.dim, model.act_bytes
    if panel_k < 1 or resident_tiles < 1 or not 0 <= reserve_kib < hw.sram_kib:
        raise ValueError("require positive panel and resident tiles, and 0 <= reserve < SRAM")
    cap = (hw.sram_kib - reserve_kib) * 1024 / resident_tiles
    rates = hw.rates()
    stages = []

    def record(name, candidates, repeat):
        chosen = select(candidates, rates, cap)
        stages.append(dict(name=name, repeat=repeat, **chosen))

    def stream(name, ddr, repeat, scratch=1024, sram_extra=0, **ops):
        record(name, [(dict(ddr=ddr, sram=2 * ddr + sram_extra, **ops), scratch, {})], repeat)

    def norm(name, rows, repeat):
        # Two scale multiplies and a square per element. Add epsilon per row.
        stream(name, a * (2 * rows * h + h), repeat, scratch=2 * a * h + 1024,
               sram_extra=(rows - 1) * a * h + rows * a * h,
               mul=rows * (3 * h + 1), add=rows, reduce=rows * (h - 1), inv=rows)

    def gemm(name, rows, inner, cols, repeat):
        record(name, gemm_candidates(rows, inner, cols, model, panel_k), repeat)

    l = model.layers
    stream("embedding_gather", 2 * a * m * h, 1)
    norm("attention_rmsnorm", m, l)
    gemm("qkv", m, h, h + 2 * k, l)
    # QKV bias fetched once. RoPE cos/sin table shared across heads:
    # d trig entries per token (half d cos, half d sin); V goes to the cache.
    stream("bias_rope_kv_append",
           a * (m * (2 * h + 2 * k + model.dim) + h + 2 * k)
           + 2 * model.kv_bytes * m * k, l, scratch=a * (h + 2 * k + model.dim) + 1024,
           sram_extra=a * ((m - 1) * (h + 2 * k) + m * (2 * (h + k) - model.dim)),
           add=m * (2 * h + 3 * k), mul=2 * m * (h + k))
    record("attention", attention_candidates(requests, model), l)
    gemm("attention_out", m, h, h, l)
    stream("attention_residual", 3 * a * m * h, l, add=m * h)
    norm("ffn_rmsnorm", m, l)
    gemm("gate_up", m, h, 2 * f, l)
    stream("swiglu", 3 * a * m * f, l, mul=2 * m * f, exp=m * f,
           add=m * f, inv=m * f)
    gemm("down", m, f, h, l)
    stream("ffn_residual", 3 * a * m * h, l, add=m * h)
    if logit_rows:
        norm("final_rmsnorm", logit_rows, 1)
        gemm("lm_head", logit_rows, h, model.vocab, 1)
        stream("greedy_argmax", a * logit_rows * model.vocab + 4 * logit_rows, 1,
               reduce=logit_rows * (model.vocab - 1))

    total = Counter()
    limiting_seconds = Counter()
    for s in stages:
        total.update({r: s["repeat"] * v for r, v in s["work"].items()})
        limiting_seconds[s["bottleneck"]] += s["repeat"] * s["seconds"]
    seconds = sum(limiting_seconds.values())
    return dict(scheduled_tokens=m, causal_pairs_per_head=pairs, logit_rows=logit_rows,
                tile_budget_bytes=cap, resident_tiles=resident_tiles, panel_k=panel_k,
                reserve_kib=reserve_kib,
                seconds=seconds, scheduled_tokens_per_s=m / seconds,
                aggregate_floor_s=max(v / rates[r] for r, v in total.items()),
                work=total, stage_limiting_seconds=limiting_seconds, stages=stages)


def parse_group(arg):
    try:
        count, q, p = map(int, arg.split(","))
        return count, q, p
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use COUNT,NEW,PREFIX") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", type=parse_group,
                        help="COUNT,NEW,PREFIX (repeatable); default 1,1,4096")
    parser.add_argument("--logit-rows", type=int, help="default one per request")
    parser.add_argument("--model-config", help="local Hugging Face Qwen2-style dense config.json")
    parser.add_argument("--panel-k", type=int, default=64, help="GEMM inner panel size")
    parser.add_argument("--resident-tiles", type=int, default=1, help="equal SRAM partitions")
    parser.add_argument("--reserve-kib", type=float, default=0, help="fixed shared SRAM reservation")
    for field, value in asdict(Hardware()).items():
        parser.add_argument("--" + field.replace("_", "-"), type=float, default=value)
    args = parser.parse_args()
    groups = args.group or [(1, 1, 4096)]
    hw = Hardware(**{field: getattr(args, field) for field in asdict(Hardware())})
    model = Model()
    if args.model_config:
        with open(args.model_config, encoding="utf-8") as file:
            cfg = json.load(file)
        if cfg.get("model_type") != "qwen2" or cfg.get("use_sliding_window", False):
            parser.error("requires Qwen2 dense full causal attention")
        model = replace(model, layers=cfg["num_hidden_layers"], hidden=cfg["hidden_size"],
                        ffn=cfg["intermediate_size"], heads=cfg["num_attention_heads"],
                        kv_heads=cfg["num_key_value_heads"], vocab=cfg["vocab_size"],
                        dim=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]))
    logits = sum(b for b, _, _ in groups) if args.logit_rows is None else args.logit_rows
    try:
        result = step(groups, logits, hw, model, panel_k=args.panel_k,
                      resident_tiles=args.resident_tiles, reserve_kib=args.reserve_kib)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(dict(model=asdict(model), hardware=asdict(hw), groups=groups, **result),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
