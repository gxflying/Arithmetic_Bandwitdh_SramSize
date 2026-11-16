"""Dense single-head forward attention, query-tile outer loop. Bytes, ns, GB/s."""
from collections import Counter
from dataclasses import asdict, dataclass

from pingpong_tradeoff import edge_sizes


@dataclass(frozen=True)
class Stage:
    name: str
    ddr_read: int
    ddr_write: int
    sram_read: int
    sram_write: int
    ns: float


def vector_time(ops, params):
    """One logical startup for each nonempty primitive in dependency order."""
    return sum(params[f"vec_{op}_ns"] + count / params[f"vec_{op}_gops"]
               for op, count in ops if count)


def attention_footprint(r, c, slots, p):
    q, o = r*p["d"]*p["q_bytes"], r*p["dv"]*p["out_bytes"]
    kv = c*(p["d"]*p["k_bytes"] + p["dv"]*p["v_bytes"])
    scratch = r*c*max(p["score_bytes"], p["prob_bytes"])
    return max(q, o) + slots*kv + scratch + 2*r*p["dv"]*p["partial_bytes"]


def prefetch(stages, size, hw):
    # Pay one full load-path startup across the window and its residual wait.
    startup = hw.ddr_read_ns + hw.dma_ns + hw.sram_write_ns
    source_rate = min(hw.ddr_read_gbs, hw.dma_gbs or hw.ddr_read_gbs)
    remaining, moved = size, []
    for stage in stages:
        # Engine writes uniformly after its write-port startup. Reserve that
        # prefix for the engine, then assign the remaining write rate to DMA.
        prefix = hw.sram_write_ns if stage.sram_write else 0
        service = stage.ns - prefix
        engine_rate = stage.sram_write/service if service > 0 else 0
        spare_rate = max(0.0, hw.sram_write_gbs - engine_rate)
        active = max(0.0, stage.ns - max(startup, prefix))
        amount = min(remaining, active*min(source_rate, spare_rate))
        moved.append(amount)
        remaining -= amount
        startup = max(0.0, startup - stage.ns)
    wait = startup + remaining/min(source_rate, hw.sram_write_gbs)
    return moved, wait


def query_tile(r, c, hw, p, *, trace=False):
    """Simulate one Q tile and return both KV policies for identical geometry."""
    d, dv = p["d"], p["dv"]
    q, out = r*d*p["q_bytes"], r*dv*p["out_bytes"]
    num = r*dv*p["partial_bytes"]
    stages, prefetched, waits = [], [], []

    def compute(name, read, write, arithmetic):
        duration = max(arithmetic, hw.sram_read_ns + read/hw.sram_read_gbs,
                       hw.sram_write_ns + write/hw.sram_write_gbs)
        stages.append(Stage(name, 0, 0, read, write, duration))

    def load(name, size):
        stages.append(Stage(name, size, 0, 0, size, hw.transfer_ns(size, to_sram=True)))

    load("Q", q)
    saved = 0.0
    for offset in range(0, p["kv_len"], c):
        width = min(c, p["kv_len"]-offset)
        key, val = width*d*p["k_bytes"], width*dv*p["v_bytes"]
        a, red = r*width, r*(width-1)
        score, prob = a*p["score_bytes"], a*p["prob_bytes"]
        load(f"KV{offset//c+1}", key+val)
        first_compute = len(stages)
        compute("QK", q+key, score, hw.tensor_ns + 2*a*d/(1000*hw.tensor_tflops))
        ops = [("mul", a), ("reduce", red), ("add", a), ("exp", a), ("reduce", red)]
        if offset:
            ops += [("reduce", r), ("add", r), ("exp", r), ("mul", r), ("add", r)]
        compute("online_softmax", score, prob, vector_time(ops, p))
        compute("PV", prob+val, num, hw.tensor_ns + 2*a*dv/(1000*hw.tensor_tflops))
        if offset:
            compute("merge_numerator", 2*num, num,
                    vector_time([("mul", r*dv), ("add", r*dv)], p))
        following = min(c, p["kv_len"]-offset-width)
        if following:
            next_size = following*(d*p["k_bytes"] + dv*p["v_bytes"])
            moved, wait = prefetch(stages[first_compute:], next_size, hw)
            saved += hw.transfer_ns(next_size, to_sram=True) - wait
            if trace:
                prefetched.extend(moved)
                waits.append(wait)
    compute("normalize", num, out, vector_time([("inv", r), ("mul", r*dv)], p))
    stages.append(Stage("store_O", 0, out, out, 0, hw.transfer_ns(out, to_sram=False)))
    serial = sum(s.ns for s in stages)
    totals = {field: sum(getattr(s, field) for s in stages)
              for field in ("ddr_read", "ddr_write", "sram_read", "sram_write")}
    result = dict(totals=totals, single_ns=serial, double_ns=serial-saved, overlap_ns=saved)
    if trace:
        result.update(stages=[asdict(s) for s in stages], prefetch_bytes_per_stage=prefetched,
                      remaining_prefetch_ns=sum(waits))
    return result


def attention_results(tile, hw, p):
    r, c = tile
    totals = Counter()
    for rows, repeats in edge_sizes(p["q_len"], r):
        result = query_tile(rows, c, hw, p)
        for field, value in result["totals"].items():
            totals[field+"_bytes"] += repeats*value
        for field in ("single_ns", "double_ns"):
            totals[field] += repeats*result[field]
    return {policy: dict(tile=tile, bytes_required=attention_footprint(r, c, slots, p),
                         ms=totals[policy+"_ns"]/1e6,
                         **{f: v for f, v in totals.items() if f.endswith("_bytes")})
            for policy, slots in (("single", 1), ("double", 2))}
