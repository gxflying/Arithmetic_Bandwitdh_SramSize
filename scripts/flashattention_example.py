"""Replay the single-head, two-KV-tile FlashAttention forward example in section 5."""
import argparse
import json
from attention_timing import attention_footprint, query_tile
from pingpong_tradeoff import Rates


def example(hw):
    p = dict(d=128, dv=128, kv_len=128, q_bytes=2, k_bytes=2, v_bytes=2, out_bytes=2,
             score_bytes=4, prob_bytes=2, partial_bytes=4)
    for op, gops in (("mul", 2000), ("add", 2000), ("exp", 250), ("reduce", 1000), ("inv", 125)):
        p[f"vec_{op}_gops"] = gops
        p[f"vec_{op}_ns"] = 0
    result = query_tile(32, 64, hw, p, trace=True)
    result["sram_capacity_bytes"] = attention_footprint(32, 64, 2, p)
    result["average_GBs"] = {k: v/result["double_ns"] for k, v in result["totals"].items()}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name, value in (("ddr-read-gbs",512), ("ddr-write-gbs",256),
                        ("sram-read-gbs",4000), ("sram-write-gbs",4000)):
        p.add_argument("--"+name, type=float, default=value)
    args = vars(p.parse_args())
    import math
    if any(not math.isfinite(v) or v <= 0 for v in args.values()):
        p.error("bandwidth must be finite and positive")
    print(json.dumps(example(Rates(**args)), indent=2))


if __name__ == "__main__":
    main()
