"""Replay the single-head, two-KV-tile FlashAttention forward example in section 5."""
import argparse
import json
from dataclasses import asdict, dataclass

from pingpong_tradeoff import Rates


@dataclass(frozen=True)
class Stage:
    name: str
    ddr_read: int
    ddr_write: int
    sram_read: int
    sram_write: int
    ns: float


def example(hw):
    # One query tile over two KV tiles. SRAM scratch holds score/probability
    # in place, local PV result, and the running FP32 numerator.
    r, c, d = 32, 64, 128
    scores, output = r*c, r*d
    q, kv = 2*output, 4*c*d
    flops = 2*scores*d
    stages = []

    def compute(name, read, write, arithmetic):
        read_ns = hw.sram_read_ns + read/hw.sram_read_gbs if read else 0
        write_ns = hw.sram_write_ns + write/hw.sram_write_gbs if write else 0
        stages.append(Stage(name, 0, 0, read, write, max(read_ns, write_ns, arithmetic)))

    def load(name, size):
        stages.append(Stage(name, size, 0, 0, size, hw.transfer_ns(size, to_sram=True)))

    load("Q", q)
    tensor = hw.tensor_ns + flops/(1000*hw.tensor_tflops)
    for block in range(2):
        load(f"KV{block+1}", kv)
        compute("QK", q+2*c*d, 4*scores, tensor)
        # Per-row scale, max, subtract, exp, sum; second tile also updates
        # running max and denominator (alpha, scaled denominator, final add).
        vector = scores/2000 + 2*r*(c-1)/1000 + scores/2000 + scores/250
        if block:
            vector += r/1000 + r/2000 + r/250 + r/2000 + r/2000
        compute("online_softmax", 4*scores, 2*scores, vector)
        compute("PV", 2*scores+2*c*d, 4*output, tensor)
        if block:
            compute("merge_numerator", 8*output, 4*output, 2*output/2000)
    compute("normalize", 4*output, q, r/125 + output/2000)
    stages.append(Stage("store_O", 0, q, q, 0, hw.transfer_ns(q, to_sram=False)))

    # DMA owns the write-port capacity left by each of the three current
    # compute stages. A deficit is drained at the KV2 consumption barrier.
    load2 = stages[5]
    if hw.ddr_read_ns or hw.dma_ns or hw.sram_write_ns:
        raise ValueError("the prefetch example assumes zero load-path startup")
    remaining = kv
    prefetched = []
    for stage in stages[2:5]:
        spare = max(0, stage.ns*hw.sram_write_gbs - stage.sram_write)
        moved = min(remaining, spare, stage.ns*min(hw.ddr_read_gbs, hw.dma_gbs or hw.ddr_read_gbs))
        prefetched.append(moved)
        remaining -= moved
    wait = hw.transfer_ns(remaining, to_sram=True)
    saved = load2.ns - wait
    serial = sum(s.ns for s in stages)
    totals = {field: sum(getattr(s, field) for s in stages)
              for field in ("ddr_read", "ddr_write", "sram_read", "sram_write")}
    footprint = q + 2*kv + 4*scores + 2*4*output
    return dict(stages=[asdict(s) for s in stages], totals=totals,
                single_ns=serial, double_ns=serial-saved, overlap_ns=saved,
                prefetch_bytes_per_stage=prefetched, remaining_prefetch_ns=wait,
                sram_capacity_bytes=footprint,
                average_GBs={k: v/(serial-saved) for k,v in totals.items()})


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
