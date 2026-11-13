"""Compare SRAM input buffering policies on an output-stationary BF16 GEMM.

Time is in ns, bandwidth in decimal GB/s, capacity in binary KiB. Input DMA
and current-panel execution may overlap within an output tile. The consumer
releases its panels on completion. The output drain is serialized across tiles.
"""
import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product

from inference_resources import Model, ceildiv, tiles


@dataclass(frozen=True)
class Rates:
    ddr_gbs: float = 512
    sram_gbs: float = 8000
    tensor_tflops: float = 128
    overlap: float = 1
    switch_ns: float = 0


def edge_sizes(length, tile):
    full, tail = divmod(length, tile)
    return [(tile, full)] + ([(tail, 1)] if tail else [])


def evaluate(shape, tile, slots, rates):
    rows, inner, cols = shape
    m, n, k = tile
    bx, bw = slots
    panels = ceildiv(inner, k)
    cap = 4 * m * n + 2 * k * (bx * m + bw * n)
    sums = Counter()
    slabs = panels - 1

    for (r, nr), (c, nc) in product(edge_sizes(rows, m), edge_sizes(cols, n)):
        count = nr * nc
        out = 2 * r * c
        # The last compute phase writes Y into SRAM; the drain reads it once.
        drain = max(out / rates.ddr_gbs, out / rates.sram_gbs)
        serial = drain
        saved = 0.0
        # All interior panels have the same cost; only the first, penultimate,
        # and last can have different accumulator I/O or a different next extent.
        indices = sorted({0, min(1, slabs), max(0, slabs - 1), slabs})
        for lo, hi in zip(indices, indices[1:] + [panels]):
            current = min(k, inner - lo * k)
            source = 2 * current * (r + c)
            partial_io = 4 * r * c * (int(lo > 0) + int(lo < slabs))
            core_io = source + partial_io + (out if lo == slabs else 0)
            dma_time = max(source / rates.ddr_gbs, source / rates.sram_gbs)
            engine_time = max(2 * r * c * current / (1000 * rates.tensor_tflops),
                              core_io / rates.sram_gbs)
            serial += (hi - lo) * (dma_time + engine_time)
            if lo < slabs:
                following = min(k, inner - (lo + 1) * k)
                prefetch = 2 * following * ((bx - 1) * r + (bw - 1) * c)
                early_time = max(prefetch / rates.ddr_gbs, prefetch / rates.sram_gbs)
                shared_time = (prefetch + core_io) / rates.sram_gbs
                joint = max(early_time, engine_time, shared_time)
                saved += (hi - lo) * max(0.0, early_time + engine_time - joint)
        sums["serial_ns"] += count * serial
        sums["raw_saved_ns"] += count * saved
        sums["output_tiles"] += count
    x_bytes = 2 * rows * inner * ceildiv(cols, n)
    w_bytes = 2 * inner * cols * ceildiv(rows, m)
    y_bytes = 2 * rows * cols
    ddr = x_bytes + w_bytes + y_bytes
    # Same SRAM ledger as the parent model for stream_panels, including DMA.
    sram = 2 * ddr + 8 * rows * cols * (panels - 1)
    switches = sums["output_tiles"] * (panels - 1) if slots != (1, 1) else 0
    captured = rates.overlap * sums["raw_saved_ns"]
    total = sums["serial_ns"] - captured + switches * rates.switch_ns
    return dict(tile=tile, slots=slots, bytes_required=cap, ddr_bytes=ddr, sram_bytes=sram,
                x_bytes=x_bytes, w_bytes=w_bytes, y_bytes=y_bytes,
                output_tiles=sums["output_tiles"], switches=switches, serial_ms=sums["serial_ns"]/1e6,
                possible_overlap_ms=sums["raw_saved_ns"]/1e6, captured_overlap_ms=captured/1e6,
                switch_ms=switches*rates.switch_ns/1e6, ms=total/1e6)


def compare(shape, kib, panels, rates, quantum=0):
    rows, inner, cols = shape
    if min(shape) < 1 or not panels or min(panels) < 1:
        raise ValueError("shape and panel sizes must be positive")
    if not all(math.isfinite(x) and x > 0 for x in
               (kib, rates.ddr_gbs, rates.sram_gbs, rates.tensor_tflops)):
        raise ValueError("capacity and resource rates must be finite and positive")
    if not 0 <= rates.overlap <= 1 or not math.isfinite(rates.switch_ns) or rates.switch_ns < 0:
        raise ValueError("overlap must be in [0,1] and switch-ns finite and nonnegative")
    if quantum < 0:
        raise ValueError("tile quantum must be nonnegative")
    def dimensions(size):
        limit = min(size, 4096)
        if quantum:
            return sorted({limit, *range(quantum, limit + 1, quantum)})
        return tiles(size, 4096)
    best = {}
    for name, slots in {"single": (1, 1), "X_only": (2, 1),
                        "W_only": (1, 2), "double": (2, 2)}.items():
        options = []
        for m, n, k in product(dimensions(rows), dimensions(cols),
                               sorted({min(inner, v) for v in panels})):
            if 4*m*n + 2*k*(slots[0]*m + slots[1]*n) <= kib*1024:
                options.append(evaluate(shape, (m, n, k), slots, rates))
        if options:
            best[name] = min(options, key=lambda v: (v["ms"], v["ddr_bytes"], v["bytes_required"]))
    if "single" not in best:
        raise ValueError("no feasible input tile at this capacity and panel size")
    return best


def main():
    model = Model()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, metavar=("M", "K", "N"),
                        default=(2048, model.hidden, 2 * model.ffn))
    parser.add_argument("--kib", type=float, nargs="+", default=[320, 384, 512])
    parser.add_argument("--panels", type=int, nargs="+", default=[64])
    parser.add_argument("--tile-quantum", type=int, default=0,
                        help="tile dimension increment; zero uses the parent power-of-two grid")
    parser.add_argument("--ddr-gbs", type=float, default=512)
    parser.add_argument("--sram-gbs", type=float, default=8000)
    parser.add_argument("--tensor-tflops", type=float, default=128)
    parser.add_argument("--overlap", type=float, default=1, help="fraction of feasible overlap captured")
    parser.add_argument("--switch-ns", type=float, default=0,
                        help="extra unhidden overhead per overlapped panel boundary")
    args = parser.parse_args()
    rates = Rates(args.ddr_gbs, args.sram_gbs, args.tensor_tflops, args.overlap, args.switch_ns)
    try:
        results = {kib: compare(args.shape, kib, args.panels, rates, args.tile_quantum)
                   for kib in args.kib}
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(dict(shape=args.shape, panels=args.panels, tile_quantum=args.tile_quantum,
                          rates=asdict(rates), results=results), indent=2))


if __name__ == "__main__":
    main()
