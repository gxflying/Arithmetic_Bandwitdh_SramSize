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
    dma_gbs: float | None = None
    ddr_ns: float = 0
    dma_ns: float = 0
    sram_ns: float = 0
    tensor_ns: float = 0
    x_bytes: int = 2
    w_bytes: int = 2
    y_bytes: int = 2

    def footprint(self, m, n, k, slots):
        return (max(4, self.y_bytes) * m * n
                + k * (slots[0] * self.x_bytes * m + slots[1] * self.w_bytes * n))

    def transfer_ns(self, size):
        if size == 0:
            return 0.0
        bandwidth = min(self.ddr_gbs, self.sram_gbs, self.dma_gbs or self.ddr_gbs)
        return self.ddr_ns + self.dma_ns + self.sram_ns + size / bandwidth


def edge_sizes(length, tile):
    full, tail = divmod(length, tile)
    return [(tile, full)] + ([(tail, 1)] if tail else [])


def evaluate(shape, tile, slots, rates):
    rows, inner, cols = shape
    m, n, k = tile
    bx, bw = slots
    panels = ceildiv(inner, k)
    cap = rates.footprint(m, n, k, slots)
    sums = Counter()
    slabs = panels - 1

    for (r, nr), (c, nc) in product(edge_sizes(rows, m), edge_sizes(cols, n)):
        count = nr * nc
        out = rates.y_bytes * r * c
        # The last compute phase writes Y into SRAM; the drain reads it once.
        drain = rates.transfer_ns(out)
        serial = drain
        saved = 0.0
        split_cost = 0.0
        # All interior panels have the same cost; only the first, penultimate,
        # and last can have different accumulator I/O or a different next extent.
        indices = sorted({0, min(1, slabs), max(0, slabs - 1), slabs})
        for lo, hi in zip(indices, indices[1:] + [panels]):
            current = min(k, inner - lo * k)
            source = current * (rates.x_bytes * r + rates.w_bytes * c)
            partial_io = 4 * r * c * (int(lo > 0) + int(lo < slabs))
            core_io = source + partial_io + (out if lo == slabs else 0)
            dma_time = rates.transfer_ns(source)
            engine_time = max(rates.tensor_ns + 2 * r * c * current / (1000 * rates.tensor_tflops),
                              rates.sram_ns + core_io / rates.sram_gbs)
            serial += (hi - lo) * (dma_time + engine_time)
            if lo < slabs:
                following = min(k, inner - (lo + 1) * k)
                prefetch = following * ((bx - 1) * rates.x_bytes * r + (bw - 1) * rates.w_bytes * c)
                early_time = rates.transfer_ns(prefetch)
                shared_time = rates.sram_ns + (prefetch + core_io) / rates.sram_gbs
                if bx != bw:
                    # Issuing separate early and late loads starts the path twice.
                    split_cost += (hi - lo) * (rates.ddr_ns + rates.dma_ns + rates.sram_ns)
                joint = max(early_time, engine_time, shared_time)
                saved += (hi - lo) * max(0.0, early_time + engine_time - joint)
        sums["serial_ns"] += count * serial
        sums["raw_saved_ns"] += count * saved
        sums["split_ns"] += count * split_cost
        sums["output_tiles"] += count
    x_bytes = rates.x_bytes * rows * inner * ceildiv(cols, n)
    w_bytes = rates.w_bytes * inner * cols * ceildiv(rows, m)
    y_bytes = rates.y_bytes * rows * cols
    ddr = x_bytes + w_bytes + y_bytes
    # Same SRAM ledger as the parent model for stream_panels, including DMA.
    sram = 2 * ddr + 8 * rows * cols * (panels - 1)
    switches = sums["output_tiles"] * (panels - 1) if slots != (1, 1) else 0
    captured = rates.overlap * sums["raw_saved_ns"]
    total = sums["serial_ns"] - captured + sums["split_ns"] + switches * rates.switch_ns
    return dict(tile=tile, slots=slots, bytes_required=cap, ddr_bytes=ddr, sram_bytes=sram,
                x_bytes=x_bytes, w_bytes=w_bytes, y_bytes=y_bytes,
                output_tiles=sums["output_tiles"], switches=switches, serial_ms=sums["serial_ns"]/1e6,
                possible_overlap_ms=sums["raw_saved_ns"]/1e6, captured_overlap_ms=captured/1e6,
                split_startup_ms=sums["split_ns"]/1e6, switch_ms=switches*rates.switch_ns/1e6,
                ms=total/1e6)


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
    if rates.dma_gbs is not None and not (math.isfinite(rates.dma_gbs) and rates.dma_gbs > 0):
        raise ValueError("DMA bandwidth must be finite and positive")
    if not all(math.isfinite(v) and v >= 0 for v in
               (rates.ddr_ns, rates.dma_ns, rates.sram_ns, rates.tensor_ns)):
        raise ValueError("startup latencies must be finite and nonnegative")
    if min(rates.x_bytes, rates.w_bytes, rates.y_bytes) < 1:
        raise ValueError("byte widths must be positive integers")
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
            if rates.footprint(m, n, k, slots) <= kib*1024:
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
    parser.add_argument("--dma-gbs", type=float, help="DMA rate; default follows DDR")
    for unit in ("ddr", "dma", "sram", "tensor"):
        parser.add_argument(f"--{unit}-ns", type=float, default=0, help="per logical service startup")
    for matrix in ("x", "w", "y"):
        parser.add_argument(f"--{matrix}-bytes", type=int, default=2, help="bytes per element")
    parser.add_argument("--overlap", type=float, default=1, help="fraction of feasible overlap captured")
    parser.add_argument("--switch-ns", type=float, default=0,
                        help="extra unhidden overhead per overlapped panel boundary")
    args = parser.parse_args()
    rates = Rates(**{field: getattr(args, field) for field in Rates.__dataclass_fields__})
    try:
        results = {kib: compare(args.shape, kib, args.panels, rates, args.tile_quantum)
                   for kib in args.kib}
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(dict(shape=args.shape, panels=args.panels, tile_quantum=args.tile_quantum,
                          rates=asdict(rates), results=results), indent=2))


if __name__ == "__main__":
    main()
