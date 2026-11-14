"""Verify directional port timing against a direct, uncompressed panel schedule."""
import unittest
from itertools import product
from dataclasses import replace
import math
import random

from pingpong_tradeoff import Rates, compare, evaluate
from flashattention_example import example


def replay(shape, tile, slots, cfg):
    rows, inner, cols = shape
    m, n, k = tile
    sx, sw = slots
    elapsed = 0.0
    engine_r = engine_w = dma_r = dma_w = 0

    def transfer(size, port, startup, *, reading_ddr=True):
        if not size:
            return 0
        ddr_rate = cfg.ddr_read_gbs if reading_ddr else cfg.ddr_write_gbs
        latency = cfg.ddr_read_ns if reading_ddr else cfg.ddr_write_ns
        link = min(ddr_rate, cfg.dma_gbs or ddr_rate, port)
        return latency + cfg.dma_ns + startup + size / link

    for i in range(0, rows, m):
        for j in range(0, cols, n):
            r = min(m, rows-i)
            c = min(n, cols-j)
            y = cfg.y_bytes * r * c
            previous = None
            for offset in range(0, inner, k):
                v = min(inner-offset, k)
                x, w = cfg.x_bytes*r*v, cfg.w_bytes*c*v
                rd = x+w + (4*r*c if offset else 0)
                wr = y if offset+v == inner else 4*r*c
                engine_r += rd
                engine_w += wr
                dma_w += x+w
                pre = ((sx-1)*x + (sw-1)*w) if previous else 0
                if previous:
                    core, prev_wr = previous
                    early = transfer(pre, cfg.sram_write_gbs, cfg.sram_write_ns)
                    joint = max(core, early, cfg.sram_write_ns + (prev_wr+pre)/cfg.sram_write_gbs)
                    elapsed += joint * cfg.overlap + (core+early) * (1-cfg.overlap)
                    if slots != (1, 1):
                        elapsed += cfg.switch_ns
                elapsed += transfer(x+w-pre, cfg.sram_write_gbs, cfg.sram_write_ns)
                previous = (max(cfg.tensor_ns + 2*r*c*v/(1000*cfg.tensor_tflops),
                                cfg.sram_read_ns + rd/cfg.sram_read_gbs,
                                cfg.sram_write_ns + wr/cfg.sram_write_gbs), wr)
            elapsed += previous[0] + transfer(y, cfg.sram_read_gbs, cfg.sram_read_ns, reading_ddr=False)
            dma_r += y
    return elapsed, engine_r, engine_w, dma_r, dma_w


class PortTiming(unittest.TestCase):
    def test_flashattention_traffic_and_time(self):
        cfg = Rates(ddr_write_gbs=256)
        model = example(cfg)
        # Count logical elements in each buffer independently of the stage table.
        q, k, v = 32*128*2, 64*128*2, 64*128*2
        score, prob, num = 32*64*4, 32*64*2, 32*128*4
        inputs = q + 2*(k+v)
        reads = 2*(q+k+v+score+prob) + 3*num + q
        writes = inputs + 2*(score+prob+num) + num + q
        self.assertEqual(model["totals"], dict(ddr_read=inputs, ddr_write=q,
                                               sram_read=reads, sram_write=writes))
        self.assertEqual(model["sram_capacity_bytes"], 112*1024)
        self.assertAlmostEqual(model["single_ns"], 239.568)
        self.assertAlmostEqual(model["double_ns"], 214.032)
        self.assertAlmostEqual(sum(model["prefetch_bytes_per_stage"])+512*model["remaining_prefetch_ns"],
                               k+v)
        for field, rate, ns in [("ddr_read_gbs", 1024, 143.152), ("ddr_write_gbs", 512, 198.032),
                                ("sram_read_gbs", 8000, 209.168), ("sram_write_gbs", 2000, 225.296)]:
            self.assertAlmostEqual(example(replace(cfg, **{field: rate}))["double_ns"], ns)

    def test_flashattention_recurrence(self):
        # FP64 semantic reference; byte width and conversion costs are modeled separately.
        rng = random.Random(37)
        d = 128
        def matrix(rows):
            return [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(rows)]
        queries, keys, values = matrix(32), matrix(128), matrix(128)
        higher_max = 0
        for q in queries:
            scores = [sum(a*b for a,b in zip(q,k))/math.sqrt(d) for k in keys]
            weights = [math.exp(s-max(scores)) for s in scores]
            den = sum(weights)
            reference = [sum(w*v[j] for w,v in zip(weights,values))/den for j in range(d)]
            largest, denom, acc = -math.inf, 0.0, [0.0]*d
            for offset in (0,64):
                local = scores[offset:offset+64]
                new_max = max(largest, max(local))
                higher_max += int(offset > 0 and new_max > largest)
                alpha = math.exp(largest-new_max)
                p = [math.exp(s-new_max) for s in local]
                denom = alpha*denom + sum(p)
                acc = [alpha*acc[j]+sum(w*v[j] for w,v in zip(p,values[offset:offset+64]))
                       for j in range(d)]
                largest = new_max
            self.assertLess(max(abs(u/denom-a) for u,a in zip(acc,reference)), 1e-12)
        self.assertGreater(higher_max, 0)

    def test_directional_replay(self):
        configs = [
            Rates(),
            Rates(sram_read_gbs=2, sram_write_gbs=91, dma_gbs=40, y_bytes=8),
            Rates(sram_read_gbs=71, sram_write_gbs=3, sram_read_ns=4, sram_write_ns=11,
                  dma_ns=2, ddr_read_ns=3, ddr_write_ns=8, tensor_ns=5, x_bytes=1, w_bytes=4, y_bytes=2),
            Rates(sram_read_gbs=5, sram_write_gbs=33, overlap=0.5, switch_ns=1.5),
            Rates(ddr_read_gbs=2, ddr_write_gbs=41, ddr_write_ns=7),
            Rates(ddr_read_gbs=43, ddr_write_gbs=2, ddr_read_ns=7),
        ]
        for inner, slots, cfg in product(range(1, 38), product((1, 2), repeat=2), configs):
            shape, tile = (13, inner, 19), (8, 16, min(inner, 4))
            with self.subTest(inner=inner, slots=slots, cfg=cfg):
                t, er, ew, dr, dw = replay(shape, tile, slots, cfg)
                actual = evaluate(shape, tile, slots, cfg)
                self.assertAlmostEqual(t, actual["ms"]*1e6, places=7)
                for key, value in dict(engine_read_bytes=er, engine_write_bytes=ew,
                                       dma_sram_read_bytes=dr, dma_sram_write_bytes=dw,
                                       sram_read_bytes=er+dr, sram_write_bytes=ew+dw,
                                       ddr_bytes=dr+dw, ddr_read_bytes=dw, ddr_write_bytes=dr).items():
                    self.assertEqual(actual[key], value)

    def test_documented_base_and_crossing(self):
        shape = (2048, 3584, 37888)
        base = compare(shape, 320, [64], Rates())
        self.assertAlmostEqual(base["single"]["ms"], 14.202241, places=6)
        self.assertAlmostEqual(base["double"]["ms"], 13.140165, places=6)
        for cfg in (Rates(ddr_read_gbs=409.5064), Rates(dma_ns=15.7373)):
            result = compare(shape, 320, [64], cfg)
            self.assertLess(abs(result["single"]["ms"]-result["double"]["ms"]), 0.00001)


if __name__ == "__main__":
    unittest.main()
