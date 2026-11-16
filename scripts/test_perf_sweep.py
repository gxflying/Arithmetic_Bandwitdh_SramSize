"""运行公开的 JSON 样例和可手算的边界条件。"""
import copy
import json
import unittest
from pathlib import Path

from attention_timing import Stage, prefetch
from perf_sweep import PORTS, VOPS, scan
from pingpong_tradeoff import Rates


CONFIG = Path(__file__).resolve().parents[1]/"configs"


class SweepTests(unittest.TestCase):
    def test_attention_grid(self):
        cfg = json.loads((CONFIG/"flashattention_sram.json").read_text())
        rows = list(scan(cfg))
        self.assertEqual(len(rows), 450)
        baseline = [r for r in rows if all(r[k] == v for k, v in cfg["plot_at"].items())]
        self.assertEqual([r["bytes_required"] for r in baseline], [81920, 114688])
        self.assertAlmostEqual(baseline[0]["ms"]*1e6, 239.568)
        self.assertAlmostEqual(baseline[1]["ms"]*1e6, 214.032)
        # Force ragged Q and KV panels; K/V are read once per Q tile.
        cfg["fixed"].update(q_len=33, kv_len=129, dv=16, out_dtype="fp32")
        for r in scan({**cfg, "sweep": {}, "fixed": {**cfg["fixed"], **cfg["plot_at"]}, "plot_at": {}}):
            self.assertEqual(r["ddr_read_bytes"], 2*(33*128+2*129*(128+16)))
            self.assertEqual(r["ddr_write_bytes"], 4*33*16)

    def test_startup_crosses_stages(self):
        hw = Rates(ddr_read_gbs=3, dma_gbs=9, sram_write_gbs=5,
                   ddr_read_ns=4, dma_ns=2, sram_write_ns=2)
        stage = Stage("compute", 0, 0, 1, 4, 6)
        # Setup consumes 6 ns + 2 ns. Second stage has 4 ns at
        # min(3, 5 - 4/4) B/ns, third has the same payload window.
        moved, wait = prefetch([stage]*3, 30, hw)
        self.assertEqual(moved, [0, 12, 12])
        self.assertEqual(wait, 2)
        self.assertEqual(prefetch([stage], 30, hw), ([0], 12))

    def test_vectors(self):
        fixed = {k: 0 if k.endswith("_ns") else 2 for k in PORTS}
        for op in VOPS:
            fixed.update({f"vec_{op}_ns": 0, f"vec_{op}_gops": 1})
        fixed.update(rows=1, length=4, x_dtype="fp32", out_dtype="fp32", capacity_kib=24/1024)
        cfg = dict(kernel="online_softmax", fixed=fixed, tiles={"v": [2]})
        a, b = scan(cfg)
        self.assertEqual([round(r["ms"]*1e6) for r in (a, b)], [54, 48])
        self.assertEqual([b[f] for f in ("ddr_read_bytes","ddr_write_bytes","sram_read_bytes","sram_write_bytes")],
                         [32, 16, 48, 48])
        # SwiGLU: two tiles of 2 elements, 5 primitive operations per element;
        # each tile takes 4 ns input, 10 ns compute and 2 ns output.
        del fixed["x_dtype"]
        fixed.update(gate_dtype=2, up_dtype=2, out_dtype=2)
        cfg["kernel"] = "swiglu"
        actual = [r["ms"]*1e6 for r in scan(cfg)]
        for value, expected in zip(actual, [32, 30, 30, 28]):
            self.assertAlmostEqual(value, expected)

    def test_infeasible_grid_and_retiling(self):
        cfg = json.loads((CONFIG/"gemm_capacity.json").read_text())
        cfg["sweep"] = {"capacity_kib": [0.01, 64, 320, 384], "partial_dtype": ["fp32"]}
        rows = list(scan(cfg))
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(r["status"] == "infeasible" and "ms" not in r for r in rows[:4]))
        double = {r["capacity_kib"]:r for r in rows if r["policy"] == "double"}
        self.assertLess(double[64]["speedup"], 1)
        self.assertGreater(double[320]["speedup"], 1)
        self.assertEqual([double[k]["tile_m"] for k in (320, 384)], [128, 256])
        self.assertLess(double[384]["ddr_read_bytes"], double[320]["ddr_read_bytes"])
        cfg["tiles"] = dict(m=[256], n=[256], k=[64])
        rows = list(scan(cfg))
        self.assertEqual([r["status"] for r in rows if r["capacity_kib"] == 320],
                         ["ok", "infeasible", "infeasible", "infeasible"])

    def test_reject_errors(self):
        base = json.loads((CONFIG/"flashattention_sram.json").read_text())
        for section, key, value in (("sweep","sram_read_gbs",[]), ("fixed","sram_read_ns",1),
                                    ("sweep","sram_write_ns",[-1]), ("fixed","partial_dtype","int4"),
                                    ("tiles","c",[0]), ("plot_at","sram_read_gbs",99)):
            cfg = copy.deepcopy(base)
            cfg[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                list(scan(cfg))
        del base["fixed"]["dma_ns"]
        with self.assertRaisesRegex(ValueError, "dma_ns"):
            list(scan(base))


if __name__ == "__main__":
    unittest.main()
