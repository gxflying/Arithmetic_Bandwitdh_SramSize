"""Vector workload timings for the mappings in sections 2.3 and 3.2."""
from attention_timing import vector_time


def vector_result(kernel, width, policy, hw, p):
    softmax = kernel == "online_softmax"
    rows, length = p["rows"], p["length"]
    if softmax:
        first, second = p["x_bytes"], 0
    else:
        first, second = p["gate_bytes"], p["up_bytes"]
    slots = {"single": (1, 1), "X_only": (2, 1), "W_only": (1, 2), "double": (2, 2)}
    sx, sw = slots[policy]
    src, dst = first+second, p["out_bytes"]
    early = (sx-1)*first + (sw-1)*second
    panels = (length+width-1)//width
    serial = gain = split = 0.0
    for pass_no in ([1, 2] if softmax else [1]):
        if softmax and pass_no == 2:
            serial += vector_time([("inv", 1)], p)
        # Only the first panel, the transition to the tail, and the tail
        # can differ from a full interior panel. Rows are independent.
        cuts = sorted({0, min(1, panels), max(0, panels-2), panels-1, panels})
        for j, end in zip(cuts, cuts[1:]):
            v = min(width, length-j*width)
            z = v*src
            y = 0 if softmax and pass_no == 1 else dst*v
            if not softmax:
                ops = [("exp", v), ("add", v), ("inv", v), ("mul", v), ("mul", v)]
            elif pass_no == 1:
                ops = [("reduce", v-1), ("add", v), ("exp", v), ("reduce", v-1)]
                if j:
                    ops += [("reduce", 1), ("add", 1), ("exp", 1), ("mul", 1), ("add", 1)]
            else:
                ops = [("add", v), ("exp", v), ("mul", v)]
            core = max(vector_time(ops, p), hw.sram_read_ns + z/hw.sram_read_gbs,
                       hw.sram_write_ns + y/hw.sram_write_gbs if y else 0)
            serial += (end-j)*(hw.transfer_ns(z, to_sram=True) + core
                               + hw.transfer_ns(y, to_sram=False))
            if j+1 < panels and early:
                pre = min(width, length-(j+1)*width)*early
                load = hw.transfer_ns(pre, to_sram=True)
                joint = max(core, load, hw.sram_write_ns + (y+pre)/hw.sram_write_gbs)
                gain += (end-j)*(core+load-joint)
                if early < src:
                    split += (end-j)*(hw.ddr_read_ns + hw.dma_ns + hw.sram_write_ns)
    reads = (2 if softmax else 1)*rows*length*src
    writes = rows*length*dst
    return dict(tile=(width,), bytes_required=width*(sx*first+sw*second+dst),
                ms=rows*(serial-gain+split)/1e6, ddr_read_bytes=reads, ddr_write_bytes=writes,
                sram_read_bytes=reads+writes, sram_write_bytes=reads+writes)
