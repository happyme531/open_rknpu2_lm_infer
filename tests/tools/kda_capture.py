"""GDB input capture for the first Ling KDA layer; source from a gdb session."""
import json
from pathlib import Path
import struct
import gdb


class Capture(gdb.Breakpoint):
    def __init__(self):
        super().__init__("rknpu2_kda_run", internal=True)
        self.state_address = None
        self.saved = set()
        self.root = Path("results/ling-kda-20261023/inputs")

    def stop(self):
        inferior = gdb.selected_inferior()
        address = int(gdb.parse_and_eval("$x0"))
        rows, heads, dim, threads, state, q, k, v, decay, beta, output = struct.unpack(
            "<11Q", bytes(inferior.read_memory(address, 88)))
        if self.state_address is None:
            self.state_address = state
        if state != self.state_address or rows not in (1, 128):
            return False
        state_bytes = bytes(inferior.read_memory(state, heads * dim * dim * 4))
        nonzero = any(x != 0.0 for (x,) in struct.iter_unpack("<f", state_bytes))
        label = f"m{rows}_" + ("nonzero" if nonzero else "zero")
        if label in self.saved or (rows == 1 and not nonzero):
            return False
        dest = self.root / label
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "state.f32").write_bytes(state_bytes)
        for name, pointer, count in [("q", q, rows * heads * dim),
                                     ("k", k, rows * heads * dim),
                                     ("v", v, rows * heads * dim),
                                     ("decay", decay, rows * heads * dim),
                                     ("beta", beta, rows * heads)]:
            (dest / (name + ".f32")).write_bytes(bytes(inferior.read_memory(pointer, count * 4)))
        (dest / "meta.json").write_text(json.dumps({"rows": rows, "heads": heads,
            "dimension": dim, "threads": threads, "layer": 0, "nonzero_initial_state": nonzero,
            "source_binary": "/tmp/ling-review-20261023/optimized"}, indent=2) + "\n")
        self.saved.add(label)
        print("KDA capture:", label, flush=True)
        if len(self.saved) == 3:
            self.enabled = False
        return False


Capture()
