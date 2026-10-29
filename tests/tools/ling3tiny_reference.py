"""Generate a small hybrid checkpoint and an independent NumPy logits oracle.

The fixture retains Ling's real KDA/MLA head geometry, with H64 and 8 experts.
The reference uses expanded MLA and key-major recurrent state. No runtime code
or native kernel is called. Requires NumPy; outputs stay outside tracked fixtures.
"""
import argparse
import json
import math
from pathlib import Path
import struct
from collections import OrderedDict

import numpy as np


def save_tensors(path, tensors):
    types = {"float32": "F32", "int32": "I32", "int64": "I64"}
    header, blocks, offset = {}, [], 0
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        data = array.tobytes()
        header[name] = {"dtype": types[str(array.dtype)], "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(data)]}
        blocks.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for data in blocks:
            f.write(data)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def norm(x, weight):
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)) * weight


def rope(x):
    x = x.copy()
    freq = 6_000_000. ** (-np.arange(32, dtype=np.float64) / 32)
    for t in range(len(x)):
        a, b = x[t, ..., 0::2].copy(), x[t, ..., 1::2].copy()
        x[t, ..., 0::2] = a * np.cos(t * freq) - b * np.sin(t * freq)
        x[t, ..., 1::2] = b * np.cos(t * freq) + a * np.sin(t * freq)
    return x


class Reference:
    def __init__(self, seed, quantize_projections=False):
        self.rng = np.random.default_rng(seed)
        self.tensors, self.quant = {}, {}
        self.quantize_projections = quantize_projections
        self.config = {"num_hidden_layers":4,"layer_group_size":4,"n_group":4,
                       "num_experts":8,"num_experts_per_tok":2,"topk_group":2}

    def matrix(self, name, n, k, quant=None):
        if quant is None:
            quant = self.quantize_projections
        if quant:
            codes = self.rng.integers(-7, 8, (n, k), dtype=np.int32)
            scales = np.full((n, 1), 0.025 / math.sqrt(k), dtype=np.float32)
            packed = np.zeros((n, k // 8), dtype=np.uint32)
            for shift in range(8):
                packed |= (codes[:, shift::8] + 8).astype(np.uint32) << (shift * 4)
            self.tensors[name + ".weight_packed"] = packed.view(np.int32)
            self.tensors[name + ".weight_shape"] = np.array([n, k], np.int64)
            self.tensors[name + ".weight_scale"] = scales
            self.quant[name] = (codes, scales)
        else:
            self.tensors[name + ".weight"] = (self.rng.standard_normal((n, k)) *
                                               (0.15 / math.sqrt(k))).astype(np.float32)

    def vector(self, name, n, value=1.):
        self.tensors[name] = np.full(n, value, dtype=np.float32)

    def linear(self, name, x):
        if name not in self.quant:
            return x @ self.tensors[name + ".weight"].T
        codes, scales = self.quant[name]
        scale = np.max(np.abs(x), axis=-1, keepdims=True) / np.float32(127.)
        scale[scale == 0] = 1
        value = x / scale
        a = (np.sign(value) * np.floor(np.abs(value) + 0.5)).clip(-127, 127).astype(np.int32)
        # Every product and partial sum is an integer below 2**53. FP64 BLAS
        # therefore gives the exact INT32 dot product, independent of sum order.
        dot = a.astype(np.float64) @ codes.astype(np.float64).T
        return (dot.astype(np.float32) * scale) * scales.T

    def ffn(self, p, x):
        gate, up = self.linear(p + ".gate_proj", x), self.linear(p + ".up_proj", x)
        return self.linear(p + ".down_proj", gate * sigmoid(gate) * up)

    def evaluate(self, tokens):
        x = self.tensors["model.word_embeddings.weight"][tokens].copy()
        rows = len(tokens)
        c = self.config
        for layer in range(c["num_hidden_layers"]):
            p = f"model.layers.{layer}"
            a = p + ".attention"
            inp = norm(x, self.tensors[p + ".input_layernorm.weight"])
            self.trace(layer, "input", inp)
            if (layer + 1) % c["layer_group_size"] != 0:
                qkv = []
                for name in ("q", "k", "v"):
                    projection = self.linear(a + f".{name}_proj", inp)
                    padded = np.pad(projection, ((3, 0), (0, 0)))
                    w = self.tensors[a + f".{name}_conv1d.weight"][:, 0]
                    conv = sum(padded[j:j + rows] * w[:, j] for j in range(4))
                    qkv.append((conv * sigmoid(conv)).reshape(rows, 16, 128))
                q, k, v = qkv
                q = q / np.sqrt((q*q).sum(-1, keepdims=True) + 1e-6)
                k = k / np.sqrt((k*k).sum(-1, keepdims=True) + 1e-6)
                forget = self.linear(a + ".f_proj", inp) + self.tensors[a + ".dt_bias"]
                decay = np.exp(-5 * sigmoid(forget.reshape(rows, 16, 128) *
                                            np.exp(self.tensors[a + ".A_log"])[None, :, None]))
                beta = sigmoid(self.linear(a + ".b_proj", inp))
                state = np.zeros((16, 128, 128), dtype=np.float64)  # key, value
                output = []
                for t in range(rows):
                    state *= decay[t, :, :, None]
                    delta = (v[t] - np.einsum("hk,hkv->hv", k[t], state)) * beta[t, :, None]
                    state += k[t, :, :, None] * delta[:, None, :]
                    output.append(np.einsum("hk,hkv->hv", q[t], state) / math.sqrt(128))
                out = np.array(output, dtype=np.float32)
                out = norm(out, self.tensors[a + ".o_norm.weight"]) * sigmoid(
                    self.linear(a + ".g_proj", inp).reshape(rows, 16, 128))
                mixed = self.linear(a + ".o_proj", out.reshape(rows, 2048))
            else:
                qa = norm(self.linear(a + ".q_a_proj", inp), self.tensors[a + ".q_a_layernorm.weight"])
                q = self.linear(a + ".q_b_proj", qa).reshape(rows, 16, 192)
                q[:, :, 128:] = rope(q[:, :, 128:])
                kv = self.linear(a + ".kv_a_proj_with_mqa", inp)
                latent = norm(kv[:, :512], self.tensors[a + ".kv_a_layernorm.weight"])
                kv_name = a + ".kv_b_proj"
                if kv_name in self.quant:
                    # Absorption uses the reconstructed static kv_b matrix.
                    codes, scales = self.quant[kv_name]
                    expanded = latent @ (codes.astype(np.float32) * scales).T
                else:
                    expanded = self.linear(kv_name, latent)
                expanded = expanded.reshape(rows, 16, 256)
                keys = np.concatenate([expanded[:, :, :128], np.broadcast_to(rope(kv[:, 512:])[:, None], (rows, 16, 64))], -1)
                output = []
                for t in range(rows):
                    scores = np.einsum("hd,thd->ht", q[t], keys[:t+1]) / math.sqrt(192)
                    probs = np.exp(scores - scores.max(-1, keepdims=True))
                    probs /= probs.sum(-1, keepdims=True)
                    output.append(np.einsum("ht,thd->hd", probs, expanded[:t+1, :, 128:]))
                out = np.array(output) * sigmoid(self.linear(a + ".g_proj", inp))[:, :, None]
                mixed = self.linear(a + ".dense", out.reshape(rows, 2048))
            self.trace(layer, "o", mixed)
            x += mixed
            inp = norm(x, self.tensors[p + ".post_attention_layernorm.weight"])
            self.trace(layer, "post", inp)
            if layer == 0:
                ffn = self.ffn(p + ".mlp", inp)
            else:
                r = p + ".mlp"
                scores = sigmoid(self.linear(r + ".gate", inp))
                biased = scores + self.tensors[r + ".gate.expert_bias"]
                groups, experts = c["n_group"], c["num_experts"]
                group_width = experts // groups
                group_scores = np.sort(biased.reshape(rows, groups, group_width), axis=-1)[:, :, -2:].sum(-1)
                selected_groups = np.argsort(-group_scores, axis=-1)[:, :c["topk_group"]]
                expert_out = np.zeros_like(inp)
                for t in range(rows):
                    candidates = [i for i in range(experts) if i // group_width in selected_groups[t]]
                    ids = sorted(candidates, key=lambda i: -biased[t, i])[:c["num_experts_per_tok"]]
                    probs = scores[t, ids] / scores[t, ids].sum() * 2.5
                    for expert, weight in zip(ids, probs):
                        expert_out[t] += self.ffn(r + f".experts.{expert}", inp[t:t+1])[0] * weight
                ffn = expert_out + self.ffn(r + ".shared_experts", inp)
            self.trace(layer, "ffn", ffn)
            x += ffn
            self.trace(layer, "hidden", x)
        return self.linear("lm_head", norm(x, self.tensors["model.norm.weight"]))

    def trace(self, layer, stage, values):
        directory = getattr(self, "trace_directory", None)
        if directory:
            values.astype("<f4").tofile(directory / f"l{layer}-{stage}.f32")


class CheckpointTensors:
    """Read only requested tensor views; no dependency on torch or safetensors."""
    def __init__(self, directory):
        self.directory = directory
        self.index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}
        for filename in set(self.index.values()):
            with (directory / filename).open("rb") as f:
                size, = struct.unpack("<Q", f.read(8))
                self.headers[filename] = (size + 8, json.loads(f.read(size)))

    def __contains__(self, name):
        return name in self.index

    def __getitem__(self, name):
        filename = self.index[name]
        base, header = self.headers[filename]
        meta = header[name]
        dtype = {"F32":"<f4", "F16":"<f2", "BF16":"<u2", "I8":"i1", "I32":"<i4", "I64":"<i8"}[meta["dtype"]]
        values = np.memmap(self.directory / filename, mode="r", dtype=dtype,
                           offset=base+meta["data_offsets"][0], shape=tuple(meta["shape"]))
        if meta["dtype"] == "BF16":
            return (values.astype(np.uint32) << 16).view(np.float32)
        return values.astype(np.float32) if meta["dtype"] == "F16" else values


class QuantizedMatrices:
    def __init__(self, tensors):
        self.tensors, self.cache = tensors, OrderedDict()

    def __contains__(self, name):
        return name + ".weight_scale" in self.tensors

    def __getitem__(self, name):
        if name not in self.cache:
            if name + ".weight_packed" in self.tensors:
                n, k = self.tensors[name + ".weight_shape"]
                packed = self.tensors[name + ".weight_packed"].view(np.uint32)
                codes = np.empty((n, k), dtype=np.int8)
                for shift in range(8):
                    codes[:, shift::8] = ((packed[:, :codes[:, shift::8].shape[1]] >> (4*shift)) & 15).astype(np.int8)-8
            else:
                codes = self.tensors[name + ".weight"]
            self.cache[name] = (codes, self.tensors[name + ".weight_scale"])
        self.cache.move_to_end(name)
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return self.cache[name]


def checkpoint_reference(directory, output, tokens, trace_directory=None):
    ref = Reference(0)
    ref.config = json.loads((directory / "config.json").read_text())
    ref.tensors = CheckpointTensors(directory)
    ref.quant = QuantizedMatrices(ref.tensors)
    if trace_directory:
        trace_directory.mkdir(parents=True, exist_ok=True)
    ref.trace_directory = trace_directory
    # Complete causal prefix once, all row logits. Retain last prompt row and
    # each teacher-forced continuation row in the same format as the checker.
    logits = ref.evaluate(tokens)
    logits.astype("<f4").tofile(output)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantize-projections", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokens", default="17,33,6,45,128,57,91,3,88,42,123,89")
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if args.checkpoint:
        tokens = [int(t) for t in args.tokens.split(",")]
        checkpoint_reference(args.checkpoint, args.output, tokens, args.trace_dir)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(__file__).parents[1] / "fixtures/ling3tiny/config.json").read_text())
    config.update(hidden_size=64, vocab_size=256, num_hidden_layers=4, num_experts=8,
                  num_experts_per_tok=2, n_group=4, topk_group=2, intermediate_size=128,
                  moe_intermediate_size=64, moe_shared_expert_intermediate_size=64,
                  eos_token_id=255, max_position_embeddings=256)
    config["quantization_config"] = {"quant_method": "compressed-tensors", "quantization_status": "compressed",
        "format": "pack-quantized", "config_groups": {"w4": {"targets": ["Linear"],
        "weights": {"num_bits": 4, "type": "int", "strategy": "channel", "symmetric": True},
        "input_activations": {"num_bits": 8, "type": "int", "strategy": "token", "symmetric": True, "dynamic": True}}}}
    ref = Reference(3701, args.quantize_projections)
    ref.tensors["model.word_embeddings.weight"] = ref.rng.standard_normal((256, 64)).astype(np.float32) * .1
    ref.vector("model.norm.weight", 64)
    ref.matrix("lm_head", 256, 64)
    for layer in range(4):
        p = f"model.layers.{layer}"
        ref.vector(p + ".input_layernorm.weight", 64)
        ref.vector(p + ".post_attention_layernorm.weight", 64)
        a = p + ".attention"
        if layer < 3:
            for name in ("q", "k", "v"):
                ref.matrix(a + f".{name}_proj", 2048, 64)
                ref.tensors[a + f".{name}_conv1d.weight"] = ref.rng.uniform(.1, .4, (2048, 1, 4)).astype(np.float32)
            for name, n in [("f_proj", 2048), ("b_proj", 16), ("g_proj", 2048), ("o_proj", 64)]:
                ref.matrix(a + "." + name, n, 2048 if name == "o_proj" else 64)
            ref.vector(a + ".A_log", 16, .4)
            ref.vector(a + ".dt_bias", 2048, .1)
            ref.vector(a + ".o_norm.weight", 128)
        else:
            for name, n, k in [("q_a_proj", 256, 64), ("q_b_proj", 3072, 256),
                               ("kv_a_proj_with_mqa", 576, 64), ("kv_b_proj", 4096, 512),
                               ("g_proj", 16, 64), ("dense", 64, 2048)]:
                ref.matrix(a + "." + name, n, k)
            ref.vector(a + ".q_a_layernorm.weight", 256)
            ref.vector(a + ".kv_a_layernorm.weight", 512)
        r = p + ".mlp"
        modules = [(r, 128, False)] if layer == 0 else [(r + ".shared_experts", 64, False)] + [
            (r + f".experts.{e}", 64, True) for e in range(8)]
        for name, f, quant in modules:
            ref.matrix(name + ".gate_proj", f, 64, quant)
            ref.matrix(name + ".up_proj", f, 64, quant)
            ref.matrix(name + ".down_proj", 64, f, quant)
        if layer > 0:
            ref.matrix(r + ".gate", 8, 64, quant=False)
            ref.tensors[r + ".gate.expert_bias"] = np.arange(8, dtype=np.float32) * .15
    save_tensors(args.output / "model.safetensors", ref.tensors)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    prompt = [17, 33, 6, 45, 128, 57, 91, 3, 88]
    continuation = [42, 123, 89]
    tokens = prompt + continuation
    # Each step is evaluated against the complete causal prefix.
    logits = np.concatenate([ref.evaluate(tokens[:i])[-1:] for i in range(len(prompt), len(tokens)+1)])
    logits.astype("<f4").tofile(args.output / "reference.f32")
    (args.output / "reference.json").write_text(json.dumps({"prompt": prompt, "continuation": continuation,
        "dtype": "float32", "rows": len(continuation)+1, "vocab": 256,
        "quantized_projections": args.quantize_projections,
        "oracle": "NumPy expanded MLA, key-major FP64 KDA state, W4/dynamic-A8 routed experts"}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
