"""int8 quantisation of an exported Decima (encoder.onnx + scorer.onnx) for CPU serving.

    uv run python -m decima.quantize export/v1a2 --out export/v1a2-int8
    uv run python -m decima.quantize export/v1a2 --out /tmp/x --method naive      # the old recipe, for comparison

Why not plain `quantize_dynamic`: MatMulInteger quantises each activation tensor with ONE
uint8 scale per call. The e5 backbone has outlier channels (e.g. 15/35/321 after LayerNorm,
|x| up to ~70 into the FFN output projections vs a median channel of ~1), so the other
channels get a handful of levels; 2-6 % relative error per MatMul compounds over 72 of
them and top-1 over 60-150 choices flips (v1a2: 0.88 agreement with fp32 on real items,
0.91 with per-channel weights). The `nbits8` recipe uses com.microsoft MatMulNBits instead:
weights int8 in blocks of 32 along K, and with accuracy_level=4 activations are quantised
per block of 32 as well, so an outlier only spoils its own block: ~0.99 agreement.

The scorer is harmless under plain dynamic int8 (agreement 1.000, mean |dp| 2e-4), and
MatMulInteger beats MatMulNBits there (many choice tokens per call), so it keeps it.
The 250k x 384 word-embedding table (80 % of the bytes) is stored int8 with one scale per
row (Gather int8 + Gather scale + Mul), which keeps the ~4x size reduction.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GraphRecipe:
    engine: str = "dynamic"               # dynamic: quantize_dynamic (MatMulInteger) | nbits: MatMulNBits
    # dynamic
    per_channel: bool = False
    reduce_range: bool = False
    op_types: tuple[str, ...] = ("MatMul", "Gather", "Gemm")
    exclude: tuple[str, ...] = ()         # substrings of node names kept fp32
    # nbits
    bits: int = 8
    block_size: int = 32
    accuracy_level: int = 4               # 4: int8 compute, activations quantised per block; 0: fp32 compute
    # both
    rowwise_embeddings: bool = False      # word embeddings → int8 table + one fp32 scale per row


@dataclass(frozen=True)
class Recipe:
    encoder: GraphRecipe | None           # None: keep fp32
    scorer: GraphRecipe | None


RECIPES: dict[str, Recipe] = {
    # the original bench/latency.py recipe: 0.88 top-1 agreement with fp32 on real items
    "naive": Recipe(GraphRecipe(), GraphRecipe()),
    # per-channel weights: 0.91 — the activations, not the weights, are the problem
    "dynamic-pc": Recipe(GraphRecipe(per_channel=True), GraphRecipe(per_channel=True)),
    # the default: 0.99 agreement, accuracy within noise, 3.8x smaller
    "nbits8": Recipe(GraphRecipe(engine="nbits", bits=8, block_size=32, accuracy_level=4, rowwise_embeddings=True),
                     GraphRecipe(per_channel=True)),
}
DEFAULT = "nbits8"


def _node_names(path: Path, patterns: tuple[str, ...]) -> list[str]:
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    return [n.name for n in m.graph.node if any(p in n.name for p in patterns)]


def _rowwise_embeddings(src: Path, dst: Path) -> None:
    """Replace the word-embedding Gather by int8 table + per-row scale: x = f32(q8[ids]) * scale[ids].
    The table is found by shape, not name (e5: word_embeddings, ModernBERT/granite: tok_embeddings):
    the Gather over the largest 2-D initializer."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(src))
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    gathers = [n for n in g.node if n.op_type == "Gather" and n.input[0] in inits and len(inits[n.input[0]].dims) == 2]
    node = max(gathers, key=lambda n: inits[n.input[0]].dims[0] * inits[n.input[0]].dims[1])
    table = node.input[0]
    w = numpy_helper.to_array(inits[table]).astype(np.float32)
    scale = (np.maximum(np.abs(w).max(1, keepdims=True), 1e-12) / 127.0).astype(np.float32)
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    g.initializer.remove(inits[table])
    g.initializer.extend([numpy_helper.from_array(q, table + "_q8"), numpy_helper.from_array(scale, table + "_scale")])
    ids, y = node.input[1], node.output[0]
    idx = list(g.node).index(node)
    g.node.remove(node)
    new = [
        helper.make_node("Gather", [table + "_q8", ids], [y + "_q8"], name=node.name + "_q8", axis=0),
        helper.make_node("Gather", [table + "_scale", ids], [y + "_s"], name=node.name + "_scale", axis=0),
        helper.make_node("Cast", [y + "_q8"], [y + "_f"], name=node.name + "_cast", to=onnx.TensorProto.FLOAT),
        helper.make_node("Mul", [y + "_f", y + "_s"], [y], name=node.name + "_mul"),
    ]
    for k, n in enumerate(new):
        g.node.insert(idx + k, n)
    onnx.save(m, str(dst))


def _nbits(src: Path, dst: Path, r: GraphRecipe, exclude: list[str]) -> None:
    """Constant-weight 2-D MatMuls → com.microsoft MatMulNBits (symmetric blocks along K).

    Same packing as onnxruntime's MatMulNBitsQuantizer (whose module needs `onnx_ir`, not a
    dependency here), through the C++ kernel it wraps."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    from onnxruntime.capi._pybind_state import quantize_matmul_4bits, quantize_matmul_8bits

    m = onnx.load(str(src))
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    uses: dict[str, int] = {}
    for n in g.node:
        for i in n.input:
            uses[i] = uses.get(i, 0) + 1
    kpack, bs = 8 // r.bits, r.block_size
    qfn = {8: quantize_matmul_8bits, 4: quantize_matmul_4bits}[r.bits]
    nodes = []
    for n in g.node:
        w = numpy_helper.to_array(inits[n.input[1]]) if n.op_type == "MatMul" and n.input[1] in inits else None
        if w is None or w.ndim != 2 or n.name in exclude:
            nodes.append(n)
            continue
        K, N = w.shape
        kb = (K + bs - 1) // bs
        wp = np.ascontiguousarray(np.pad(w.astype(np.float32), ((0, kb * bs - K), (0, 0))))
        packed = np.zeros((N, kb, (bs + kpack - 1) // kpack), dtype=np.uint8)
        scales = np.zeros((N, kb), dtype=np.float32)
        zp = np.zeros((N, (kb + kpack - 1) // kpack), dtype=np.uint8)
        qfn(packed, wp, scales, zp, bs, N, K, True)
        b = n.input[1]
        g.initializer.extend([numpy_helper.from_array(packed, f"{b}_Q{r.bits}"), numpy_helper.from_array(scales, f"{b}_scales")])
        if uses[b] == 1:
            g.initializer.remove(inits[b])
        nodes.append(helper.make_node("MatMulNBits", [n.input[0], f"{b}_Q{r.bits}", f"{b}_scales"], list(n.output),
                                      name=f"{n.name}_Q{r.bits}", domain="com.microsoft", K=K, N=N, bits=r.bits,
                                      block_size=bs, accuracy_level=r.accuracy_level))
    del g.node[:]
    g.node.extend(nodes)
    if not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.save(m, str(dst))


def quantize_graph(src: Path, dst: Path, r: GraphRecipe | None) -> None:
    if r is None:
        shutil.copyfile(src, dst)
        return
    from onnxruntime.quantization import QuantType, quantize_dynamic

    with tempfile.TemporaryDirectory(dir=dst.parent) as tmp:
        cur = src
        exclude = _node_names(cur, r.exclude) if r.exclude else []
        if r.rowwise_embeddings:
            cur = Path(tmp) / "rowwise.onnx"
            _rowwise_embeddings(src, cur)
            exclude += _node_names(cur, ("word_embeddings/Gather_q8", "word_embeddings/Gather_scale"))
        if r.engine == "nbits":
            _nbits(cur, dst, r, exclude)
        else:
            quantize_dynamic(str(cur), str(dst), weight_type=QuantType.QInt8, per_channel=r.per_channel,
                             reduce_range=r.reduce_range, op_types_to_quantize=list(r.op_types), nodes_to_exclude=exclude)


def quantize(src_dir: str | Path, dst_dir: str | Path, method: str = DEFAULT) -> Path:
    """fp32 export dir → quantised export dir with the same layout. The tokenizer is copied, not linked:
    an absolute symlink broke the int8 export on every other machine (found on x86, docs/BENCH-x86.md),
    and symlinks do not survive a Hub upload."""
    src, dst = Path(src_dir), Path(dst_dir)
    r = RECIPES[method]
    dst.mkdir(parents=True, exist_ok=True)
    quantize_graph(src / "encoder.onnx", dst / "encoder.onnx", r.encoder)
    quantize_graph(src / "scorer.onnx", dst / "scorer.onnx", r.scorer)
    shutil.copyfile(src / "decima.json", dst / "decima.json")
    tok = dst / "tokenizer"
    if tok.is_symlink() or tok.is_file():
        tok.unlink()
    elif tok.exists():
        shutil.rmtree(tok)
    shutil.copytree(src / "tokenizer", tok)
    (dst / "quantize.txt").write_text(f"method={method}\nsrc={src.name}\n{r}\n")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="fp32 export dir (decima.export)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default=DEFAULT, choices=sorted(RECIPES))
    args = ap.parse_args()
    d = quantize(args.src, args.out, args.method)
    print(f"{args.method} → {d}: " + ", ".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in sorted(d.glob('*.onnx'))))


if __name__ == "__main__":
    main()
