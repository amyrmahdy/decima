"""Parameter counts for Decima-small and every competitor, measured the same way (docs/EVAL.md §6).

    uv run python scripts/audit/params.py            # → runs/audit/params.json + a table

Split, for every system:
- vocab_table = the token-embedding matrix;
- body        = the backbone minus the token embeddings (layers, position embeddings, norms);
- head        = everything outside the backbone.
Decima-small is counted from its checkpoint (`DecimaModel.param_counts()`). Competitors are counted from
tensor shapes in the safetensors headers of the pinned Hub revisions below — no weights are loaded
(`huggingface_hub.get_safetensors_metadata`, which reads only the header; if the Hub is unreachable the
header of the locally cached file is parsed instead). Kev's scoring head lives in `head.pt` (a small torch
pickle) and is loaded on CPU. Kev's LoRA adapter is reported separately: it is merged into the base weights
at load time, so it adds no parameters to the model as run. Kev-0.8B's base checkpoint also carries a
vision tower and an MTP (multi-token prediction) block that Kev never runs; they are reported as `unused`
and excluded from the total.
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

DECIMA = "checkpoints/v1i/best"
OUT = Path("runs/audit/params.json")
REV = {  # pinned revisions (the ones every competitor was evaluated at)
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    "Qwen/Qwen3.5-0.8B-Base": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    "jaredpalmer/kev-0.5b": "9ce2fd39db3a397c89733f94af948e3d1fdfffcd",
    "jaredpalmer/kev-0.8b": "54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8",
    "convaiinnovations/laya": "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8",
    "convaiinnovations/laya-multilingual": "82d57fc4f2d1be3d2caac494045f2ec51d0842f3",
}


def shapes(repo: str) -> dict[str, int]:
    """{tensor name: element count} for every safetensors file of `repo` at its pinned revision."""
    rev = REV[repo]
    try:
        from huggingface_hub import get_safetensors_metadata

        meta = get_safetensors_metadata(repo, revision=rev)
        return {k: math.prod(t.shape) for f in meta.files_metadata.values() for k, t in f.tensors.items()}
    except Exception as e:  # offline: parse the headers of the cached files
        from huggingface_hub import scan_cache_dir

        files = [f.file_path for r in scan_cache_dir().repos if r.repo_id == repo
                 for rv in r.revisions if rv.commit_hash == rev for f in rv.files
                 if f.file_name.endswith(".safetensors")]
        if not files:
            raise SystemExit(f"{repo}@{rev[:7]}: Hub unreachable ({e}) and nothing cached") from e
        out = {}
        for p in files:
            with open(p, "rb") as fh:
                h = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
            out.update({k: math.prod(v["shape"]) for k, v in h.items() if k != "__metadata__"})
        return out


def kev_head(repo: str) -> int:
    import torch
    from huggingface_hub import hf_hub_download

    d = torch.load(hf_hub_download(repo, "head.pt", revision=REV[repo]), map_location="cpu", weights_only=False)
    return sum(v.numel() for v in d["head"].values())


def split(t: dict[str, int], vocab: str, body: str, head: bool = False, unused: tuple[str, ...] = ()) -> dict:
    """Split tensors by name prefix: `vocab` is the embedding tensor, `body` the backbone prefix,
    `head=True` counts the complement of the backbone as the head, `unused` prefixes excluded from the total."""
    un = sum(n for k, n in t.items() if k.startswith(unused)) if unused else 0
    v = t[vocab]
    b = sum(n for k, n in t.items() if k.startswith(body)) - v
    h = sum(n for k, n in t.items() if not k.startswith(body) and not (unused and k.startswith(unused))) if head else 0
    return {"vocab_table": v, "body": b, "head": h, "unused": un}


def kev(repo: str, base: str, vocab: str, body: str, unused: tuple[str, ...] = ()) -> dict:
    c = split(shapes(base), vocab, body, unused=unused)
    c["head"] = kev_head(repo)
    c["lora_merged"] = sum(shapes(repo).values())   # adapter_model.safetensors
    c["source"] = f"{base}@{REV[base][:7]} (tied embeddings) + {repo}@{REV[repo][:7]} head.pt"
    return c


def decima() -> dict:
    import torch

    from decima.model import DecimaModel

    torch.set_num_threads(min(2, os.cpu_count() or 1))
    c = DecimaModel.load(DECIMA, "cpu").param_counts()
    return {"vocab_table": c["vocab_table"], "body": c["transformer_body"], "head": c["head"], "unused": 0,
            "source": f"{DECIMA} param_counts()"}


def main() -> None:
    rows = {
        "decima-small": decima(),
        "kev-0.5b": kev("jaredpalmer/kev-0.5b", "Qwen/Qwen2.5-0.5B", "model.embed_tokens.weight", "model."),
        "kev-0.8b": kev("jaredpalmer/kev-0.8b", "Qwen/Qwen3.5-0.8B-Base",
                        "model.language_model.embed_tokens.weight", "model.language_model.",
                        unused=("model.visual.", "mtp.")),
    }
    for name in ("laya", "laya-multilingual"):
        repo = f"convaiinnovations/{name}"
        rows[name] = {**split(shapes(repo), "encoder.embeddings.tok_embeddings.weight", "encoder.", head=True),
                      "source": f"{repo}@{REV[repo][:7]} model.safetensors"}
    for r in rows.values():
        r["total"] = r["vocab_table"] + r["body"] + r["head"]
        r["non_embedding"] = r["body"] + r["head"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"revisions": REV, "systems": rows}, indent=2) + "\n")
    print(f"{'system':<18}{'total':>14}{'vocab table':>14}{'body':>14}{'head':>12}{'non-emb':>14}")
    for k, r in rows.items():
        print(f"{k:<18}{r['total']:>14,}{r['vocab_table']:>14,}{r['body']:>14,}{r['head']:>12,}{r['non_embedding']:>14,}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
