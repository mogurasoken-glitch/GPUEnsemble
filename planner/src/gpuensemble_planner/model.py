"""Turn a GGUF header into the per-layer numbers the planner needs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .gguf import GGUFFile

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.")
_EMBED_RE = re.compile(r"(^|_)embd(\.|_)")


@dataclass
class LayerProfile:
    """Bytes one transformer block needs.

    dense_bytes:    weights read for every token (attention, norms, routers,
                    shared experts, or the whole FFN of a dense model)
    expert_bytes:   routed MoE expert weights (0 for dense models); only
                    ``n_expert_used / n_expert`` of them are read per token
    kv_bytes_per_token: KV cache bytes this layer adds per context token
    state_bytes:    bytes of the state a recurrent layer (e.g. Gated DeltaNet in a hybrid
                    model) keeps for each sequence, whatever the context length
    kv_window:      for a layer that looks back only at the last tokens (sliding window
                    attention, e.g. most layers of Gemma 4), how many; 0 for a layer that
                    looks back at the whole context
    """

    dense_bytes: int
    expert_bytes: int = 0
    kv_bytes_per_token: int = 0
    state_bytes: int = 0
    kv_window: int = 0


@dataclass
class ModelProfile:
    name: str
    architecture: str
    layers: list[LayerProfile]
    head_bytes: int  # output head + final norm, placed on the last stage
    embed_bytes: int  # token embedding table, kept in host RAM (row lookup only)
    n_expert: int = 0
    n_expert_used: int = 0
    n_vocab: int = 0  # tokens in the vocabulary: the head gives one number for each (0: not known)
    notes: list[str] = field(default_factory=list)

    @property
    def logits_bytes(self) -> int:
        """Bytes the output head produces for every token: a 32-bit float for each vocabulary entry."""
        return self.n_vocab * 4

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 0 and any(layer.expert_bytes for layer in self.layers)

    @property
    def active_expert_ratio(self) -> float:
        if not self.is_moe or self.n_expert_used <= 0:
            return 1.0
        return min(1.0, self.n_expert_used / self.n_expert)

    @property
    def total_weight_bytes(self) -> int:
        return (
            sum(layer.dense_bytes + layer.expert_bytes for layer in self.layers)
            + self.head_bytes
            + self.embed_bytes
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfile":
        layers = [LayerProfile(**layer) for layer in data["layers"]]
        fields = {k: v for k, v in data.items() if k != "layers" and not k.startswith("_")}
        return cls(layers=layers, **fields)

    @classmethod
    def from_json_file(cls, path: str) -> "ModelProfile":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "layers" not in data and "uniform" in data:
            return cls.from_uniform(data)
        return cls.from_dict(data)

    @classmethod
    def from_uniform(cls, data: dict[str, Any]) -> "ModelProfile":
        """Build a profile from a short description where every layer is alike.

        Used for estimates when the model file is not downloaded yet.
        """
        u = data["uniform"]
        gib = 1024**3
        layer = LayerProfile(
            dense_bytes=int(u["dense_gib_per_layer"] * gib),
            expert_bytes=int(u.get("expert_gib_per_layer", 0) * gib),
            kv_bytes_per_token=int(u["kv_bytes_per_token_per_layer"]),
            state_bytes=int(u.get("state_bytes_per_layer", 0)),
        )
        return cls(
            name=data["name"],
            architecture=data.get("architecture", "unknown"),
            layers=[LayerProfile(**asdict(layer)) for _ in range(int(u["n_layers"]))],
            head_bytes=int(data["head_gib"] * gib),
            embed_bytes=int(data["embed_gib"] * gib),
            n_expert=int(data.get("n_expert", 0)),
            n_expert_used=int(data.get("n_expert_used", 0)),
            n_vocab=int(data.get("n_vocab", 0)),
            notes=list(data.get("notes", [])),
        )


def _per_layer(value: Any, n_layers: int, default: int) -> list[int]:
    if value is None:
        return [default] * n_layers
    if isinstance(value, list):
        if len(value) != n_layers:
            raise ValueError(f"per-layer metadata has {len(value)} entries, expected {n_layers}")
        return [int(v) for v in value]
    return [int(value)] * n_layers


def _delta_state_bytes(g: GGUFFile, block: int) -> int:
    """Bytes one linear-attention layer keeps for one sequence when the file gives no ssm sizes
    (GLM-5.3-Flash's KDA layers: a head_dim x head_dim matrix per head, and the last
    kernel - 1 inputs of each of its convolutions), as 32-bit floats."""
    head_dim = int(g.arch_key("kda.head_dim", 0) or 0)
    if not head_dim:
        return 0
    prefix = f"blk.{block}."
    mine = [t for t in g.tensors if t.name.startswith(prefix)]
    heads = next((t.shape[0] for t in mine if t.name == prefix + "ssm_a" or t.name.startswith(prefix + "ssm_a.")), 0)
    conv = sum((t.shape[0] - 1) * t.shape[-1] for t in mine if ".ssm_conv1d" in t.name and len(t.shape) >= 2)
    return (int(heads) * head_dim * head_dim + conv) * 4


def _attention_key(g: GGUFFile, suffix: str, default: Any = None) -> Any:
    """``<arch>.attention.<suffix>``, as llama.cpp writes it, or ``<arch>.<suffix>``."""
    value = g.arch_key(f"attention.{suffix}")
    return value if value is not None else g.arch_key(suffix, default)


def _largest(value: Any) -> int:
    """A count that some models give once and others layer by layer (Nemotron-H: 0 on a layer
    without experts): the count of the layers that have them."""
    if isinstance(value, list):
        return max((int(v) for v in value), default=0)
    return int(value or 0)


def _vocabulary(g: GGUFFile) -> int:
    """How many tokens the model has: from its metadata, else from the shape of its embedding table."""
    n = g.arch_key("vocab_size")
    if n:
        return int(n)
    tokens = g.metadata.get("tokenizer.ggml.tokens")
    if isinstance(tokens, list) and tokens:
        return len(tokens)
    for t in g.tensors:
        if t.name == "token_embd.weight" and len(t.shape) == 2:
            return int(max(t.shape))  # (embedding length, vocabulary)
    return 0


def _recurrent_state_bytes(g: GGUFFile) -> int:
    """Bytes one recurrent layer keeps for one sequence, as llama.cpp sizes them (32-bit floats):
    its state (state_size x inner_size) and the last conv_kernel - 1 inputs of its convolution.

    Checked against Qwen3.8-27B on llama.cpp b12375: 48 such layers and 4 sequences
    gave S 576.00 MiB and R 22.50 MiB, exactly these two terms."""
    d_state = int(g.arch_key("ssm.state_size", 0) or 0)
    d_inner = int(g.arch_key("ssm.inner_size", 0) or 0)
    n_group = int(g.arch_key("ssm.group_count", 0) or 0)
    d_conv = int(g.arch_key("ssm.conv_kernel", 0) or 0)
    state = d_state * d_inner
    conv = max(0, d_conv - 1) * (d_inner + 2 * n_group * d_state)
    return (state + conv) * 4


def profile_from_gguf(g: GGUFFile, kv_bytes_per_element: int = 2) -> ModelProfile:
    """Group tensors by block and estimate the KV cache from the metadata."""
    n_layers = g.arch_key("block_count")
    block_ids = {int(m.group(1)) for t in g.tensors if (m := _BLOCK_RE.match(t.name))}
    if n_layers is None:
        n_layers = max(block_ids) + 1 if block_ids else 0
    n_layers = int(n_layers)
    # Blocks at the end that only predict tokens further ahead (MTP). llama.cpp counts them in
    # block_count, so they keep their place in -ngl and -ts, but it loads none of their tensors.
    ahead = min(n_layers, max(0, int(g.arch_key("nextn_predict_layers", 0) or 0)))
    used_layers = n_layers - ahead

    dense = [0] * n_layers
    experts = [0] * n_layers
    head = 0
    embed = 0
    not_loaded = 0
    has_output = False
    token_embed = 0
    notes: list[str] = []
    other: list[str] = []

    for t in g.tensors:
        m = _BLOCK_RE.match(t.name)
        if m:
            i = int(m.group(1))
            if i >= n_layers:
                # e.g. an MTP / next-token-prediction block beyond block_count
                other.append(t.name)
                continue
            if i >= used_layers:
                not_loaded += t.nbytes
                continue
            if "_exps." in t.name:
                experts[i] += t.nbytes
            else:
                dense[i] += t.nbytes
        elif t.name.startswith("output"):
            head += t.nbytes
            if t.name == "output.weight":
                has_output = True
        elif t.name.startswith("token_embd"):
            embed += t.nbytes
            token_embed += t.nbytes
        elif _EMBED_RE.search(t.name):
            # Other tables a token's row is looked up in (per-layer or n-gram embeddings): kept in RAM too.
            embed += t.nbytes
        else:
            other.append(t.name)

    if not has_output:
        # Tied embeddings: the output head multiplies by the embedding table,
        # so a copy of it has to live on the last GPU.
        head += token_embed
        notes.append("tied embeddings: the embedding table is also used as the output head")

    if other:
        # Put unclaimed tensors (rope tables, MTP heads...) with the head so they are counted.
        claimed = sum(t.nbytes for t in g.tensors if t.name in set(other))
        head += claimed
        notes.append(f"{len(other)} tensors outside the blocks were counted with the output head")
    if ahead:
        notes.append(f"the last {ahead} of the {n_layers} blocks only predict tokens further ahead "
                     f"({not_loaded / 1024**3:.2f} GiB): llama.cpp does not load them, so they are counted as empty")

    n_head = g.arch_key("attention.head_count", 0) or 0
    n_embd = int(g.arch_key("embedding_length", 0) or 0)
    head_counts = _per_layer(n_head, n_layers, 0)
    kv_counts = _per_layer(g.arch_key("attention.head_count_kv"), n_layers, 0)
    if not any(kv_counts):
        kv_counts = head_counts

    key_len = g.arch_key("attention.key_length")
    value_len = g.arch_key("attention.value_length")
    # Sliding window attention (Gemma 4): the layers the pattern marks look back at the last `window`
    # tokens only, and may have heads of another size.
    window = int(_attention_key(g, "sliding_window", 0) or 0)
    pattern = _attention_key(g, "sliding_window_pattern")
    sliding = [bool(x) for x in pattern] if window > 0 and isinstance(pattern, list) and len(pattern) == n_layers \
        else [False] * n_layers
    key_swa = _attention_key(g, "key_length_swa")
    value_swa = _attention_key(g, "value_length_swa")
    kv_per_token: list[int] = []
    for i in range(n_layers):
        heads = head_counts[i] or (max(head_counts) if any(head_counts) else 0)
        head_dim = (n_embd // heads) if heads else 0
        k = int(key_len) if key_len else head_dim
        v = int(value_len) if value_len else head_dim
        if sliding[i]:
            k = int(key_swa) if key_swa else k
            v = int(value_swa) if value_swa else v
        kv_per_token.append(kv_counts[i] * (k + v) * kv_bytes_per_element)
    if not any(kv_per_token):
        notes.append("no attention metadata found: KV cache size is unknown and counted as 0")
    if any(sliding):
        notes.append(f"{sum(sliding)} of the {n_layers} layers look back at the last {window} tokens only")
    # The last layers of some models (Gemma 4 E2B/E4B) use the cache of earlier layers and keep none.
    shared = int(_attention_key(g, "shared_kv_layers", 0) or 0)
    if 0 < shared < n_layers:
        for i in range(n_layers - shared, n_layers):
            kv_per_token[i] = 0
        notes.append(f"the last {shared} layers use the KV cache of earlier layers")
    # Layers that keep a state of a fixed size instead of looking back at every token
    # (e.g. Gated DeltaNet in a hybrid model): no KV cache. The same holds for blocks that are not loaded.
    state = _recurrent_state_bytes(g)
    ssm_blocks = {int(m.group(1)) for t in g.tensors if (m := _BLOCK_RE.match(t.name)) and ".ssm_" in t.name}
    if not state and ssm_blocks:
        state = _delta_state_bytes(g, min(ssm_blocks))
    recurrent = g.arch_key("attention.recurrent_layers")
    if not (isinstance(recurrent, list) and len(recurrent) == n_layers):
        # Some files of a hybrid model have no such list, only how often a layer looks back at
        # every token: the last of every `every` layers does (Qwen3.8-27B: layers 3, 7, ... of 4).
        # Taken as that only next to the sizes of a recurrent state, which no other model has.
        every = int(g.arch_key("full_attention_interval", 0) or 0)
        recurrent = [state > 0 and every > 1 and (i + 1) % every != 0 for i in range(n_layers)]
        kv_listed = g.arch_key("attention.head_count_kv")
        if state > 0 and ssm_blocks:
            # The blocks that hold the recurrent part's weights (ssm_*): Mamba hybrids such as
            # Nemotron-H, which also have plain feed-forward or MoE blocks with neither.
            recurrent = [i in ssm_blocks for i in range(n_layers)]
        elif state > 0 and not every and isinstance(kv_listed, list) and len(kv_listed) == n_layers:
            # Mamba hybrids (e.g. Nemotron-H) list, layer by layer, how many KV heads each has: 0 for
            # a layer without attention. Such a layer is taken as recurrent. Some of them may be plain
            # feed-forward layers that keep no state; counting a state for those too errs on the safe side.
            recurrent = [int(kv) == 0 for kv in kv_listed]
    state_bytes = [0] * n_layers
    for i in range(n_layers):
        if i >= used_layers:
            kv_per_token[i] = 0
        elif recurrent[i]:
            kv_per_token[i] = 0
            state_bytes[i] = state
    n_recurrent = sum(1 for i in range(used_layers) if recurrent[i])
    if n_recurrent:
        notes.append(f"{n_recurrent} of the {used_layers} layers keep a state of a fixed size for each sequence "
                     f"({state / 1024**2:.1f} MiB a layer) instead of a KV cache; it is counted with the KV cache")

    return ModelProfile(
        name=str(g.metadata.get("general.name", g.path)),
        architecture=g.architecture,
        layers=[LayerProfile(dense[i], experts[i], kv_per_token[i], state_bytes[i],
                             window if sliding[i] and kv_per_token[i] else 0) for i in range(n_layers)],
        head_bytes=head,
        embed_bytes=embed,
        n_expert=_largest(g.arch_key("expert_count", 0)),
        n_expert_used=_largest(g.arch_key("expert_used_count", 0)),
        n_vocab=_vocabulary(g),
        notes=notes,
    )
