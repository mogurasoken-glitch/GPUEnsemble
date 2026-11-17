import os
import struct
import tempfile
import unittest

from gpuensemble_planner.gguf import GGUFError, read_model, split_parts
from gpuensemble_planner.model import profile_from_gguf

ALIGN = 32


def _str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    if isinstance(value, str):
        return _str(key) + struct.pack("<I", 8) + _str(value)
    if isinstance(value, list) and value and isinstance(value[0], bool):
        return _str(key) + struct.pack("<IIQ", 9, 7, len(value)) + struct.pack(f"<{len(value)}?", *value)
    if isinstance(value, list):
        return _str(key) + struct.pack("<IIQ", 9, 4, len(value)) + struct.pack(f"<{len(value)}I", *value)
    return _str(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def write_gguf(path: str, metadata: dict, tensors: list[tuple[str, int]]) -> None:
    """Write a GGUF v3 file whose tensors are zero bytes of the given sizes."""
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += _kv(key, value)
    offset = 0
    infos = b""
    for name, nbytes in tensors:
        infos += _str(name) + struct.pack("<I", 1) + struct.pack("<Q", nbytes) + struct.pack("<IQ", 0, offset)
        offset += (nbytes + ALIGN - 1) // ALIGN * ALIGN
    header += infos
    pad = (-len(header)) % ALIGN
    with open(path, "wb") as f:
        f.write(header + b"\0" * pad)
        for i, (_, nbytes) in enumerate(tensors):
            f.write(b"\0" * nbytes)
            if i < len(tensors) - 1:
                f.write(b"\0" * ((-nbytes) % ALIGN))


DENSE_META = {
    "general.architecture": "llama",
    "general.name": "tiny",
    "llama.block_count": 2,
    "llama.embedding_length": 64,
    "llama.attention.head_count": 8,
    "llama.attention.head_count_kv": 2,
}


class ReadGGUFTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name: str) -> str:
        return os.path.join(self.tmp.name, name)

    def test_sizes_come_from_offsets(self) -> None:
        p = self.path("m.gguf")
        write_gguf(
            p,
            DENSE_META,
            [
                ("token_embd.weight", 1000),
                ("blk.0.attn_q.weight", 640),
                ("blk.0.ffn_up.weight", 2000),
                ("blk.1.attn_q.weight", 640),
                ("blk.1.ffn_up.weight", 2000),
                ("output_norm.weight", 64),
                ("output.weight", 1000),
            ],
        )
        g = read_model(p)
        sizes = {t.name: t.nbytes for t in g.tensors}
        # Each size is the real size rounded up to the alignment, except the last.
        self.assertEqual(sizes["blk.0.attn_q.weight"], 640)
        self.assertEqual(sizes["token_embd.weight"], 1024)
        self.assertEqual(sizes["output.weight"], 1000)

        prof = profile_from_gguf(g)
        self.assertEqual(prof.n_layers, 2)
        self.assertEqual(prof.layers[0].dense_bytes, 640 + 2016)  # 2000 padded to 32
        self.assertEqual(prof.head_bytes, 64 + 1000)
        self.assertEqual(prof.embed_bytes, 1024)
        # 2 KV heads x (8 + 8) head dim x 2 bytes (f16)
        self.assertEqual(prof.layers[0].kv_bytes_per_token, 2 * 16 * 2)
        self.assertFalse(prof.is_moe)

    def test_vocabulary_size(self) -> None:
        p = self.path("m.gguf")
        tensors = [("token_embd.weight", 1000), ("blk.0.attn_q.weight", 640), ("blk.1.attn_q.weight", 640),
                   ("output.weight", 1000)]
        write_gguf(p, dict(DENSE_META, **{"llama.vocab_size": 32000, "tokenizer.ggml.tokens": [0] * 7}), tensors)
        prof = profile_from_gguf(read_model(p))
        self.assertEqual((prof.n_vocab, prof.logits_bytes), (32000, 128000))  # a 32-bit float for each
        write_gguf(p, dict(DENSE_META, **{"tokenizer.ggml.tokens": [0] * 7}), tensors)  # else the token list
        self.assertEqual(profile_from_gguf(read_model(p)).n_vocab, 7)
        write_gguf(p, DENSE_META, tensors)
        self.assertEqual(profile_from_gguf(read_model(p)).n_vocab, 0)  # not known: counted as nothing

    def test_moe_experts_and_tied_embeddings(self) -> None:
        meta = dict(DENSE_META, **{"llama.expert_count": 8, "llama.expert_used_count": 2})
        p = self.path("moe.gguf")
        write_gguf(
            p,
            meta,
            [
                ("token_embd.weight", 512),
                ("blk.0.attn_q.weight", 64),
                ("blk.0.ffn_up_exps.weight", 4096),
                ("blk.1.attn_q.weight", 64),
                ("blk.1.ffn_up_exps.weight", 4096),
                ("output_norm.weight", 64),
            ],
        )
        prof = profile_from_gguf(read_model(p))
        self.assertTrue(prof.is_moe)
        self.assertEqual(prof.layers[1].expert_bytes, 4096)
        self.assertEqual(prof.active_expert_ratio, 0.25)
        # No output.weight: the embedding table doubles as the head.
        self.assertEqual(prof.head_bytes, 64 + 512)

    def test_other_lookup_tables_stay_in_ram(self) -> None:
        # Gemma 4 E4B's per-layer embeddings, Qwen3.8-Flash-Next's n-gram table: rows looked up per token,
        # kept in RAM like the token embeddings, and not part of a tied head.
        p = self.path("tables.gguf")
        write_gguf(p, DENSE_META, [("token_embd.weight", 512), ("per_layer_token_embd.weight", 4096),
                                   ("ngram_embd.weight", 8192), ("blk.0.attn_q.weight", 64),
                                   ("blk.1.attn_q.weight", 64), ("output_norm.weight", 64)])
        prof = profile_from_gguf(read_model(p))
        self.assertEqual(prof.embed_bytes, 512 + 4096 + 8192)
        self.assertEqual(prof.head_bytes, 64 + 512)

    def test_hybrid_model_with_a_block_that_is_not_loaded(self) -> None:
        # Like Qwen3.8-27B: three of every four layers keep a state of a fixed size instead of a KV cache,
        # and one more block at the end, counted in block_count, only predicts tokens further ahead.
        meta = {
            "general.architecture": "qwen35",
            "general.name": "tiny hybrid",
            "qwen35.block_count": 5,
            "qwen35.embedding_length": 64,
            "qwen35.attention.head_count": 8,
            "qwen35.attention.head_count_kv": 2,
            "qwen35.attention.key_length": 16,
            "qwen35.attention.value_length": 16,
            "qwen35.nextn_predict_layers": 1,
            "qwen35.attention.recurrent_layers": [True, True, True, False, True],
            "qwen35.ssm.state_size": 4,
            "qwen35.ssm.inner_size": 24,
            "qwen35.ssm.group_count": 2,
            "qwen35.ssm.conv_kernel": 4,
        }
        tensors = [("token_embd.weight", 1024)] + [(f"blk.{i}.attn_q.weight", 640) for i in range(4)]
        tensors += [("blk.4.nextn.eh_proj.weight", 640), ("blk.4.attn_q.weight", 320),
                    ("output_norm.weight", 64), ("output.weight", 1024)]
        p = self.path("hybrid.gguf")
        write_gguf(p, meta, tensors)
        prof = profile_from_gguf(read_model(p))
        # llama.cpp counts the last block as a layer too, so -ngl and -ts stay in step with it...
        self.assertEqual(prof.n_layers, 5)
        # ...but loads nothing of it,
        self.assertEqual([layer.dense_bytes for layer in prof.layers], [640] * 4 + [0])
        self.assertEqual(prof.head_bytes, 64 + 1024)
        # and only the one layer that looks back at every token has a KV cache: 2 heads x (16 + 16) x 2 bytes.
        self.assertEqual([layer.kv_bytes_per_token for layer in prof.layers], [0, 0, 0, 128, 0])
        # The other layers keep, for each sequence, a state (4 x 24) and three inputs of their
        # convolution (24 + 2 x 2 x 4 each), as 32-bit floats.
        self.assertEqual([layer.state_bytes for layer in prof.layers], [(96 + 120) * 4] * 3 + [0, 0])
        self.assertEqual(len(prof.notes), 2, prof.notes)
        self.assertIn("the last 1 of the 5 blocks", prof.notes[0])
        self.assertIn("3 of the 4 layers", prof.notes[1])

        # Another file of the same model (Qwen3.8-27B Q8_0) has no list of its recurrent layers,
        # only how often a layer looks back at every token. llama.cpp reads the same layout from it.
        del meta["qwen35.attention.recurrent_layers"]
        write_gguf(p, dict(meta, **{"qwen35.full_attention_interval": 4}), tensors)
        self.assertEqual(profile_from_gguf(read_model(p)).layers, prof.layers)
        # Without the sizes of a recurrent state, that number says nothing about a KV cache.
        plain = {key: value for key, value in meta.items() if ".ssm." not in key}
        write_gguf(p, dict(plain, **{"qwen35.full_attention_interval": 4}), tensors)
        self.assertEqual([layer.kv_bytes_per_token for layer in profile_from_gguf(read_model(p)).layers],
                         [128, 128, 128, 128, 0])

    def test_recurrent_state_as_llama_cpp_reports_it(self) -> None:
        # Qwen3.8-27B on llama.cpp b12375, 48 recurrent layers and 4 sequences:
        # "llama_memory_recurrent: size = 598.50 MiB (...), R (f32): 22.50 MiB, S (f32): 576.00 MiB".
        meta = {
            "general.architecture": "qwen35",
            "qwen35.block_count": 1,
            "qwen35.attention.recurrent_layers": [True],
            "qwen35.ssm.state_size": 128,
            "qwen35.ssm.inner_size": 6144,
            "qwen35.ssm.group_count": 16,
            "qwen35.ssm.conv_kernel": 4,
        }
        p = self.path("state.gguf")
        write_gguf(p, meta, [("blk.0.attn_q.weight", 64)])
        layer = profile_from_gguf(read_model(p)).layers[0]
        self.assertEqual(layer.state_bytes * 48 * 4, int(598.5 * 1024**2))

    def test_mamba_hybrid_lists_kv_heads_layer_by_layer(self) -> None:
        # Like Nemotron-H: no list of recurrent layers and no interval, but a KV head count for each
        # layer, 0 where the layer has no attention.
        meta = {
            "general.architecture": "nemotron_h",
            "nemotron_h.block_count": 4,
            "nemotron_h.embedding_length": 64,
            "nemotron_h.attention.head_count": [0, 8, 0, 0],
            "nemotron_h.attention.head_count_kv": [0, 2, 0, 0],
            "nemotron_h.ssm.state_size": 4,
            "nemotron_h.ssm.inner_size": 24,
            "nemotron_h.ssm.group_count": 2,
            "nemotron_h.ssm.conv_kernel": 4,
        }
        p = self.path("mamba.gguf")
        write_gguf(p, meta, [(f"blk.{i}.w.weight", 64) for i in range(4)])
        prof = profile_from_gguf(read_model(p))
        # 2 KV heads x (8 + 8) x 2 bytes on the one attention layer, a state on the others.
        self.assertEqual([layer.kv_bytes_per_token for layer in prof.layers], [0, 64, 0, 0])
        self.assertEqual([layer.state_bytes for layer in prof.layers], [864, 0, 864, 864])

        # Expert counts given layer by layer too (Nemotron-3.5-Bolt: 0 or 9 for each layer).
        write_gguf(p, dict(meta, **{"nemotron_h.expert_count": 16, "nemotron_h.expert_used_count": [0, 0, 4, 4]}),
                   [(f"blk.{i}.w.weight", 64) for i in range(4)])
        prof = profile_from_gguf(read_model(p))
        self.assertEqual((prof.n_expert, prof.n_expert_used), (16, 4))

    def test_sliding_window_layers_like_gemma_4(self) -> None:
        # Gemma-4-31B: 5 of every 6 layers look back at the last 1024 tokens, with 16 KV heads of 256;
        # every 6th looks back at everything, with 4 KV heads of 512.
        meta = {
            "general.architecture": "gemma4",
            "gemma4.block_count": 6,
            "gemma4.embedding_length": 64,
            "gemma4.attention.head_count": 32,
            "gemma4.attention.head_count_kv": [16, 16, 16, 16, 16, 4],
            "gemma4.attention.key_length": 512,
            "gemma4.attention.value_length": 512,
            "gemma4.attention.key_length_swa": 256,
            "gemma4.attention.value_length_swa": 256,
            "gemma4.attention.sliding_window": 1024,
            "gemma4.attention.sliding_window_pattern": [True, True, True, True, True, False],
        }
        p = self.path("gemma4.gguf")
        tensors = [(f"blk.{i}.attn_k.weight", 64) for i in range(6)] + [("token_embd.weight", 256)]
        write_gguf(p, meta, tensors)
        prof = profile_from_gguf(read_model(p))
        self.assertEqual([layer.kv_bytes_per_token for layer in prof.layers], [16 * 512 * 2] * 5 + [4 * 1024 * 2])
        self.assertEqual([layer.kv_window for layer in prof.layers], [1024] * 5 + [0])
        self.assertTrue(any("look back at the last 1024 tokens" in n for n in prof.notes), prof.notes)

        # Gemma-4-E4B: the last layers use the cache of earlier ones.
        write_gguf(p, dict(meta, **{"gemma4.attention.shared_kv_layers": 2}), tensors)
        prof = profile_from_gguf(read_model(p))
        self.assertEqual([layer.kv_bytes_per_token for layer in prof.layers], [16 * 512 * 2] * 4 + [0, 0])
        self.assertEqual([layer.kv_window for layer in prof.layers], [1024] * 4 + [0, 0])

        # Without the pattern, a window size alone (gpt-oss has one) changes nothing.
        del meta["gemma4.attention.sliding_window_pattern"]
        write_gguf(p, meta, tensors)
        self.assertEqual([layer.kv_window for layer in profile_from_gguf(read_model(p)).layers], [0] * 6)

    def test_mamba_blocks_are_known_by_their_weights(self) -> None:
        # Nemotron-3.5-Bolt: attention, MoE and Mamba blocks. Only the Mamba ones (ssm_* weights) keep a state.
        meta = {
            "general.architecture": "nemotron_h_moe",
            "nemotron_h_moe.block_count": 3,
            "nemotron_h_moe.embedding_length": 64,
            "nemotron_h_moe.attention.head_count": 8,
            "nemotron_h_moe.attention.head_count_kv": [2, 0, 0],
            "nemotron_h_moe.ssm.state_size": 4,
            "nemotron_h_moe.ssm.inner_size": 24,
            "nemotron_h_moe.ssm.group_count": 2,
            "nemotron_h_moe.ssm.conv_kernel": 4,
        }
        p = self.path("bolt.gguf")
        write_gguf(p, meta, [("blk.0.attn_q.weight", 64), ("blk.1.ffn_up_exps.weight", 64),
                             ("blk.2.ssm_in.weight", 64), ("blk.2.ssm_conv1d.weight", 64)])
        prof = profile_from_gguf(read_model(p))
        self.assertEqual([layer.kv_bytes_per_token for layer in prof.layers], [64, 0, 0])
        self.assertEqual([layer.state_bytes for layer in prof.layers], [0, 0, 864])

    def test_linear_attention_state_without_ssm_sizes(self) -> None:
        # GLM-5.3-Flash: KDA layers (ssm_* weights, no ssm.* sizes) and MLA layers. A KDA layer keeps a
        # head_dim x head_dim matrix for each of its heads and the last 3 inputs of its convolutions.
        meta = {
            "general.architecture": "glm5-next",
            "glm5-next.block_count": 2,
            "glm5-next.embedding_length": 64,
            "glm5-next.attention.head_count": 8,
            "glm5-next.attention.head_count_kv": [0, 1],
            "glm5-next.attention.key_length": 16,
            "glm5-next.attention.value_length": 16,
            "glm5-next.kda.head_dim": 4,
        }
        p = self.path("kda.gguf")
        write_gguf(p, meta, [("blk.0.ssm_a", 64), ("blk.0.ssm_conv1d_q.weight", 64),
                             ("blk.0.ssm_conv1d_k.weight", 64), ("blk.1.attn_kv_a_mqa.weight", 64)])
        # write_gguf gives every tensor one dimension (its size); shape it as the real file does.
        g = read_model(p)
        for t in g.tensors:
            if t.name == "blk.0.ssm_a":
                t.shape = (6,)
            elif ".ssm_conv1d" in t.name:
                t.shape = (4, 1, 16)
        prof = profile_from_gguf(g)
        self.assertEqual([layer.state_bytes for layer in prof.layers], [(6 * 4 * 4 + 2 * 3 * 16) * 4, 0])
        self.assertEqual([layer.kv_bytes_per_token for layer in prof.layers], [0, 64])

    def test_a_model_without_such_layers_is_read_as_before(self) -> None:
        tensors = [("token_embd.weight", 1024), ("blk.0.attn_q.weight", 640), ("blk.1.attn_q.weight", 640),
                   ("output.weight", 1024)]
        p = self.path("m.gguf")
        # A list of the wrong length says nothing about the layers.
        write_gguf(p, dict(DENSE_META, **{"llama.attention.recurrent_layers": [True, False, True]}), tensors)
        prof = profile_from_gguf(read_model(p))
        self.assertEqual([layer.kv_bytes_per_token for layer in prof.layers], [64, 64])
        self.assertEqual([layer.state_bytes for layer in prof.layers], [0, 0])
        self.assertEqual(prof.notes, [])

    def test_split_model_reads_every_part(self) -> None:
        write_gguf(self.path("m-00001-of-00002.gguf"), DENSE_META, [("blk.0.attn_q.weight", 100)])
        write_gguf(self.path("m-00002-of-00002.gguf"), {}, [("blk.1.attn_q.weight", 200)])
        parts = split_parts(self.path("m-00002-of-00002.gguf"))
        self.assertEqual(len(parts), 2)
        g = read_model(parts[0])
        self.assertEqual(sorted(t.name for t in g.tensors), ["blk.0.attn_q.weight", "blk.1.attn_q.weight"])

    def test_missing_part_is_an_error(self) -> None:
        write_gguf(self.path("m-00001-of-00002.gguf"), DENSE_META, [("blk.0.attn_q.weight", 100)])
        with self.assertRaises(GGUFError):
            read_model(self.path("m-00001-of-00002.gguf"))

    def test_not_gguf(self) -> None:
        p = self.path("x.gguf")
        with open(p, "wb") as f:
            f.write(b"NOPE" + b"\0" * 64)
        with self.assertRaises(GGUFError):
            read_model(p)


if __name__ == "__main__":
    unittest.main()
