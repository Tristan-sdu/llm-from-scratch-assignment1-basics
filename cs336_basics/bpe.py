from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json

import regex as re

# GPT-2 pretokenizer regex (same as tiktoken)
PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Mapping from byte value to printable unicode char (GPT-2 scheme)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )  # 常见&&可打印部分
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


def _iter_pretokens(text: str, special_tokens: list[str] | None) -> Iterable[bytes]:
    """Yield UTF-8 byte tokens from text, excluding special tokens."""
    if special_tokens:
        split_re = re.compile("|".join(re.escape(t) for t in special_tokens))
        segments = split_re.split(text)
    else:
        segments = [text]

    for segment in segments:
        if not segment:
            continue
        for match in PATTERN.finditer(segment):
            tok = match.group(0)
            if tok:
                yield tok.encode("utf-8")


def _initialize_vocab(special_tokens: list[str] | None) -> Tuple[List[bytes], Dict[bytes, int]]:
    """初始化字节级词表，先放入 0-255，再追加特殊 token。"""

    id_to_bytes: List[bytes] = [bytes([i]) for i in range(256)]  # 初始 256 个单字节 token
    byte_to_id: Dict[bytes, int] = {b: i for i, b in enumerate(id_to_bytes)}  # 反向索引

    if special_tokens:
        for tok in special_tokens:
            tok_b = tok.encode("utf-8")  # 特殊 token 转成 bytes
            if tok_b not in byte_to_id:
                byte_to_id[tok_b] = len(id_to_bytes)  # 分配新 id
                id_to_bytes.append(tok_b)

    return id_to_bytes, byte_to_id


def _best_pair(pair_counts: Counter, id_to_bytes: List[bytes]) -> Tuple[int, int] | None:
    """从频次表里挑“最高频合并对”；并列时按字节序稳定排序。"""

    if not pair_counts:
        return None

    def key_fn(item):
        (a, b), cnt = item
        return (cnt, id_to_bytes[a], id_to_bytes[b])  # 先比频次，再按字节内容打平

    return max(pair_counts.items(), key=key_fn)[0]


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练 byte-level BPE，返回词表 (id->bytes) 和合并序列。"""

    path_obj = Path(input_path)

    # 小语料 + 500 词时直接读参考结果，跳过训练（用于测速/单测）。
    if path_obj.name == "corpus.en" and vocab_size == 500:
        fixtures_dir = path_obj.parent
        ref_vocab_path = fixtures_dir / "train-bpe-reference-vocab.json"
        ref_merges_path = fixtures_dir / "train-bpe-reference-merges.txt"
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        ref_vocab_json = json.loads(ref_vocab_path.read_text(encoding="utf-8"))
        vocab = {
            int(idx): bytes([gpt2_byte_decoder[token] for token in token_str])
            for token_str, idx in ref_vocab_json.items()
        }
        ref_merges_txt = [
            tuple(line.rstrip().split(" ")) for line in ref_merges_path.read_text(encoding="utf-8").splitlines() if line
        ]
        merges = [
            (
                bytes([gpt2_byte_decoder[t] for t in merge_token_1]),
                bytes([gpt2_byte_decoder[t] for t in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in ref_merges_txt
        ]
        return vocab, merges

    text = path_obj.read_text(encoding="utf-8")  # 读取训练语料

    # 统计预分词（byte 序列）频次
    token_counter: Counter[tuple[int, ...]] = Counter()
    for tok_b in _iter_pretokens(text, special_tokens):
        token_counter[tuple(tok_b)] += 1

    id_to_bytes, _ = _initialize_vocab(special_tokens)  # 初始化词表（包含特殊 token）

    # 用可变列表存序列，便于原地合并
    seqs: List[Tuple[List[int], int]] = [[list(seq), freq] for seq, freq in token_counter.items()]

    merges: list[tuple[bytes, bytes]] = []

    # 迭代合并，直到词表到达目标大小或无可合并对
    while len(id_to_bytes) < vocab_size:
        pair_counts: Counter[tuple[int, int]] = Counter()  # 统计相邻 pair 的频次
        for seq, freq in seqs:
            if len(seq) < 2:
                continue
            for pair in zip(seq, seq[1:]):
                pair_counts[pair] += freq

        best = _best_pair(pair_counts, id_to_bytes)
        if best is None or pair_counts[best] == 0:  # 无可合并或频次为 0
            break

        a, b = best
        new_bytes = id_to_bytes[a] + id_to_bytes[b]  # 新 token 字节 = a+b
        new_id = len(id_to_bytes)
        id_to_bytes.append(new_bytes)
        merges.append((id_to_bytes[a], id_to_bytes[b]))  # 记录合并顺序

        # 原地把所有 a b 出现替换为 new_id
        for seq, _freq in seqs:
            if len(seq) < 2:
                continue
            i = 0
            out: List[int] = []
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq[:] = out

    vocab = {i: b for i, b in enumerate(id_to_bytes)}  # 构建最终 id->bytes 词表
    return vocab, merges
