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
    best_pair = None
    max_score = (-1, b"", b"")

    for pair, count in pair_counts.items():
        if count <= 0:
            continue

        # Tie-breaking logic: (count, first_token_bytes, second_token_bytes)
        score = (count, id_to_bytes[pair[0]], id_to_bytes[pair[1]])
        if score > max_score:
            max_score = score
            best_pair = pair

    return best_pair


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练 byte-level BPE，返回词表 (id->bytes) 和合并序列。"""

    path_obj = Path(input_path)
    text = path_obj.read_text(encoding="utf-8")  # 读取训练语料

    # 统计预分词（byte 序列）频次
    token_counter: Counter[tuple[int, ...]] = Counter()
    for tok_b in _iter_pretokens(text, special_tokens):
        token_counter[tuple(tok_b)] += 1

    id_to_bytes, _ = _initialize_vocab(special_tokens)  # 初始化词表（包含特殊 token）

    # 用可变列表存序列，便于原地合并
    seqs: List[List] = [[list(seq), freq] for seq, freq in token_counter.items()]

    merges: list[tuple[bytes, bytes]] = []

    # 初始统计所有相邻 pair 的频次
    pair_counts: Counter[tuple[int, int]] = Counter()
    for seq, freq in seqs:
        for i in range(len(seq) - 1):
            pair_counts[(seq[i], seq[i + 1])] += freq

    # 迭代合并，直到词表到达目标大小或无可合并对
    while len(id_to_bytes) < vocab_size:
        best = _best_pair(pair_counts, id_to_bytes)
        if best is None:
            break

        a, b = best
        new_bytes = id_to_bytes[a] + id_to_bytes[b]  # 新 token 字节 = a+b
        new_id = len(id_to_bytes)
        id_to_bytes.append(new_bytes)
        merges.append((id_to_bytes[a], id_to_bytes[b]))  # 记录合并顺序

        # 增量更新 pair 频次
        for s_info in seqs:
            seq, freq = s_info
            if len(seq) < 2:
                continue

            i = 0
            new_seq = []
            changed = False
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    # 左右相邻对受影响，先减去旧频次
                    pair_counts[(a, b)] -= freq
                    if i > 0:
                        pair_counts[(new_seq[-1], a)] -= freq
                    if i < len(seq) - 2:
                        pair_counts[(b, seq[i + 2])] -= freq

                    # 合并为新 ID
                    new_seq.append(new_id)

                    # 加上新形成的 pair 频次
                    if len(new_seq) > 1:
                        pair_counts[(new_seq[-2], new_id)] += freq
                    if i < len(seq) - 2:
                        pair_counts[(new_id, seq[i + 2])] += freq

                    i += 2
                    changed = True
                else:
                    new_seq.append(seq[i])
                    i += 1
            if changed:
                s_info[0] = new_seq

        # 定期清理 pair_counts 中的 0 项，以加速 _best_pair 遍历
        if len(id_to_bytes) % 100 == 0:
            pair_counts = Counter({k: v for k, v in pair_counts.items() if v > 0})

    vocab = {i: b for i, b in enumerate(id_to_bytes)}  # 构建最终 id->bytes 词表
    return vocab, merges
