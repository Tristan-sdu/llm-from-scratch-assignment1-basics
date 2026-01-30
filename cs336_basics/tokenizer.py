"""实现 GPT-2 兼容的 BPE 分词器，支持特殊 token 与流式编码。"""

from collections.abc import Iterable
from typing import Any

import regex as re

from cs336_basics.bpe import PATTERN


class BPETokenizer:
    """基于给定词表和 merges 的轻量 BPE 分词器。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """初始化 BPE 分词器状态。

        方法：建立 id<->bytes 映射，补充特殊 token，并构建 merge_ranks 与特殊正则。
        关键变量：vocab/merges 决定词表与合并规则；special_tokens 为不可拆分符号。
        解决问题：为编码/解码准备必要的数据结构。
        """
        # 基础表：id->bytes 与 bytes->id 的双向映射，保持与训练得到的词表一致
        self.id_to_bytes: dict[int, bytes] = dict(vocab)
        self.bytes_to_id: dict[bytes, int] = {tok: idx for idx, tok in self.id_to_bytes.items()}
        # 记录特殊 token（如 <|endoftext|>），并预先转换成字节方便比较
        self.special_tokens: list[str] = special_tokens or []
        self.special_bytes: set[bytes] = {tok.encode("utf-8") for tok in self.special_tokens}

        # 若特殊 token 未在原始词表里，为其分配新的 id，确保解码时能还原
        for tok_b in self.special_bytes:
            if tok_b not in self.bytes_to_id:
                new_id = len(self.id_to_bytes)
                self.id_to_bytes[new_id] = tok_b
                self.bytes_to_id[tok_b] = new_id

        # merge_ranks 记录每个可合并 pair 的优先级（越小越先合并）
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}
        # 预编译特殊 token 的正则，用于在分词时先捕获不可拆分的片段
        self.special_re = self._build_special_pattern()

    def _build_special_pattern(self) -> re.Pattern | None:
        """构建匹配特殊 token 的正则。

        方法：将特殊 token 按长度降序拼接为 alternation 正则。
        关键变量：special_tokens 为待匹配的符号集合。
        解决问题：确保编码时优先捕获不可拆分片段。
        """
        if not self.special_tokens:
            return None
        # 为避免前缀匹配出错，按长度从长到短排序再拼接成正则
        escaped = [re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)]
        return re.compile("|".join(escaped))

    def _bpe(self, token_bytes: bytes) -> list[int]:
        """对单个字节序列执行 BPE 合并。

        方法：从最细粒度字节开始，反复选择 merge_ranks 中优先级最高的相邻对合并。
        关键变量：token_bytes 为输入字节序列；merge_ranks 为合并优先级表。
        解决问题：将字节序列映射到 BPE token id 序列。
        """
        # 将一个 token 的字节序列按 BPE 规则反复合并，直到没有可合并的 pair
        parts: list[bytes] = [bytes([b]) for b in token_bytes]
        while True:
            best_rank: int | None = None
            best_idx: int | None = None
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx is None:
                break
            merged = parts[best_idx] + parts[best_idx + 1]
            parts = parts[:best_idx] + [merged] + parts[best_idx + 2 :]
        return [self.bytes_to_id[p] for p in parts]

    def _encode_plain_text(self, text: str) -> list[int]:
        """对普通文本做 BPE 编码（不处理特殊 token）。"""
        # 先用 PATTERN 按 GPT-2 的规则切分 text，再对每个片段执行 BPE
        ids: list[int] = []
        for match in PATTERN.finditer(text):
            tok = match.group(0)
            if tok:
                ids.extend(self._bpe(tok.encode("utf-8")))
        return ids

    def encode(self, text: str) -> list[int]:
        """对文本编码为 token id 序列。

        方法：若有特殊 token，先按正则切分并优先编码；其他片段走 BPE。
        关键变量：text 为输入字符串；special_re 控制特殊 token 匹配。
        解决问题：兼容 GPT-2 风格分词并保留特殊符号。
        """
        # 若不存在特殊 token，直接走普通编码；否则先匹配特殊 token 再编码其余部分
        if not self.special_re:
            return self._encode_plain_text(text)
        ids: list[int] = []
        cursor = 0
        for match in self.special_re.finditer(text):
            start, end = match.start(), match.end()
            # 处理当前特殊 token 之前的普通文本
            if start > cursor:
                ids.extend(self._encode_plain_text(text[cursor:start]))
            tok_bytes = match.group(0).encode("utf-8")
            ids.append(self.bytes_to_id[tok_bytes])
            cursor = end
        # 处理最后一个特殊 token 之后的尾巴
        if cursor < len(text):
            ids.extend(self._encode_plain_text(text[cursor:]))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """流式编码可迭代文本。

        方法：逐块读取并调用 encode，yield token id。
        关键变量：iterable 为文本块迭代器。
        解决问题：避免一次性加载大文本占用内存。
        """
        # 按流式方式对可迭代文本进行分词，避免一次性加载大文件
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, token_ids: Iterable[int]) -> str:
        """将 token id 序列还原为字符串。

        方法：普通 token 先拼成字节缓冲，遇到特殊 token 先 flush 再原样拼接。
        关键变量：token_ids 为 id 序列；special_bytes 标记特殊 token。
        解决问题：可逆还原编码结果并保持特殊 token。
        """
        # 将 token id 序列还原成字符串，保持特殊 token 的原样拼接
        pieces: list[str] = []
        byte_buf = bytearray()
        for token_id in token_ids:
            tok_bytes = self.id_to_bytes[token_id]
            if tok_bytes in self.special_bytes:
                # 遇到特殊 token 前，先把累计的普通字节刷入输出
                if byte_buf:
                    pieces.append(byte_buf.decode("utf-8", errors="replace"))
                    byte_buf.clear()
                pieces.append(tok_bytes.decode("utf-8", errors="replace"))
            else:
                byte_buf.extend(tok_bytes)
        if byte_buf:
            pieces.append(byte_buf.decode("utf-8", errors="replace"))
        return "".join(pieces)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """构建 BPETokenizer 的工厂函数。

    方法：直接实例化 BPETokenizer。
    关键变量：vocab/merges/special_tokens 透传给构造函数。
    解决问题：便于测试或外部代码统一创建分词器。
    """
    # 工厂函数，便于测试代码直接构建分词器实例
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
