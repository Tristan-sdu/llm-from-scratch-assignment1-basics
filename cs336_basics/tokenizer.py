from collections.abc import Iterable
from typing import Any

import regex as re

from cs336_basics.bpe import PATTERN


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.id_to_bytes: dict[int, bytes] = dict(vocab)
        self.bytes_to_id: dict[bytes, int] = {tok: idx for idx, tok in self.id_to_bytes.items()}
        self.special_tokens: list[str] = special_tokens or []
        self.special_bytes: set[bytes] = {tok.encode("utf-8") for tok in self.special_tokens}

        for tok_b in self.special_bytes:
            if tok_b not in self.bytes_to_id:
                new_id = len(self.id_to_bytes)
                self.id_to_bytes[new_id] = tok_b
                self.bytes_to_id[tok_b] = new_id

        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}
        self.special_re = self._build_special_pattern()

    def _build_special_pattern(self) -> re.Pattern | None:
        if not self.special_tokens:
            return None
        escaped = [re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)]
        return re.compile("|".join(escaped))

    def _bpe(self, token_bytes: bytes) -> list[int]:
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
        ids: list[int] = []
        for match in PATTERN.finditer(text):
            tok = match.group(0)
            if tok:
                ids.extend(self._bpe(tok.encode("utf-8")))
        return ids

    def encode(self, text: str) -> list[int]:
        if not self.special_re:
            return self._encode_plain_text(text)
        ids: list[int] = []
        cursor = 0
        for match in self.special_re.finditer(text):
            start, end = match.start(), match.end()
            if start > cursor:
                ids.extend(self._encode_plain_text(text[cursor:start]))
            tok_bytes = match.group(0).encode("utf-8")
            ids.append(self.bytes_to_id[tok_bytes])
            cursor = end
        if cursor < len(text):
            ids.extend(self._encode_plain_text(text[cursor:]))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, token_ids: Iterable[int]) -> str:
        pieces: list[str] = []
        byte_buf = bytearray()
        for token_id in token_ids:
            tok_bytes = self.id_to_bytes[token_id]
            if tok_bytes in self.special_bytes:
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
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
