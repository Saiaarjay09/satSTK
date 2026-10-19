"""
dsp/payload.py - the 63 KB test payload with 64-byte aligned error detection, and helpers to compare what was sent with what came back.

The test payload is plain ASCII text made of 1,008 blocks of exactly 64 bytes (63 KB = 64,512 bytes = 1,008 x 64). Every block is one line:

    B0042 <48 pseudo-random printable characters> 1a2b3c4d\n
    '-----' index  '--- 48 data characters ---'   CRC-32 (8 hex digits) of everything before it, including the space

6 + 48 + 1 + 8 + 1 = 64 bytes. Because every block carries its own CRC-32, a receiver (or you) can tell WHICH 64-byte blocks were damaged,
not just that something went wrong - useful for seeing how few bits it takes to ruin a frame, and for finding a stuck bit pattern.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

BLOCK_BYTES = 64
DATA_CHARS = 48
TEST_BLOCKS = 1008                      # 1008 x 64 = 64,512 bytes = 63 KB
_ALPHABET = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-*/=<>#%&", dtype=np.uint8)


def make_block(index: int, rng: np.random.Generator) -> bytes:
    head = f"B{index % 10000:04d} ".encode("ascii") + bytes(rng.choice(_ALPHABET, DATA_CHARS)) + b" "
    block = head + f"{zlib.crc32(head):08x}".encode("ascii") + b"\n"
    assert len(block) == BLOCK_BYTES
    return block


def generate_test_payload(n_blocks: int = TEST_BLOCKS, seed: int = 63) -> bytes:
    """The 63 KB (by default) test payload: `n_blocks` self-checking 64-byte lines. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    return b"".join(make_block(i, rng) for i in range(n_blocks))


@dataclass
class BlockReport:
    n_blocks: int
    n_ok: int
    bad_blocks: list              # indices of blocks whose CRC (or format) failed
    aligned: bool                 # the length is a whole number of 64-byte blocks
    is_test_payload: bool         # the data looks like the generated test payload

    @property
    def n_bad(self) -> int:
        return len(self.bad_blocks)


def verify_blocks(data: bytes | None, expected_blocks: int | None = None) -> BlockReport:
    """Check every 64-byte block's own CRC-32. Works on any data; `is_test_payload` says whether it has the block format."""
    if not data:
        return BlockReport(expected_blocks or 0, 0, list(range(expected_blocks or 0)), False, False)
    n = len(data) // BLOCK_BYTES
    bad = []
    fmt_ok = 0
    for i in range(n):
        b = data[i * BLOCK_BYTES:(i + 1) * BLOCK_BYTES]
        ok = False
        try:
            ok = (b[0:1] == b"B" and b[5:6] == b" " and b[54:55] == b" " and b[63:64] == b"\n"
                  and int(b[55:63], 16) == zlib.crc32(b[:55]))
        except ValueError:
            ok = False
        fmt_ok += b[0:1] == b"B" and b[5:6] == b" " and b[63:64] == b"\n"
        if not ok:
            bad.append(i)
    total = expected_blocks if expected_blocks is not None else n
    bad += list(range(n, total))                                   # blocks that never arrived
    return BlockReport(total, total - len(bad), bad, len(data) % BLOCK_BYTES == 0, n > 0 and fmt_ok >= 0.5 * n)


def byte_diff(sent: bytes, received: bytes | None):
    """Boolean mask over `sent`: True where the byte is wrong or missing in `received`."""
    s = np.frombuffer(sent, dtype=np.uint8)
    wrong = np.ones(len(s), dtype=bool)
    if received:
        r = np.frombuffer(received, dtype=np.uint8)
        m = min(len(s), len(r))
        wrong[:m] = s[:m] != r[:m]
    return wrong
