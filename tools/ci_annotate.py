"""Turn the tail of a log file into GitHub Actions error annotations (visible without logging in). Usage: python tools/ci_annotate.py tests.log [title]"""
import sys

path = sys.argv[1]
title = sys.argv[2] if len(sys.argv) > 2 else "log"
text = open(path, encoding="utf-8", errors="replace").read().replace("\r", "")
lines = [l for l in text.splitlines() if l.strip()][-70:]
chunks, cur = [], ""
for line in lines:
    line = line.replace("%", "%25")[:600]
    if len(cur) + len(line) > 420 and cur:
        chunks.append(cur)
        cur = ""
    cur += (" | " if cur else "") + line
if cur:
    chunks.append(cur)
for i, c in enumerate(chunks[-9:]):
    print(f"::error title={title} {i + 1}::{c}")
