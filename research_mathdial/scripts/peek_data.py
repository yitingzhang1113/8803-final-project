"""
Quick data peek utility for MathDial JSONL files.
Prints the first N samples with key fields to understand structure.
Usage:
  python scripts/peek_data.py data/train.jsonl 3
"""
import json
import os
import sys
from typing import Any, Dict


def peek(path: str, n: int = 3) -> None:
    abs_path = path if os.path.isabs(path) else os.path.join(os.path.dirname(os.path.dirname(__file__)), path)
    print(f"Reading: {abs_path}")
    cnt = 0
    with open(abs_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex: Dict[str, Any] = json.loads(line)
            print("-" * 80)
            print(f"qid: {ex.get('qid')} | scenario: {ex.get('scenario')} | has conversation: {'conversation' in ex}")
            conv = ex.get("conversation", "")
            # Show first ~200 chars of conversation
            snippet = conv[:200].replace("\n", " ") + ("..." if len(conv) > 200 else "")
            print(f"conversation snippet: {snippet}")
            print(f"question: {ex.get('question', '')[:120]}...")
            cnt += 1
            if cnt >= n:
                break


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/train.jsonl"
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    peek(in_path, num)
