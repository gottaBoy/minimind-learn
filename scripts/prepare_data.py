"""Stream a bounded JSONL subset for a short training run."""

import argparse
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/main"
FILES = {"pretrain": "pretrain_t2t_mini.jsonl", "sft": "sft_t2t_mini.jsonl"}


def write_subset(source, output, kind, max_samples):
    """Validate records and replace the destination only after a successful read."""
    if max_samples <= 0:
        raise ValueError("max_samples must be greater than zero")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                if kind == "pretrain":
                    if not isinstance(sample.get("text"), str) or not sample["text"].strip():
                        raise ValueError(f"Line {number}: nonempty text is required")
                    sample = {"text": sample["text"]}
                else:
                    conversations = sample.get("conversations")
                    if not isinstance(conversations, list) or not conversations:
                        raise ValueError(f"Line {number}: conversations are required")
                    if not all(isinstance(turn, dict) and isinstance(turn.get("content"), str)
                               and turn.get("role") in {"system", "user", "assistant"} for turn in conversations):
                        raise ValueError(f"Line {number}: invalid conversation")
                    if not any(turn["role"] == "assistant" and turn["content"].strip() for turn in conversations):
                        raise ValueError(f"Line {number}: an assistant response is required")
                    # This repository trains answer text; it has no separate
                    # reasoning channel or MiniMind-3 protocol implementation.
                    sample = {"conversations": [{"role": turn["role"], "content": turn["content"]}
                                                for turn in conversations]}
                destination.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
                if count == max_samples:
                    break
        if count == 0:
            raise ValueError("The source contains no samples")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=FILES, required=True)
    parser.add_argument("--max_samples", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", help="Optional local JSONL path or HTTPS URL")
    args = parser.parse_args()
    output = args.output or Path(f"dataset/{args.kind}_day.jsonl")
    output = output if output.is_absolute() else ROOT / output
    source = args.source or f"{BASE_URL}/{FILES[args.kind]}"
    if source.startswith("https://"):
        request = urllib.request.Request(source, headers={"User-Agent": "minimind-learn"})
        with urllib.request.urlopen(request, timeout=60) as response:
            count = write_subset(response, output, args.kind, args.max_samples)
    else:
        local = Path(source).expanduser()
        local = local if local.is_absolute() else ROOT / local
        with local.open("r", encoding="utf-8") as stream:
            count = write_subset(stream, output, args.kind, args.max_samples)
    print(f"Saved {count} {args.kind} samples to {output}")


if __name__ == "__main__":
    main()
