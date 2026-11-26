"""Execute explicitly marked Python examples from the local course documents."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_course import FENCE


def examples(document):
    """Select complete Python fences beginning with '# doc-example:'."""
    fence = None
    block = []
    selected = []
    for number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
        marker = FENCE.match(line)
        if marker and fence is None:
            token, info = marker.groups()
            fence = (token[0], len(token), info.strip(), number)
            block = []
            continue
        if fence is not None:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1]:
                source = "\n".join(block)
                if fence[2] == "python" and source.lstrip().startswith("# doc-example:"):
                    selected.append((fence[3], source))
                fence = None
            else:
                block.append(line)
    if fence is not None:
        raise ValueError(f"{document}:{fence[3]}: unclosed code fence")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=Path, action="append",
                        help="Check a selected document; repeat to select several")
    parser.add_argument("--list", action="store_true", help="List without executing")
    parser.add_argument("--timeout", type=int, default=90, help="Seconds per document")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        namespace = {"__name__": "__doc_example__", "__file__": str(args.worker)}
        for line, source in examples(args.worker):
            exec(compile("\n" * line + source, str(args.worker), "exec"), namespace)
        return 0
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    documents = args.document or sorted((ROOT / "doc").glob("*.md"))
    total = 0
    failures = 0
    checked = 0
    for document in documents:
        document = document if document.is_absolute() else ROOT / document
        blocks = examples(document)
        if not blocks:
            continue
        checked += 1
        total += len(blocks)
        name = os.path.relpath(document, ROOT)
        if args.list:
            print(f"{name}: {len(blocks)} executable block(s)", flush=True)
            continue
        # Each document gets a fresh interpreter; its marked snippets share
        # variables. Shell fences and debugger fragments are never executed.
        environment = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        try:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(document)],
                cwd=ROOT, env=environment, capture_output=True, text=True,
                timeout=args.timeout,
            )
        except subprocess.TimeoutExpired:
            failures += 1
            print(f"FAIL {name}: exceeded {args.timeout}s", flush=True)
            continue
        if result.returncode:
            failures += 1
            print(f"FAIL {name}\n{result.stdout}\n{result.stderr}", flush=True)
        else:
            print(f"PASS {name}: {len(blocks)} executable block(s)", flush=True)
    print(f"{checked} documents, {total} executable blocks, {failures} failures")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
