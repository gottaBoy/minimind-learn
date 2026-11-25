"""Check local course links, Python snippets, and debugger configurations."""

import argparse
import ast
import json
from pathlib import Path
import re
import tomllib
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
INLINE_CODE = re.compile(r"(`+).*?\1")


def check_documents(root):
    errors = []
    counts = {"documents": 0, "links": 0, "python_blocks": 0}
    documents = [root / "readme.md", *sorted((root / "doc").glob("*.md"))]
    for document in documents:
        if not document.is_file():
            errors.append(f"{document.relative_to(root)}: file is missing")
            continue
        counts["documents"] += 1
        name = document.relative_to(root)
        fence = None
        block = []
        for number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            marker = FENCE.match(line)
            if marker and fence is None:
                token, info = marker.groups()
                fence = (token[0], len(token), info.strip(), number)
                block = []
                continue
            if fence is not None:
                if marker and marker[1][0] == fence[0] and len(marker[1]) >= fence[1]:
                    if fence[2] == "python":
                        counts["python_blocks"] += 1
                        try:
                            ast.parse("\n".join(block))
                        except SyntaxError as error:
                            location = fence[3] + (error.lineno or 1)
                            errors.append(f"{name}:{location}: Python syntax: {error.msg}")
                    fence = None
                else:
                    block.append(line)
                continue
            # A log such as `Epoch:[1/1](1/3)` is not a Markdown link.
            prose = INLINE_CODE.sub("", line)
            for match in LINK.finditer(prose):
                target = match[1].split("#", 1)[0]
                if not target or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                    continue
                counts["links"] += 1
                if not (document.parent / unquote(target)).exists():
                    errors.append(f"{name}:{number}: missing local link: {target}")
        if fence is not None:
            errors.append(f"{name}:{fence[3]}: unclosed code fence")
    return errors, counts


def check_configurations(root):
    errors = []
    try:
        config = json.loads((root / ".vscode/launch.json").read_text(encoding="utf-8"))
        for entry in config["configurations"]:
            if "program" in entry:
                path = entry["program"].replace("${workspaceFolder}/", "")
            else:
                path = entry["module"].replace(".", "/") + ".py"
            if not (root / path).is_file():
                errors.append(f".vscode/launch.json: {entry['name']}: missing entry point {path}")

        tree = ast.parse((root / "scripts/lesson_lab.py").read_text(encoding="utf-8"))
        lessons = next(
            node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "LESSONS"
                    for target in node.targets)
        )
        if not isinstance(lessons, ast.Dict):
            raise ValueError("LESSONS must be a dictionary literal")
        lesson_names = {ast.literal_eval(key) for key in lessons.keys}
        picker = next(item for item in config["inputs"] if item["id"] == "lessonName")
        if set(picker["options"]) != lesson_names:
            errors.append(".vscode/launch.json: lessonName options do not match LESSONS")
        if picker["default"] not in lesson_names:
            errors.append(".vscode/launch.json: invalid default lesson")
        if "all" in lesson_names:
            errors.append("scripts/lesson_lab.py: all is reserved for running every lesson")

        manifest = tomllib.loads((root / "pixi.toml").read_text(encoding="utf-8"))
        for name in ("lesson", "check-course"):
            command = manifest["tasks"][name]
            script = command.split()[1]
            if not (root / script).is_file():
                errors.append(f"pixi.toml: {name}: missing entry point {script}")
    except (OSError, ValueError, SyntaxError, KeyError, TypeError, StopIteration, IndexError) as error:
        errors.append(f"Course configuration could not be checked: {error}")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository directory to check")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    errors, counts = check_documents(root)
    errors.extend(check_configurations(root))
    for error in errors:
        print(error)
    if errors:
        print(f"FAIL: {len(errors)} course issue(s)")
        return 1
    print(f"PASS: {counts['documents']} documents, {counts['links']} local references, "
          f"{counts['python_blocks']} Python blocks, VS Code and Pixi entry points")
    print("Local checks only: snippets are parsed, not executed; remote URLs are not requested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
