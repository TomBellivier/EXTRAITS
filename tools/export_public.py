"""Export the public EXTRAITS repository from this (private) one, with a clean history.

Copies the files tracked by git (`git ls-files`: nothing untracked or ignored) that are
under the ALLOWED paths below, strips the outputs of the notebooks (they may show
private images), then checks the copy: no image, model, table or archive, no private
string (home folders, user names, private datasets), no file over MAX_SIZE. It never
runs git in the target: the commands to create the public history are printed.

    python tools/export_public.py ../EXTRAITS-public            # export + checks
    python tools/export_public.py ../EXTRAITS-public --list     # only list what would be copied
    python tools/export_public.py /tmp/check --working-tree     # check before committing

Review ALLOWED and the printed list before the first push.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Paths of the public repository (files, or folders copied whole). Anything else stays
# private: HANDOFF.md, .TODO.txt, weights, data folders, ...
ALLOWED = [
    ".dockerignore",
    ".gitattributes",
    ".github",
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "LICENSES",
    "MODEL_LICENSE.md",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "docker",
    "docker-compose.yml",
    "docs",
    "examples",
    "pyproject.toml",
    "research",
    "src",
    "tests",
    "tools/export_public.py",
    "tools/upload_models.py",
]

# Never in the public repository, even under an allowed path.
FORBIDDEN_FILES = [
    "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp", "*.webp", "*.gif",
    "*.pt", "*.pth", "*.onnx", "*.safetensors", "*.joblib", "*.pkl", "*.ckpt",
    "*.csv", "*.parquet", "*.zip", "*.tar", "*.gz", "*.7z", "*.db",
    "*/__pycache__/*", "*.pyc", "*.egg-info/*",
]
# Files that are data-like but expected.
EXPECTED_EXCEPTIONS = {"src/extraits/annotation/annotation_protocol.pdf"}
MAX_SIZE = 1_000_000          # bytes; larger files must be listed in EXPECTED_EXCEPTIONS

# Strings that must not leak (checked in every text file of the export).
PRIVATE_PATTERNS = [
    r"/home/[A-Za-z0-9_.-]+",           # home folders
    r"[A-Za-z]:\\Users\\",              # Windows home folders
    r"\btbellivier\b",                  # local user name
    r"\btombe\b",
    r"TomBellivier/all_images",         # private Hugging Face dataset
    r"[A-Za-z0-9._%+-]+@gmail\.com",    # personal e-mail addresses
]
# Lines where a pattern is expected (path relative to the repository: allowed pattern).
PRIVATE_EXCEPTIONS = {
    "tests/test_models.py": r"/home/someone",
    "tools/export_public.py": r".*",
}


def tracked_files(working_tree: bool = False) -> list[str]:
    """Files tracked by git; with `working_tree`, also the untracked ones that are not
    ignored (to check an export before committing)."""
    extra = ["--others", "--exclude-standard"] if working_tree else []
    out = subprocess.run(["git", "ls-files", "-z", "--cached", *extra], cwd=REPO,
                         capture_output=True, check=True)
    return sorted(f for f in out.stdout.decode().split("\0") if f)


def selected(files: list[str]) -> list[str]:
    def allowed(path: str) -> bool:
        return any(path == a or path.startswith(a.rstrip("/") + "/") for a in ALLOWED)

    return [f for f in files if allowed(f) and (REPO / f).is_file()]


def strip_notebook(path: Path) -> None:
    """Remove outputs, execution counts and widget state, keep the code and the text."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.get("metadata", {}).pop("execution", None)
    nb.get("metadata", {}).pop("widgets", None)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def check(target: Path, files: list[str]) -> list[str]:
    problems = []
    for rel in files:
        path = target / rel
        if rel in EXPECTED_EXCEPTIONS:
            continue
        if any(fnmatch.fnmatch(rel, pattern) for pattern in FORBIDDEN_FILES):
            problems.append(f"forbidden file type: {rel}")
        if path.stat().st_size > MAX_SIZE:
            problems.append(f"large file ({path.stat().st_size:,} bytes): {rel}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"binary file: {rel}")
            continue
        allowed = PRIVATE_EXCEPTIONS.get(rel)
        for pattern in PRIVATE_PATTERNS:
            for match in re.finditer(pattern, text):
                if allowed and re.fullmatch(allowed, match.group(0)):
                    continue
                line = text.count("\n", 0, match.start()) + 1
                problems.append(f"private string {match.group(0)!r}: {rel}:{line}")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("target", type=Path, help="Folder of the public repository (new or empty).")
    p.add_argument("--list", action="store_true", help="Only list the files to export.")
    p.add_argument("--working-tree", action="store_true",
                   help="Also export untracked (not ignored) files: a check before committing, "
                        "not for the real export.")
    args = p.parse_args(argv)

    if subprocess.run(["git", "status", "--porcelain", "--", *ALLOWED], cwd=REPO,
                      capture_output=True, text=True).stdout.strip():
        print("/!\\ Uncommitted changes under the exported paths: only the committed "
              "(tracked) state of each file is guaranteed; commit first.")
    files = selected(tracked_files(args.working_tree))
    files = [f for f in files if (REPO / f).exists()]       # deleted but not yet committed
    if args.list:
        print("\n".join(files))
        print(f"\n{len(files)} files")
        return 0

    target = args.target.expanduser().resolve()
    if target == REPO or REPO in target.parents:
        raise SystemExit("The target must be outside this repository.")
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{target} is not empty.")
    for rel in files:
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dest)
        if rel.endswith(".ipynb"):
            strip_notebook(dest)

    problems = check(target, files)
    size = sum((target / f).stat().st_size for f in files)
    print(f"{len(files)} files ({size / 1e6:.1f} MB) exported to {target}")
    if problems:
        print(f"\n{len(problems)} problem(s): fix them (or the lists of this script) and "
              "export again.")
        print("\n".join(f"  {p}" for p in problems))
        return 1
    print("\nChecks passed. Create the public history (by hand, after reviewing the files):\n"
          f"  cd {target}\n"
          "  git init -b main && git add -A && git commit -m \"EXTRAITS 0.1.0\"\n"
          "  git remote add origin git@github.com:TomBellivier/EXTRAITS.git\n"
          "  git push -u origin main")
    return 0


if __name__ == "__main__":
    sys.exit(main())
