#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def git_output(args: list[str]) -> list[Path]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def changed_staged_tex() -> list[Path]:
    return git_output(["diff", "--cached", "--name-only", "--diff-filter=ACMR", "--", "*.tex"])


def changed_tex_in_range(old_rev: str, new_rev: str) -> list[Path]:
    return git_output(["diff", "--name-only", old_rev, new_rev, "--", "*.tex"])


def changed_worktree_tex() -> list[Path]:
    return git_output(["diff", "--name-only", "--", "*.tex"])


def is_standalone_tex(path: Path) -> bool:
    if not path.exists() or path.suffix != ".tex":
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "\\documentclass" in text


def needs_bibtex(aux_path: Path) -> bool:
    if not aux_path.exists():
        return False
    try:
        text = aux_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "\\bibdata" in text


def compile_tex(tex_path: Path) -> Path:
    tex_path = tex_path.resolve()
    workdir = tex_path.parent
    tex_name = tex_path.name
    stem = tex_path.stem
    aux_path = workdir / f"{stem}.aux"
    pdf_path = workdir / f"{stem}.pdf"

    print(f"[tex-build] compiling {tex_path.relative_to(ROOT)}", flush=True)
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name], workdir)

    if needs_bibtex(aux_path):
        run(["bibtex", stem], workdir)
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name], workdir)
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name], workdir)
    else:
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name], workdir)

    return pdf_path


def normalize_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    normalized: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if is_standalone_tex(resolved):
            seen.add(resolved)
            normalized.append(resolved)
    return normalized


def stage_paths(paths: list[Path]) -> None:
    if not paths:
        return
    rel_paths = [str(path.relative_to(ROOT)) for path in paths if path.exists()]
    if rel_paths:
        run(["git", "add", *rel_paths], ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="编译变动的独立 LaTeX 文件，并按需 stage 对应 PDF。"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="编译已 stage 的 .tex 文件")
    mode.add_argument(
        "--range",
        nargs=2,
        metavar=("OLD_REV", "NEW_REV"),
        help="编译两个 git revision 之间变化的 .tex 文件",
    )
    mode.add_argument("--worktree", action="store_true", help="编译工作区变化的 .tex 文件")
    mode.add_argument(
        "--files",
        nargs="+",
        metavar="TEX",
        help="直接编译指定的 .tex 文件",
    )
    parser.add_argument(
        "--stage-pdfs",
        action="store_true",
        help="编译成功后将对应 PDF git add",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.staged:
        tex_paths = changed_staged_tex()
    elif args.range:
        tex_paths = changed_tex_in_range(args.range[0], args.range[1])
    elif args.worktree:
        tex_paths = changed_worktree_tex()
    else:
        tex_paths = [ROOT / path for path in args.files]

    standalone_paths = normalize_paths(tex_paths)
    if not standalone_paths:
        print("[tex-build] no changed standalone .tex files detected", flush=True)
        return

    built_pdfs: list[Path] = []
    for tex_path in standalone_paths:
        built_pdfs.append(compile_tex(tex_path))

    if args.stage_pdfs:
        stage_paths(built_pdfs)
        print("[tex-build] staged generated pdf files", flush=True)


if __name__ == "__main__":
    main()
