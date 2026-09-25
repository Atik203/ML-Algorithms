"""
Build the completed laboratory manual as a typeset PDF with XeLaTeX.

Pipeline:
    ML_Lab_Manual_Completed.md --(pandoc + custom preamble)--> XeLaTeX --> PDF

Usage:
    python scripts/build_manual_pdf.py

Outputs:
    Machine_Learning_Lab_Manual_Completed.pdf

Notes:
    - The Markdown file is the single source of truth; edit it and re-run this script.
    - The DOCX edition is built separately with:
        pandoc ML_Lab_Manual_Completed.md -o Machine_Learning_Lab_Manual_Completed.docx \
               --reference-doc=Machine_Learning_Lab_Manual.docx
    - Requires Pandoc + TeX Live (xelatex) and the DejaVu fonts.
"""

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
MD = ROOT / "ML_Lab_Manual_Completed.md"
PREAMBLE = ROOT / "scripts" / "latex" / "preamble.tex"
OUT_PDF = ROOT / "Machine_Learning_Lab_Manual_Completed.pdf"

PANDOC_CANDIDATES = [
    r"C:\Program Files\Pandoc\pandoc.exe",
    "pandoc",
]
XELATEX_CANDIDATES = [
    r"C:\texlive\2025\bin\windows\xelatex.exe",
    "xelatex",
]


def which(candidates: list[str]) -> str:
    for cand in candidates:
        if pathlib.Path(cand).exists():
            return cand
        found = shutil.which(cand)
        if found:
            return found
    sys.exit(f"Not found: {candidates}")


def strip_yaml(text: str) -> str:
    """Drop the YAML metadata block (the Markdown cover acts as the title page)."""
    return re.sub(r"(?s)\A---\n.*?\n---\n", "", text, count=1)


def pdf_pages(pdf_path: pathlib.Path) -> int:
    data = pdf_path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


def main() -> None:
    pandoc = which(PANDOC_CANDIDATES)
    xelatex = which(XELATEX_CANDIDATES)

    work = pathlib.Path(tempfile.mkdtemp(prefix="manual_tex_"))
    md_path = work / "manual.md"
    md_path.write_text(strip_yaml(MD.read_text(encoding="utf-8")), encoding="utf-8")

    cmd = [
        pandoc,
        str(md_path),
        "-o", str(OUT_PDF),
        "--pdf-engine", xelatex,
        "--toc", "--toc-depth=1",
        "--highlight-style=tango",
        "-V", "mainfont=DejaVu Serif",
        "-V", "monofont=DejaVu Sans Mono",
        "-V", "fontsize=10pt",
        "-V", "geometry:a4paper",
        "--include-in-header", str(PREAMBLE),
    ]
    print("running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not OUT_PDF.exists():
        print(result.stdout[-2000:])
        print(result.stderr[-3000:])
        sys.exit("PDF build failed")

    # surface overfull/unicode warnings that could clip content
    warnings = [ln for ln in result.stderr.splitlines() if "Overfull" in ln or "Missing character" in ln]
    if warnings:
        print(f"layout warnings: {len(warnings)} (first 5 shown)")
        for w in warnings[:5]:
            print("  ", w.strip()[:160])

    print(f"built: {OUT_PDF.name} | {OUT_PDF.stat().st_size/1e6:.2f} MB | {pdf_pages(OUT_PDF)} pages")


if __name__ == "__main__":
    main()
