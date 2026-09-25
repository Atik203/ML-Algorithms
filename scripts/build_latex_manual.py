"""
Build the native-LaTeX edition of the completed laboratory manual.

Structure created under latex/:
    main.tex                        master file (title page, TOC, preface, inputs)
    preamble.tex                    shared style (fonts, listings, tables, headers)
    sections/01_clustering.tex      Experiments 1-4
    sections/02_density_based.tex   Experiments 5-6
    sections/03_semi_supervised.tex Experiment 7
    sections/04_ensemble.tex        Experiments 8-12
    sections/05_mlp.tex             Experiment 13
    sections/06_rnn.tex             Experiment 14
    sections/07_som.tex             Experiment 15
    sections/08_hmm.tex             Experiment 16
    sections/09_svm.tex             Experiment 17
    sections/10_llm.tex             Experiment 18
    sections/11_grnn.tex            Experiment 19
    sections/12_mini_project_appendices.tex  Mini project + Appendices A-D

Pipeline:
    ML_Lab_Manual_Completed.md --(pandoc fragments, --listings)--> sections/*.tex
    xelatex (twice) --> latex/main.pdf --> Machine_Learning_Lab_Manual_Completed.pdf

Usage:  python scripts/build_latex_manual.py
"""

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
MD = ROOT / "ML_Lab_Manual_Completed.md"
LATEX_DIR = ROOT / "latex"
SECTIONS = LATEX_DIR / "sections"
OUT_PDF = ROOT / "Machine_Learning_Lab_Manual_Completed.pdf"

PANDOC_CANDIDATES = [r"C:\Program Files\Pandoc\pandoc.exe", "pandoc"]
XELATEX_CANDIDATES = [r"C:\texlive\2025\bin\windows\xelatex.exe", "xelatex"]

CATEGORIES = [
    ("01_clustering", "Clustering", lambda t: re.match(r"Experiment ([1-4]):", t)),
    ("02_density_based", "Density-Based Learning", lambda t: re.match(r"Experiment ([5-6]):", t)),
    ("03_semi_supervised", "Semi-Supervised Learning", lambda t: re.match(r"Experiment 7:", t)),
    ("04_ensemble", "Ensemble Learning", lambda t: re.match(r"Experiment (8|9|10|11|12):", t)),
    ("05_mlp", "Multilayer Perceptron", lambda t: re.match(r"Experiment 13:", t)),
    ("06_rnn", "Recurrent Neural Network", lambda t: re.match(r"Experiment 14:", t)),
    ("07_som", "Self-Organizing Map", lambda t: re.match(r"Experiment 15:", t)),
    ("08_hmm", "Hidden Markov Model", lambda t: re.match(r"Experiment 16:", t)),
    ("09_svm", "Support Vector Machine", lambda t: re.match(r"Experiment 17:", t)),
    ("10_llm", "Large Language Model", lambda t: re.match(r"Experiment 18:", t)),
    ("11_grnn", "Generalized Regression Neural Network", lambda t: re.match(r"Experiment 19:", t)),
    ("12_mini_project_appendices", "Mini Project and Appendices",
     lambda t: t.startswith("Integrated Mini Project") or t.startswith("Appendix ")),
]


def which(candidates: list[str]) -> str:
    for cand in candidates:
        if pathlib.Path(cand).exists():
            return cand
        found = shutil.which(cand)
        if found:
            return found
    sys.exit(f"Not found: {candidates}")


def split_chunks(md_text: str) -> dict[str, str]:
    """Map each level-1 heading title -> its markdown chunk (fence-aware)."""
    chunks: dict[str, str] = {}
    title = None
    buf: list[str] = []
    in_fence = False
    for line in md_text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and re.match(r"^# ", line):
            if title is not None:
                chunks[title] = "".join(buf)
            title = line[2:].strip()
            buf = [line]
        else:
            buf.append(line)
    if title is not None:
        chunks[title] = "".join(buf)
    return chunks


def to_latex(pandoc: str, markdown: str, tmp_dir: pathlib.Path, top: str = "chapter") -> str:
    src = tmp_dir / "frag.md"
    dst = tmp_dir / "frag.tex"
    src.write_text(markdown, encoding="utf-8")
    cmd = [pandoc, str(src), "-f", "markdown", "-t", "latex",
           "--listings", f"--top-level-division={top}"]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stderr[-2000:])
        sys.exit("pandoc fragment conversion failed")
    return dst.read_text(encoding="utf-8") if dst.exists() else result.stdout


def main() -> None:
    pandoc = which(PANDOC_CANDIDATES)
    xelatex = which(XELATEX_CANDIDATES)
    SECTIONS.mkdir(parents=True, exist_ok=True)

    md_text = MD.read_text(encoding="utf-8")
    chunks = split_chunks(md_text)
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="latex_frag_"))

    used = set()
    for fname, category, match in CATEGORIES:
        selected = [title for title in chunks if match(title)]
        assert selected, f"no chunks matched for {fname}"

        def sort_key(t: str) -> tuple[int, str]:
            m = re.search(r"Experiment (\d+)", t)
            return (int(m.group(1)) if m else 99, t)

        selected.sort(key=sort_key)
        used.update(selected)

        if fname == "12_mini_project_appendices":
            mini = [t for t in selected if t.startswith("Integrated Mini Project")]
            appx = [t for t in selected if t.startswith("Appendix ")]
            body = to_latex(pandoc, "\n\n".join(chunks[t] for t in mini), tmp_dir)
            body += "\n\\appendix\n\n"
            body += to_latex(pandoc, "\n\n".join(chunks[t] for t in appx), tmp_dir)
        else:
            body = to_latex(pandoc, "\n\n".join(chunks[t] for t in selected), tmp_dir)

        header = (f"% ===========================================================================\n"
                  f"% Category: {category}\n"
                  f"% Generated from ML_Lab_Manual_Completed.md - edit the Markdown and re-run\n"
                  f"% scripts/build_latex_manual.py instead of editing this file by hand.\n"
                  f"% ===========================================================================\n\n")
        (SECTIONS / f"{fname}.tex").write_text(header + body, encoding="utf-8")
        print(f"wrote sections/{fname}.tex  ({len(selected)} section(s): {', '.join(selected[:2])}"
              f"{' ...' if len(selected) > 2 else ''})")

    missing = set(chunks) - used - {"Machine Learning Laboratory Manual \u2014 Completed Edition"}
    if missing:
        print("chunks not placed:", sorted(missing)[:5])

    # --- compile -----------------------------------------------------------
    for run in (1, 2):
        result = subprocess.run(
            [xelatex, "-interaction=nonstopmode", "main.tex"],
            cwd=LATEX_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
        pdf = LATEX_DIR / "main.pdf"
        if result.returncode != 0 and not pdf.exists():
            log = (result.stdout or "")[-4000:]
            print(log)
            sys.exit(f"xelatex failed on run {run}")
    shutil.copy2(LATEX_DIR / "main.pdf", OUT_PDF)
    # count pages via pdftotext form feeds (XeTeX PDFs use compressed object streams)
    pdftotext = which([r"C:\texlive\2025\bin\windows\pdftotext.exe", "pdftotext"])
    txt = subprocess.run([pdftotext, str(OUT_PDF), "-"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    pages = txt.count("\f") + (1 if txt else 0)
    warnings = [ln for ln in (result.stdout or "").splitlines() if "Overfull" in ln]
    print(f"\nbuilt: {OUT_PDF.name} | {OUT_PDF.stat().st_size/1e6:.2f} MB | "
          f"{pages} pages | overfull boxes: {len(warnings)}")


if __name__ == "__main__":
    main()
