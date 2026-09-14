"""
Export executed Jupyter notebooks (with outputs) to print-ready A4 PDFs.

Pipeline per notebook:
    .ipynb --(nbconvert)--> HTML --(headless Edge/Chrome --print-to-pdf)--> PDF

Usage:
    python scripts/export_pdf.py                 # all notebooks
    python scripts/export_pdf.py 01_clustering_algorithms.ipynb 04_ensemble_learning.ipynb

Outputs:
    pdf/*.pdf                    one PDF per notebook
    pdf/html/*.html              intermediate HTML (also printable from a browser)
    pdf/00_combined_all_notebooks.pdf   every notebook in a single PDF

Notes:
    - A4 portrait with minimal margins (5-6 mm), defined via CSS @page in the injected stylesheet.
    - Math (LaTeX) is typeset by MathJax loaded in the HTML, so the first run needs internet.
    - Code lines wrap instead of being clipped, so nothing is cut off on paper.
"""

import argparse
import pathlib
import re
import subprocess
import sys

from nbconvert import HTMLExporter

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "pdf"
HTML_DIR = OUT_DIR / "html"

BROWSERS = [
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

# --- A4 print stylesheet injected into every exported HTML -------------------
PRINT_CSS = """
<style>
  @page { size: A4 portrait; margin: 5mm 5mm 6mm 5mm; }

  html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body { font-size: 12px !important; }

  /* Wrap long code/output lines instead of clipping them on paper */
  pre, code, .highlight { white-space: pre-wrap !important; word-break: break-word !important; overflow: visible !important; }
  .jp-InputArea-editor pre, .jp-InputArea-editor .highlight, .input_area pre { font-size: 12.5px !important; line-height: 1.2 !important; }
  .jp-OutputArea-output pre, .output_subarea pre { font-size: 12px !important; line-height: 1.2 !important; }
  .jp-OutputArea-output, .output_subarea { overflow: visible !important; max-height: none !important; }

  /* Plots stay whole on a page but are capped in height (no half-page figures) */
  .jp-RenderedImage, .output_png, .jp-OutputArea-output img { page-break-inside: avoid; }
  img, svg { max-width: 100% !important; max-height: 65mm !important; width: auto !important; height: auto !important; }
  /* Text outputs and tables may split across pages to avoid large empty gaps */
  .jp-OutputArea, .jp-OutputArea-child, .output_area, .output_subarea, table { page-break-inside: auto; }
  h1, h2, h3, h4, h5 { break-after: avoid-page; page-break-after: avoid; }

  /* Hide execution prompts to save horizontal + vertical space */
  .jp-InputPrompt, .jp-OutputPrompt, .prompt { display: none !important; }

  /* Compact cell spacing and slightly smaller tables */
  .jp-Cell, .cell { margin: 4px 0 !important; padding: 1px 0 !important; }
  .jp-InputArea-editor { border: 1px solid #ddd !important; }
  .jp-RenderedHTMLCommon table { font-size: 11px !important; }

  /* One notebook per page start in the combined document */
  section.notebook { page-break-before: always; }
  section.notebook:first-of-type { page-break-before: auto; }
</style>
"""


def find_browser() -> str:
    for path in BROWSERS:
        if pathlib.Path(path).exists():
            return path
    sys.exit("No Edge/Chrome found. Install one or edit BROWSERS in this script.")


def export_html(ipynb: pathlib.Path) -> str:
    exporter = HTMLExporter()
    exporter.template_name = "lab"
    try:  # make sure markdown-referenced images are embedded too (older/newer API)
        exporter.embed_images = True
    except Exception:
        pass
    body, _ = exporter.from_filename(str(ipynb))
    if "</head>" not in body:
        sys.exit(f"Unexpected HTML structure for {ipynb.name}")
    return body.replace("</head>", PRINT_CSS + "</head>", 1)


def html_to_pdf(browser: str, html_path: pathlib.Path, pdf_path: pathlib.Path) -> None:
    """Print a local HTML file to PDF with headless Chromium."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    url = html_path.resolve().as_uri()
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=20000",  # wait for MathJax/fonts/plots to finish
        "--no-pdf-header-footer",       # modern flag (no URL/date headers)
        "--print-to-pdf-no-header",     # legacy flag, ignored by new versions
        f"--print-to-pdf={pdf_path.resolve()}",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not pdf_path.exists():
        sys.exit(f"PDF not produced for {html_path.name}\n{result.stderr[-1500:]}")


def build_combined_html(html_docs: list[tuple[str, str]], combined_path: pathlib.Path) -> None:
    """Join notebook HTML bodies into one document, each starting on a new page."""
    first_name, first_html = html_docs[0]
    sections = []
    for name, html in html_docs:
        body = re.search(r"<body[^>]*>(.*)</body>", html, flags=re.S | re.I)
        inner = body.group(1) if body else html
        sections.append(f'<section class="notebook" data-notebook="{name}">{inner}</section>')
    # lambda replacement: avoids re.sub treating backslashes in notebook content as escapes
    combined = re.sub(
        r"<body[^>]*>.*</body>",
        lambda _m: "<body>" + "\n".join(sections) + "</body>",
        first_html,
        flags=re.S | re.I,
    )
    combined_path.write_text(combined, encoding="utf-8")


def a4_check(pdf_path: pathlib.Path) -> str:
    """Report the MediaBox size so we can confirm A4 (595 x 842 pt)."""
    data = pdf_path.read_bytes()
    m = re.search(rb"/MediaBox\s*\[\s*0(?:\.0+)?\s+0(?:\.0+)?\s+([\d.]+)\s+([\d.]+)", data)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
    if not m:
        return f"{pages} pages | MediaBox not found"
    w, h = float(m.group(1)), float(m.group(2))
    size = "A4" if abs(w - 595) < 3 and abs(h - 842) < 3 else f"{w:.0f}x{h:.0f}pt!"
    return f"{pages:3d} pages | {size}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export notebooks to A4 PDFs.")
    parser.add_argument("notebooks", nargs="*", help="notebook filenames (default: all)")
    parser.add_argument("--combined-only", action="store_true",
                        help="only rebuild the combined PDF from existing HTML files")
    args = parser.parse_args()

    if args.notebooks:
        paths = [NB_DIR / n for n in args.notebooks]
    else:
        paths = sorted(NB_DIR.glob("*.ipynb"))
    if not paths:
        sys.exit("No notebooks found.")

    browser = find_browser()
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    html_docs: list[tuple[str, str]] = []
    if args.combined_only:
        # Reuse HTML files produced by a previous run
        for ipynb in paths:
            html_path = HTML_DIR / (ipynb.stem + ".html")
            if not html_path.exists():
                sys.exit(f"Missing {html_path.name}; run without --combined-only first.")
            print(f"[html] {html_path.name} (reused)")
            html_docs.append((ipynb.stem, html_path.read_text(encoding="utf-8")))
    else:
        for ipynb in paths:
            print(f"[html] {ipynb.name}")
            html = export_html(ipynb)
            html_path = HTML_DIR / (ipynb.stem + ".html")
            html_path.write_text(html, encoding="utf-8")
            html_docs.append((ipynb.stem, html))

        for ipynb, (_, html) in zip(paths, html_docs):
            pdf_path = OUT_DIR / (ipynb.stem + ".pdf")
            print(f"[pdf ] {pdf_path.name}")
            html_to_pdf(browser, HTML_DIR / (ipynb.stem + ".html"), pdf_path)

    if len(html_docs) > 1:
        combined_html = HTML_DIR / "00_combined_all_notebooks.html"
        build_combined_html(html_docs, combined_html)
        combined_pdf = OUT_DIR / "00_combined_all_notebooks.pdf"
        print(f"[pdf ] {combined_pdf.name} (combined)")
        html_to_pdf(browser, combined_html, combined_pdf)

    print("\n--- output summary ---")
    for pdf in sorted(OUT_DIR.glob("*.pdf")):
        print(f"{pdf.name:45s} {pdf.stat().st_size/1024/1024:6.2f} MB | {a4_check(pdf)}")


if __name__ == "__main__":
    main()
