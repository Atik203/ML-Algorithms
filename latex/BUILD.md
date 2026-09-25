# Building the Manual

The project is a standard LaTeX project and builds with **XeLaTeX** (required by
`fontspec` and the Unicode characters used in the manual). The included
`.latexmkrc` selects XeLaTeX automatically.

## Local build (TeX Live)

```bash
cd latex
latexmk main.tex           # automatic multi-pass XeLaTeX build -> main.pdf
latexmk -c                 # remove auxiliary files
latexmk -C                 # remove auxiliary files and main.pdf
```

`latexmk -pvc main.tex` rebuilds automatically whenever a file is saved.

> Do **not** pass `-pdf`: that flag forces pdfLaTeX and overrides the engine chosen
> in `.latexmkrc`. Use `latexmk -xelatex main.tex` if you prefer explicit flags.

## Overleaf

1. Upload the project folder (or zip) with `main.tex` at the top level.
2. The bundled `.latexmkrc` makes Overleaf compile with XeLaTeX automatically.
   (Menu -> Compiler can stay on the default; if in doubt select **XeLaTeX**.)
3. Press **Recompile**.

## Manual build without latexmk

```bash
xelatex main.tex
xelatex main.tex      # second pass for the table of contents
```

## Project layout

```
main.tex                 master file (title page, contents, parts)
preamble.tex             packages and rich styling
.latexmkrc               XeLaTeX build configuration
chapters/                one file per category (00 front matter + 01-12)
code/                    complete notebook code listings (\input by the chapters)
figures/                 result plots from the executed notebooks
```
