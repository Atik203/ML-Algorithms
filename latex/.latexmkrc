# ===========================================================================
# .latexmkrc - build configuration for the Machine Learning Laboratory Manual
#
# Local build:     latexmk main.tex      (uses XeLaTeX automatically)
# Clean:           latexmk -c            (remove aux files)
#                  latexmk -C            (also remove the PDF)
#
# Overleaf:        honored automatically (XeLaTeX build).
#
# NOTE FOR EDITORS: many LaTeX editors invoke latexmk with "-pdf", which selects
# the pdflatex rule and would ignore the XeLaTeX setting below. To keep the
# "Build" button working unchanged, the pdflatex rule is redirected to XeLaTeX.
# ===========================================================================

# 5 = XeLaTeX (required by fontspec and the Unicode characters in the manual)
$pdf_mode = 5;
$xelatex = 'xelatex -interaction=nonstopmode -halt-on-error -synctex=1 %O %S';

# Redirect the "-pdf" rule (pdflatex) to XeLaTeX as well
$pdflatex = 'xelatex -interaction=nonstopmode -halt-on-error -synctex=1 %O %S';

# No bibliography or index in this manual
$bibtex_use = 0;

# Enough passes for the table of contents and cross-references
$max_repeat = 5;

# Files removed by "latexmk -c"
$clean_ext = 'synctex.gz fdb_latexmk fls';
