#!/usr/bin/env bash
# Regenerate every generated table, figure and in-text number of the paper from the stored result files (CPU, a few
# minutes), then compare the tables with those of the paper's source.  Usage: bash paper/make_all.sh [PYTHON]
set -euo pipefail
PY=${1:-python}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE/tables"
"$PY" make_paper_tables.py      # main text and Appendices A-H
"$PY" make_rb_tables.py         # Appendix I tables and rb_numbers.tex
"$PY" make_binmean_tables.py    # Appendix J
cd "$HERE/figures"
"$PY" make_paper_figures.py
"$PY" make_rb_figures.py
cd "$HERE"
"$PY" check_tables.py
