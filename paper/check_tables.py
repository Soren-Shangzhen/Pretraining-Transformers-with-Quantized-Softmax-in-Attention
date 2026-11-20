#!/usr/bin/env python3
"""Compare the regenerated table files with those of the paper's source.

paper/reference/tables/ holds the table files of the arXiv source of the paper (the 50 files it \\input's, among them
rb_numbers.tex with the numbers printed in the text) and the tabulars of the four tables of Appendix J, which the
source holds inline. paper/tables/ is written by make_paper_tables.py, make_rb_tables.py and make_binmean_tables.py.
Files are compared byte for byte; rb_numbers.tex is compared without its TeX comments (the generator writes the source
row of each number as a comment, and the arXiv source carries no comments).

Exit status 0 if every file is identical, 1 otherwise.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF, GEN = HERE / "reference" / "tables", HERE / "tables"


def strip_comments(s):
    out = []
    for line in s.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        if m is None:
            out.append(line)
        elif line[:m.start()].strip() == "":
            continue                      # comment-only line
        else:
            out.append(line[:m.start()] + "%")
    return "\n".join(out)


def main():
    ref = sorted(REF.glob("*.tex"))
    bad = []
    for r in ref:
        g = GEN / r.name
        if not g.exists():
            bad.append((r.name, "not generated"))
            continue
        a, b = r.read_bytes(), g.read_bytes()
        if r.name == "rb_numbers.tex":
            a, b = a.decode(), strip_comments(b.decode())
        if a != b:
            bad.append((r.name, "differs"))
    extra = sorted(p.name for p in GEN.glob("*.tex") if not (REF / p.name).exists())
    for name, why in bad:
        print(f"MISMATCH {name}: {why}")
    for name in extra:
        print(f"note: {name} is generated but is not a table file of the paper")
    print(f"{len(ref) - len(bad)}/{len(ref)} table files identical to the paper's source")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
