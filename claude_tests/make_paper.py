#!/usr/bin/env python3
"""make_paper.py -- build docs/paper/paper.pdf (and paper.tex) from docs/WRITEUP.md with pandoc + tectonic. Inserts figures after their 'Figure N' paragraphs,
drops the repository-only checklist (section 8) and the draft banner. Needs: pandoc, tectonic (brew install pandoc tectonic); figures from make_figures.py.
    python make_paper.py"""
import re
import subprocess
from pathlib import Path

D = Path("docs"); OUT = D / "paper"; OUT.mkdir(exist_ok=True)
t = (D / "WRITEUP.md").read_text()
t = t.split("\n## 8. Open items before release")[0]                      # release checklist is not part of the paper
lines = t.split("\n"); title = lines[0].lstrip("# ").strip(); body = "\n".join(lines[1:])
body = re.sub(r"^\*Draft v[^\n]*\n", "", body, count=1, flags=re.M)
fig = {"1": "fig1_forest", "2": "fig2_learning_curve", "3": "fig3_neutral_false_alarm"}
def add_fig(m):
    n = m.group(1); return m.group(0) + f"\n\n![](../figures/{fig[n]}.pdf)\n"
body = re.sub(r"^\*\*Figure (\d)\*\*[^\n]*$", add_fig, body, flags=re.M)
(OUT / "paper.md").write_text(f"---\ntitle: \"{title}\"\nauthor: \"[AUTHOR NAME]\"\ndate: \"2026\"\n---\n\n" + body)
common = ["pandoc", "paper.md", "--from=gfm+tex_math_dollars-yaml_metadata_block+yaml_metadata_block", "-V", "geometry:margin=1in", "-V", "fontsize=10pt",
          "-V", "mainfont=Times New Roman", "-V", "monofont=Menlo", "-V", "colorlinks=true", "--toc", "--toc-depth=2"]
subprocess.run(common + ["--pdf-engine=tectonic", "-o", "paper.pdf"], check=True, cwd=OUT, stderr=subprocess.DEVNULL)
subprocess.run(common + ["-s", "-o", "paper.tex"], check=True, cwd=OUT)
print("wrote", OUT / "paper.pdf", OUT / "paper.tex")
