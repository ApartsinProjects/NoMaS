# -*- coding: utf-8 -*-
"""Build the elsarticle two-column LaTeX PDF for ADRank (TwoColPaper route A).

Runs html2tex (convert --columns 2 -> pack elsarticle), then grafts the
paper-specific front matter onto _tex/main.tex: abstract into the frontmatter,
real author/affiliation block with corresponding author, journal name,
full-width promotion of the two wide figures, journal back-matter conventions,
and Unicode/badge cleanup. The bibliography is built by the converter itself
(the div.references fix now landed in convert_to_tex.py). Output:
_tex/main.pdf -> docs/adrank-2col-latex.pdf.
"""
import os, re, io, subprocess, sys, shutil

ROOT = r"E:\Projects\Submitted\ADRank"
SKILL = r"C:\Users\apart\.claude\skills\html2tex"
PY = sys.executable

UNI = {'\u2208': r'$\in$', '\u2193': '', '\u00d7': r'$\times$'}


def sh(cmd):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def uni(s):
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s


def main():
    # start clean: stale _tex artifacts (e.g. a previous paper's appendix.tex) get grafted by pack
    # even when the current HTML has no appendix. Always rebuild _tex from scratch.
    shutil.rmtree(os.path.join(ROOT, "_tex"), ignore_errors=True)
    sh([PY, os.path.join(SKILL, "scripts", "convert_to_tex.py"),
        "--input", "docs/index.html", "--out-dir", "_tex", "--columns", "2"])
    sh([PY, os.path.join(SKILL, "scripts", "pack_tmlr_bundle.py"),
        "--in-dir", "_tex", "--template", "elsarticle"])

    main_p = os.path.join(ROOT, "_tex", "main.tex")
    tex = io.open(main_p, encoding="utf-8").read()

    # carve the abstract out of the body (house-style div.abstract comes through
    # as a \section{Abstract}); re-insert it into elsarticle's frontmatter.
    m = re.search(r"\\section\{Abstract\}\\label\{abstract\}\s*(.*?)\s*\\section\{Introduction\}", tex, re.S)
    abstract = uni(m.group(1).strip())
    assert len(abstract) > 500, "abstract extraction failed"

    # drop the pre-Introduction body cruft (web badges, HTML byline, the
    # \section{Abstract} block) by splicing frontmatter straight to Introduction.
    fm_end = tex.index("\\end{frontmatter}") + len("\\end{frontmatter}")
    intro = tex.index("\\section{Introduction}")
    tex = tex[:fm_end] + "\n\n" + tex[intro:]

    # the converter's thebibliography prints its own heading; remove the phantom
    # \section{References} left by the <h2>References</h2> whose <div> was extracted.
    tex = re.sub(r"\\section\{References\}\\label\{references\}\n?", "", tex)
    # journals NUMBER Limitations (converter stars it per the ACL convention).
    tex = tex.replace("\\section*{Limitations}", "\\section{Limitations}")
    # availability is unnumbered back matter, not a numbered section.
    tex = tex.replace("\\section{Data and code availability}",
                      "\\section*{Data and code availability}")
    # strip any leaked download-badge hrefs; map stray unicode
    tex = re.sub(r"\\href\{nomas[^}]*\}\{[^}]*\}", "", tex)
    # strip the HTML page footer ("ADReal $\cdot$ <title>") that the converter pulls into the body
    tex = re.sub(r"ADReal \$\\cdot\$ [^\n]*\n?", "", tex)

    # de-float the short data tables (markdown pipe tables -> tabularx). Their captions live in
    # the preceding "\textbf{Table N.}" paragraph, so as floats the grid drifts away from the
    # caption. Rendering the tabularx inline keeps each grid directly under its caption. Figures
    # keep their float (\begin{figure}); only \begin{table} wrappers are removed.
    tex = re.sub(r"\\begin\{table\}(\[[^\]]*\])?\s*(\\centering\s*)?", "\n\\\\vspace{2pt}\\\\noindent\n", tex)
    tex = tex.replace("\\end{table}", "\n\\vspace{2pt}\n")
    tex = uni(tex)

    # make long bibliography URLs breakable: xurl breaks anywhere, and inside
    # thebibliography normalize \href{U}{T} -> \url{U} and wrap bare URLs, so a
    # long arXiv/DOI/proceedings link wraps instead of running into the margin.
    tex = tex.replace("\\usepackage{newtxmath}", "\\usepackage{newtxmath}\n\\usepackage{xurl}")
    def _bib_urls(mo):
        seg = re.sub(r"\\href\{([^}]*)\}\{[^}]*\}", r"\\url{\1}", mo.group(0))
        seg = re.sub(r"(?<![{/])(https?://[^\s{}]+)", r"\\url{\1}", seg)
        return seg
    tex = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", _bib_urls, tex, flags=re.S)

    # promote the two WIDE figures (fig_anisotropy two-panel, fig_leaderboard bars) to full-width figure*
    def promote(mobj):
        blk = mobj.group(0)
        if "figures/fig_" in blk or "\\includegraphics" in blk:  # all figures are wide/multi-panel
            blk = blk.replace(r"\begin{figure}[tbp]", r"\begin{figure*}[t]").replace(r"\end{figure}", r"\end{figure*}")
        return blk
    tex = re.sub(r"\\begin\{figure\}\[tbp\].*?\\end\{figure\}", promote, tex, flags=re.S)

    # frontmatter grafting: abstract, real author block (Aperstein corresponding),
    # journal, and a tighter top margin above the title.
    tex = tex.replace("\\begin{abstract}\n\n\\end{abstract}",
                      "\\begin{abstract}\n" + abstract + "\n\\end{abstract}")
    tex = tex.replace(
        "\\author{Anonymous Authors}\n\\address{Anonymous Affiliations}",
        "\\author[hit]{Alexander Apartsin}\n"
        "\\author[afeka]{Yehudit Aperstein\\corref{cor1}}\n"
        "\\ead{apersteiny@afeka.ac.il}\n"
        "\\cortext[cor1]{Corresponding author}\n"
        "\\address[hit]{School of Computer Science, Faculty of Sciences, Holon Institute of Technology (HIT), Holon, Israel}\n"
        "\\address[afeka]{Intelligent Systems, Afeka Academic College of Engineering, Tel-Aviv, Israel}")
    tex = tex.replace("__JOURNAL__", "Neurocomputing")
    NEWTITLE = "No Easy Wins: A Contamination-Controlled Benchmark for Evaluating Anomaly Detection and Model Selection"
    tex = tex.replace("\\title{" + NEWTITLE + "}",
                      "\\title{\\vspace*{-2\\baselineskip}" + NEWTITLE + "}")

    io.open(main_p, "w", encoding="utf-8").write(tex)
    print(f"grafted: abstract {len(abstract)}c, figure* = {tex.count(chr(92)+'begin{figure*}')}, "
          f"bibitems = {tex.count(chr(92)+'bibitem')}")

    sh([PY, os.path.join(SKILL, "scripts", "compile_local.py"), "--in-dir", "_tex", "--auto-patch"])
    shutil.copy(os.path.join(ROOT, "_tex", "main.pdf"), os.path.join(ROOT, "docs", "nomas-2col-latex.pdf"))
    shutil.copy(main_p, os.path.join(ROOT, "paper", "latex", "main.tex"))
    print("done -> docs/nomas-2col-latex.pdf")


if __name__ == "__main__":
    main()
