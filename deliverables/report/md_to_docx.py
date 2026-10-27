"""Convert report/REPORT.md to a submittable .docx.

Handles what this report actually uses: headings, paragraphs with **bold** /
*italic* / `code`, bullet and numbered lists, pipe tables, images, fenced code
blocks, block quotes and horizontal rules. LaTeX is preserved as plain text in
a monospace font (it is a small fraction of the report and re-typing it in
Word's equation editor is a manual step we flag to the user).
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

ROOT = Path("report")
SRC = ROOT / "REPORT.md"
OUT = ROOT / "EE6483_Group_Report.docx"

INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|\$[^$]+\$)")


GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "theta": "θ", "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "phi": "φ", "psi": "ψ", "omega": "ω", "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω",
}
SYMBOLS = {
    "to": "→", "rightarrow": "→", "times": "×", "cdot": "·", "dots": "…", "ldots": "…",
    "in": "∈", "neq": "≠", "approx": "≈", "propto": "∝", "le": "≤", "ge": "≥", "pm": "±",
    "sum": "Σ", "prod": "Π", "infty": "∞", "partial": "∂", "nabla": "∇",
    "quad": "  ", "qquad": "    ", ",": " ", ";": " ", "!": "", " ": " ",
}
CAL = {"X": "𝒳", "Y": "𝒴", "Z": "𝒵", "L": "ℒ", "D": "𝒟", "N": "𝒩"}
BB = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "E": "𝔼", "P": "ℙ"}


def latex_to_unicode(tex: str) -> str:
    r"""Best-effort LaTeX -> Unicode for the constructs this report uses.

    Not a general converter: it handles greek letters, common symbols,
    \frac, \mathcal/\mathbb, \text/\mathrm, \hat and sub/superscript
    braces. Anything it does not recognise is passed through with the
    backslashes removed, which is always readable even when imperfect.
    """
    s = tex.strip()

    # \frac{A}{B} -> A/B  (repeat until none left, innermost first)
    def frac(m):
        return f"{m.group(1)}/{m.group(2)} "
    for _ in range(6):
        new = re.sub(r"\\frac\{((?:[^{}]|\{[^{}]*\})*)\}\{((?:[^{}]|\{[^{}]*\})*)\}", frac, s)
        if new == s:
            break
        s = new

    s = re.sub(r"\\(?:text|mathrm|operatorname)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathcal\{([A-Z])\}", lambda m: CAL.get(m.group(1), m.group(1)), s)
    s = re.sub(r"\\mathbb\{([A-Z])\}", lambda m: BB.get(m.group(1), m.group(1)), s)
    s = re.sub(r"\\hat\{([^{}]*)\}", lambda m: m.group(1) + "\u0302", s)
    s = re.sub(r"\\tilde\{([^{}]*)\}", lambda m: m.group(1) + "\u0303", s)
    s = re.sub(r"\\(log|ln|exp|sin|cos|det)\b", r" \1 ", s)
    s = re.sub(r"\\(?:arg)?(min|max)", r"\1", s)
    s = re.sub(r"\\(?:big|Big|bigg|left|right)([()\[\]|])", r"\1", s)
    s = s.replace("\\{", "{").replace("\\}", "}").replace("\\%", "%")
    s = re.sub(r"\\ell", "ℓ", s)

    def sym(m):
        name = m.group(1)
        if name in GREEK:
            return GREEK[name]
        if name in SYMBOLS:
            return SYMBOLS[name]
        return name
    s = re.sub(r"\\([A-Za-z]+)", sym, s)
    s = re.sub(r"\\[,;!]", " ", s)
    s = s.replace("\\ ", " ")

    # sub/superscript braces -> bare (_{i=1}^{N} -> _i=1^N)
    s = re.sub(r"[_^]\{([^{}]*)\}", lambda m: m.group(0)[0] + m.group(1), s)
    s = s.replace("$", "")
    s = re.sub(r"(?<=[0-9A-Za-z)\]}])(?=Σ)", " ", s)   # 1/BΣ -> 1/B Σ
    s = re.sub(r"\s+([,.;:])", r"\1", s)               # drop space before punctuation
    return re.sub(r"\s{2,}", " ", s).strip()


def add_inline(par, text: str, size: float | None = None, base_bold=False, base_italic=False):
    for tok in INLINE.split(text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = par.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = par.add_run(tok[1:-1]); r.font.name = "Consolas"
            r.font.size = Pt((size or 10.5) - 1)
            r.font.color.rgb = RGBColor(0, 0, 0)
        elif tok.startswith("$") and tok.endswith("$") and len(tok) > 2:
            r = par.add_run(latex_to_unicode(tok[1:-1]))
            r.italic = True
            r.font.name = "Cambria Math"
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = par.add_run(tok[1:-1]); r.italic = True
        else:
            r = par.add_run(tok)
        if size:
            r.font.size = Pt(size) if not (tok.startswith("`")) else r.font.size
        if base_bold:
            r.bold = True
        if base_italic:
            r.italic = True


def flush_table(doc, rows: list[str]):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [c for c in cells if not all(re.fullmatch(r":?-{2,}:?", x or "-") for x in c)]
    if not cells:
        return
    ncol = max(len(c) for c in cells)
    t = doc.add_table(rows=0, cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    # wide tables need a smaller face to fit A4 portrait
    fsize = 8.5 if ncol <= 6 else (7.5 if ncol <= 8 else 6.5)
    for i, row in enumerate(cells):
        cells_row = t.add_row().cells
        for j in range(ncol):
            txt = row[j] if j < len(row) else ""
            p = cells_row[j].paragraphs[0]
            add_inline(p, txt, size=fsize, base_bold=(i == 0))
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
    doc.add_paragraph()


def add_page_footer(doc, text: str):
    """Footer with 'text | page N' using a Word PAGE field."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    for section in doc.sections:
        par = section.footer.paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = par.add_run(text + "   |   page ")
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

        fld_begin = OxmlElement("w:fldChar"); fld_begin.set(qn("w:fldCharType"), "begin")
        instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve"); instr.text = "PAGE"
        fld_end = OxmlElement("w:fldChar"); fld_end.set(qn("w:fldCharType"), "end")
        r = par.add_run(); r.font.size = Pt(8)
        r._r.append(fld_begin); r._r.append(instr); r._r.append(fld_end)


def main():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    doc = Document()
    # ---- academic defaults: Times New Roman, justified, black headings
    from docx.oxml.ns import qn

    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    pf = st.paragraph_format
    pf.space_after = Pt(6)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.15
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for name, size, bold, italic, before, after in [
        ("Heading 1", 14, True, False, 14, 6),
        ("Heading 2", 12, True, False, 12, 4),
        ("Heading 3", 11, True, True, 10, 3),
        ("Heading 4", 11, False, True, 8, 2),
    ]:
        hs = doc.styles[name]
        hs.font.name = "Times New Roman"
        hs.font.size = Pt(size)
        hs.font.bold = bold
        hs.font.italic = italic
        hs.font.color.rgb = RGBColor(0, 0, 0)
        hs.element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
        hs.paragraph_format.space_before = Pt(before)
        hs.paragraph_format.space_after = Pt(after)
        hs.paragraph_format.keep_with_next = True
        hs.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    for s in doc.sections:
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)   # A4
        s.left_margin = s.right_margin = Cm(2.5)
        s.top_margin = s.bottom_margin = Cm(2.5)
    cp = doc.core_properties
    cp.title = "EE6483 Mini Project — Dogs vs Cats (Option 2)"
    cp.subject = "Binary image classification: transfer learning, ablation, CIFAR-10, class imbalance"
    cp.author = "EE6483 project group"
    cp.keywords = "ResNet, transfer learning, Grad-CAM, long-tailed, CIFAR-10"

    i, n = 0, len(lines)
    in_code = False
    code_buf: list[str] = []
    table_buf: list[str] = []

    def flush_code():
        nonlocal code_buf
        if not code_buf:
            return
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.4)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run("\n".join(code_buf))
        r.font.name = "Consolas"
        r.font.size = Pt(8)
        code_buf = []

    while i < n:
        ln = lines[i]

        if ln.strip().startswith("```"):
            if in_code:
                flush_code(); in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(ln); i += 1; continue

        if ln.strip().startswith("|") and ln.strip().endswith("|"):
            table_buf.append(ln); i += 1
            if i < n and not (lines[i].strip().startswith("|")):
                flush_table(doc, table_buf); table_buf = []
            continue
        if table_buf:
            flush_table(doc, table_buf); table_buf = []

        s = ln.strip()
        if not s or s.startswith("<div") or s == "---":
            i += 1
            continue

        if s.startswith("#### "):
            h = doc.add_heading(s[5:], level=4)
        elif s.startswith("### "):
            h = doc.add_heading(s[4:], level=3)
        elif s.startswith("## "):
            h = doc.add_heading(s[3:], level=2)
        elif s.startswith("# "):
            h = doc.add_heading(s[2:], level=1)
        elif s.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
            if m:
                p = (ROOT / m.group(2)).resolve()
                if p.exists():
                    # insert at the figure's natural size (300 dpi), capped to the
                    # 16 cm text column, so small figures are never blown up
                    from PIL import Image as _Img

                    with _Img.open(p) as _im:
                        natural_cm = _im.size[0] / 300 * 2.54
                    doc.add_picture(str(p), width=Cm(min(15.5, natural_cm)))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    if m.group(1):
                        cap = doc.add_paragraph()
                        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        r = cap.add_run(m.group(1))
                        r.italic = True
                        r.font.size = Pt(9)
                        r.font.name = "Times New Roman"
        elif s.startswith("$$") and s.endswith("$$") and len(s) > 4:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
            r = p.add_run(latex_to_unicode(s[2:-2]))
            r.italic = True
            r.font.name = "Cambria Math"
            r.font.size = Pt(11)
        elif s.startswith("> "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            add_inline(p, s[2:], size=9.5, base_italic=True)
        elif re.match(r"^[-*] ", s):
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, s[2:])
        elif re.match(r"^\d+\. ", s):
            # Keep the number from the source text instead of using Word's
            # "List Number" style: all lists would otherwise share one counter and
            # continue numbering across sections (1..10 in the contents, then 11, 12 ...
            # in the first list of section 1).
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.75)
            add_inline(p, s)
        else:
            p = doc.add_paragraph()
            add_inline(p, s)
        i += 1

    if code_buf:
        flush_code()
    if table_buf:
        flush_table(doc, table_buf)

    add_page_footer(doc, "EE6483 Mini Project — Dogs vs Cats")
    doc.save(OUT)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(doc.paragraphs)} paragraphs)")


if __name__ == "__main__":
    main()
