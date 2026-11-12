from pathlib import Path
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "REPORT_BLUEPRINT.md"
OUTPUT = ROOT / "final" / "FRAM_Biocon_Report_Blueprint.docx"


def shade(cell, color):
    tc_pr = cell._tc.get_or_add_tcPr()
    fill = tc_pr.find(qn("w:shd"))
    if fill is None:
        fill = OxmlElement("w:shd")
        tc_pr.append(fill)
    fill.set(qn("w:fill"), color)


def set_cell_border(cell, color="D9D9D9", size="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=110, start=110, bottom=110, end=110):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn("w:" + name))
        if node is None:
            node = OxmlElement("w:" + name)
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    run.font.size = Pt(8)
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr, fld_char2])


def clean_inline(text):
    text = text.replace("`", "")
    text = text.replace("**", "")
    text = text.replace("$", "")
    text = text.replace("sigma_E", "σE").replace("sigma_V", "σV")
    text = text.replace("<=", "≤").replace(">=", "≥")
    return text


def parse_table(lines):
    rows = []
    for line in lines:
        cells = [clean_inline(cell.strip()) for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) > 1 and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in rows[1]):
        rows.pop(1)
    return rows


def add_table(doc, rows):
    if not rows:
        return
    cols = max(len(row) for row in rows)
    table = doc.add_table(rows=len(rows), cols=cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for r_idx, values in enumerate(rows):
        row = table.rows[r_idx]
        if r_idx == 0:
            repeat_header(row)
        for c_idx in range(cols):
            cell = row.cells[c_idx]
            value = values[c_idx] if c_idx < len(values) else ""
            cell.text = value
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_border(cell)
            set_cell_margins(cell)
            if r_idx == 0:
                shade(cell, "1F4E78")
                for run in cell.paragraphs[0].runs:
                    run.font.color.rgb = RGBColor(255, 255, 255)
                    run.bold = True
                    run.font.size = Pt(8.5)
            else:
                if r_idx % 2 == 0:
                    shade(cell, "EAF2F8")
                for run in cell.paragraphs[0].runs:
                    run.font.size = Pt(8.3)
            cell.paragraphs[0].paragraph_format.space_after = Pt(0)
            cell.paragraphs[0].paragraph_format.line_spacing = 1.0
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_rich_paragraph(doc, text, style=None):
    paragraph = doc.add_paragraph(style=style)
    text = clean_inline(text)
    paragraph.add_run(text)
    return paragraph


def configure_document(doc):
    section = doc.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.7)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = RGBColor(31, 31, 31)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08

    title = doc.styles["Title"]
    title.font.name = "Aptos Display"
    title.font.size = Pt(24)
    title.font.bold = True
    title.font.color.rgb = RGBColor(0, 0, 0)
    title.paragraph_format.space_after = Pt(10)

    heading_sizes = {"Heading 1": 16, "Heading 2": 13, "Heading 3": 11}
    for name, size in heading_sizes.items():
        style = doc.styles[name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(10 if name != "Heading 1" else 14)
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.keep_with_next = True

    header = section.header.paragraphs[0]
    header.text = "FRAM Biocon Technical Report Blueprint"
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.name = "Aptos"
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(90, 90, 90)
    add_page_number(section.footer.paragraphs[0])


def build():
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    doc = Document()
    configure_document(doc)

    title = doc.add_paragraph(style="Title")
    title.add_run("FRAM Biocon Technical Report Blueprint")
    subtitle = doc.add_paragraph()
    subtitle.add_run("A structured plan for the ten page technical report and supporting appendices").italic = True
    subtitle.paragraph_format.space_after = Pt(12)
    intro = doc.add_paragraph(
        "This document converts the assignment requirements and completed Biocon analysis into a practical report plan. "
        "It identifies the required content, exhibits, interpretation and evidence for each section while keeping the main report within the ten page limit."
    )
    intro.paragraph_format.space_after = Pt(14)

    i = 1  # skip markdown title
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            add_table(doc, parse_table(table_lines))
            continue
        if stripped.startswith("### "):
            doc.add_heading(clean_inline(stripped[4:]), level=3)
        elif stripped.startswith("## "):
            doc.add_heading(clean_inline(stripped[3:]), level=1)
        elif stripped.startswith("# "):
            doc.add_heading(clean_inline(stripped[2:]), level=1)
        elif re.match(r"^\d+\.\s+", stripped):
            add_rich_paragraph(doc, re.sub(r"^\d+\.\s+", "", stripped), style="List Number")
        elif stripped.startswith("- [ ] "):
            add_rich_paragraph(doc, "☐ " + stripped[6:])
        elif stripped.startswith("- "):
            add_rich_paragraph(doc, stripped[2:], style="List Bullet")
        else:
            add_rich_paragraph(doc, stripped)
        i += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.core_properties.title = "FRAM Biocon Technical Report Blueprint"
    doc.core_properties.subject = "Report structure, required exhibits and submission checklist"
    doc.core_properties.author = ""
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
