
import os
import re
import base64
import subprocess
from io import BytesIO
from typing import List, Dict
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from .config import logger
from .analysis_cls import ClassifiedSegment

def latex_escape(text: str) -> str:
    replacements = {"\\": r"\\", "{": r"\{", "}": r"\}", "$": r"\$","&": r"\&", "#": r"\#", "_": r"\_", "%": r"\%","~": r"\textasciitilde{}", "^": r"\^{}"}
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text

def _strip_latex_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:latex)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    return s.strip()

def validate_and_fix_latex(content: str) -> str:
    content = re.sub(r'\$([^$]*)\$', lambda m: f'${m.group(1).strip()}$', content)
    content = re.sub(r'\\begin\{equation\}\s*\\end\{equation\}', '', content)
    def fix_align(match):
        align_content = match.group(1)
        lines = align_content.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.endswith('\\\\') and i < len(lines) - 1:
                line += ' \\\\'
            fixed_lines.append(line)
        return f'\\begin{{align}}\\n' + "\\n".join(fixed_lines) + '\\n\\end{{align}}'
    content = re.sub(r'\\begin\{align\}(.*?)\\end\{align\}', fix_align, content, flags=re.DOTALL)
    return content

def convert_latex_to_html(text: str) -> str:
    html_text = text
    html_text = re.sub(r'\\section\*?\{([^}]+)\}', r'<h2>\1</h2>', html_text)
    html_text = re.sub(r'\\subsection\*?\{([^}]+)\}', r'<h3>\1</h3>', html_text)
    html_text = re.sub(r'\\textbf\{([^}]+)\}', r'<strong>\1</strong>', html_text)
    html_text = re.sub(r'\\textit\{([^}]+)\}', r'<em>\1</em>', html_text)
    html_text = re.sub(r'\n\s*\n', '</p><p>', html_text)
    if not html_text.startswith('<'):
        html_text = '<p>' + html_text
    if not html_text.endswith('</p>'):
        html_text = html_text + '</p>'
    html_text = re.sub(r'(?<!>)\n(?!<)', '<br>', html_text)
    return html_text

def _pdf_doc(out_path: str):
    return SimpleDocTemplate(out_path, pagesize=A4,rightMargin=20 * mm, leftMargin=20 * mm,topMargin=15 * mm, bottomMargin=15 * mm)

def _pdf_styles():
    styles = getSampleStyleSheet()
    styles["BodyText"].leading = 14
    return styles

CATEGORY_TITLES_RU = {"определение": "Определения", "теорема": "Теоремы", "доказательство": "Доказательства","пример": "Примеры", "формула": "Формулы", "важный_факт": "Важные факты","дата": "Даты и события", "общий_текст": "Общий текст"}

def export_study_pack_pdf(segments: List[ClassifiedSegment], out_path="study_pack.pdf") -> str:
    doc = _pdf_doc(out_path)
    styles = _pdf_styles()
    h1, h2, body = styles["Heading1"], styles["Heading2"], styles["BodyText"]
    by_cat = {}
    for s in segments:
        key = s.category.lower().strip()
        by_cat.setdefault(key, []).append(s.text.strip())
    order = ["определение","теорема","доказательство","пример","формула","важный_факт","дата","общий_текст"]
    story = [Paragraph("Учебный конспект", h1), Spacer(1, 8)]
    for cat in order:
        items = by_cat.get(cat, [])
        if not items: continue
        story.append(Paragraph(CATEGORY_TITLES_RU.get(cat, cat.title()), h2))
        story.append(Spacer(1, 6))
        for t in items:
            story.append(Paragraph("• " + t.replace("\n", "<br/>"), body))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 8))
    doc.build(story)
    logger.info(f"PDF сохранён: {out_path}")
    return out_path

def extract_content_from_latex(text: str) -> str:
    content_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', text, re.DOTALL)
    if content_match:
        return content_match.group(1).strip()
    return clean_latex_for_document_body(text)

def clean_latex_for_document_body(text: str) -> str:
    text = re.sub(r'\\documentclass(?:\[.*?\])?\{.*?\}', '', text)
    text = re.sub(r'\\usepackage(?:\[.*?\])?\{.*?\}', '', text)
    text = re.sub(r'\\geometry\{.*?\}', '', text)
    text = re.sub(r'\\begin\{document\}', '', text)
    text = re.sub(r'\\end\{document\}', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def simplify_latex_content(content: str) -> str:
    content = re.sub(r'\\begin\{align\*?\}', r'\\begin{equation}', content)
    content = re.sub(r'\\end\{align\*?\}', r'\\end{equation}', content)
    content = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}',lambda m: m.group(0) if any(cmd in m.group(0) for cmd in ['frac','sqrt','sum','int','text']) else '',content)
    return content

def try_build_latex_pdf(tex_path: str, tmpdir: str, engines=("lualatex","xelatex","pdflatex"), passes=2, timeout=35):
    last_err = ""
    env = os.environ.copy()
    env.setdefault("LANG", "en_US.UTF-8")
    env.setdefault("LC_ALL", "en_US.UTF-8")
    for eng in engines:
        try:
            ok = True
            for i in range(passes):
                result = subprocess.run([eng, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", tex_path],cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=False, timeout=timeout, check=False, env=env)
                stdout = (result.stdout or b"").decode("utf-8", errors="replace")
                stderr = (result.stderr or b"").decode("utf-8", errors="replace")
                if result.returncode != 0:
                    ok = False
                    last_err = f"[{eng} pass {i + 1}] rc={result.returncode}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
                    logger.error(last_err); break
            if ok:
                pdf_path = os.path.join(tmpdir, "output.pdf")
                if os.path.exists(pdf_path):
                    logger.info(f"{eng}: PDF успешно собран: {pdf_path}")
                    return pdf_path, ""
        except subprocess.TimeoutExpired:
            last_err = f"{eng} timeout after {timeout}s"; logger.error(last_err)
        except FileNotFoundError:
            last_err = f"{eng} not found"; logger.error(last_err); continue
        except Exception as e:
            last_err = f"{eng} unexpected error: {e}"; logger.exception(last_err)
    return None, last_err

_MATH_TRIGGERS = (r"\\int", r"\\frac", r"\\sum", r"\\prod", r"\\lim", r"\\sqrt",r"\\sin", r"\\cos", r"\\tan", r"\\log", r"\\ln",r"\\alpha", r"\\beta", r"\\gamma", r"\\infty", r"\\left", r"\\right",)

def _strip_fake_single_letter_cmds(s: str) -> str:
    import re
    s = re.sub(r"\\([a-zA-Z])(?=\s*[\(\[\{])", r"\1", s)
    s = re.sub(r"\\([a-zA-Z])(\b)", r"\1\2", s)
    return s

def _line_needs_math(line: str) -> bool:
    import re
    if "$" in line or r"\[" in line or r"\(" in line:
        return False
    for p in _MATH_TRIGGERS:
        if re.search(p, line): return True
    if re.search(r"[\^_]\s*[\{\(]?", line): return True
    return False

def auto_wrap_display_math(text: str) -> str:
    lines = text.splitlines(); out = []
    for ln in lines:
        l = ln.strip()
        if _line_needs_math(l):
            out.append(r"\["); out.append(ln); out.append(r"\]")
        else:
            out.append(ln)
    return "\n".join(out)

def heal_latex_content(s: str) -> str:
    import re
    s = _strip_latex_fences(s or "")
    if s.count("$") % 2 == 1: s = s.replace("$", "")
    opens = len(re.findall(r"\\\[", s)); closes = len(re.findall(r"\\\]", s))
    if opens > closes: s += ("\n" + "\\]") * (opens - closes)
    s = re.sub(r'\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}', r'\\begin{equation}\1\\end{equation}', s)
    s = re.sub(r'\\(write|input|include)\b.*', '', s)
    s = _strip_fake_single_letter_cmds(s)
    return s

def generate_latex_document(segments: List[ClassifiedSegment], original_content: str = "") -> str:
    by_cat = {}
    for s in segments:
        key = s.category.lower().strip()
        by_cat.setdefault(key, []).append(s.text.strip())
    order = ["определение","теорема","доказательство","пример","формула","важный_факт","дата","общий_текст"]
    latex_content = r"""\documentclass[12pt,a4paper]{article}
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[russian]{babel}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage[colorlinks=true, linkcolor=blue]{hyperref}
\usepackage{xcolor}

\definecolor{definitionbg}{RGB}{220, 252, 231}
\definecolor{theorembg}{RGB}{254, 226, 226}
\definecolor{examplebg}{RGB}{254, 243, 199}
\definecolor{formulabg}{RGB}{207, 250, 254}
\definecolor{datebg}{RGB}{233, 213, 255}

\newcommand{\highlightdefinition}[1]{\colorbox{definitionbg}{#1}}
\newcommand{\highlighttheorem}[1]{\colorbox{theorembg}{#1}}
\newcommand{\highlightexample}[1]{\colorbox{examplebg}{#1}}
\newcommand{\highlightformula}[1]{\colorbox{formulabg}{#1}}
\newcommand{\highlightdate}[1]{\colorbox{datebg}{#1}}

\begin{document}

\title{Классифицированный конспект}
\author{ClassifyMyNotes AI}
\date{\today}
\maketitle

\tableofcontents
\newpage

"""
    if original_content.strip():
        latex_content += r"""
\section{Оригинальный текст}
""" + original_content + r"""

\newpage
"""
    latex_content += r"""
\section{Классификация по типам}

"""
    for cat in order:
        items = by_cat.get(cat, [])
        if not items: continue
        section_title = CATEGORY_TITLES_RU.get(cat, cat.title())
        latex_content += f"\\subsection{{{section_title}}}\n\n"
        for item in items:
            if cat == "формула":
                latex_content += f"\\begin{{equation}}\n{item}\n\\end{{equation}}\n\n"
            else:
                escaped_item = item.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
                latex_content += f"\\textbf{{•}} {escaped_item}\n\n"
    latex_content += r"""
\end{document}
"""
    return latex_content
