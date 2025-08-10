import os
import re
import time
import json
from PIL import Image
import base64
import subprocess
import mimetypes
import logging
import asyncio
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from io import BytesIO
import tempfile
import shutil
from fastapi.responses import FileResponse, PlainTextResponse
import os
import tempfile
import subprocess
import logging
from fastapi import UploadFile
from PyPDF2 import PdfReader
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn
from io import BytesIO
import pdfplumber
from pdf2image import convert_from_bytes
import mimetypes
from openai import OpenAI

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

import spacy.cli
import spacy
from collections import Counter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL_OCR = os.getenv("OPENAI_MODEL_OCR", "gpt-4o-mini")
OPENAI_MODEL_CLS = os.getenv("OPENAI_MODEL_CLS", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClassifyMyNotes")

app = FastAPI(title="ClassifyMyNotes API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/web", StaticFiles(directory="static", html=True), name="web")


@app.get("/")
def root():
    return RedirectResponse(url="/web/")


client = OpenAI(api_key=OPENAI_API_KEY)


def chat_call_with_retry(**kwargs):
    backoff = 2.0
    for i in range(6):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg or "temporarily" in msg:
                sleep_s = backoff * (i + 1)
                logger.warning(f"Retry {i + 1}: {e} -> sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue
            raise


def strip_latex_preamble(text: str) -> str:
    # Убираем \documentclass, \usepackage и \begin{document}, \end{document}
    text = re.sub(r'\\documentclass.*', '', text)
    text = re.sub(r'\\usepackage.*', '', text)
    text = re.sub(r'\\begin\{document\}', '', text)
    text = re.sub(r'\\end\{document\}', '', text)
    return text.strip()


def load_spacy_ru():
    import spacy
    try:
        return spacy.load("ru_core_news_sm")
    except OSError:
        import spacy.cli
        logger.info("Скачиваю spacy ru_core_news_sm...")
        spacy.cli.download("ru_core_news_sm")
        return spacy.load("ru_core_news_sm")


nlp = load_spacy_ru()


def latex_escape(text: str) -> str:
    """
    Экранирует спецсимволы LaTeX, чтобы документ не падал при компиляции.
    """
    replacements = {
        "\\": r"\\",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "%": r"\%",
        "~": r"\textasciitilde{}",
        "^": r"\^{}"
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text


@dataclass
class ClassifiedSegment:
    text: str
    category: str
    confidence: float
    start_pos: int
    end_pos: int


class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    suggestions: List[str]


class ProcessingResult(BaseModel):
    success: bool
    extracted_text: str
    segments: List[Dict]
    highlighted_html: str
    summary: Dict
    file_info: Dict
    latex_source: Optional[str] = None

# ===== OCR smart utils =====


OPENAI_MODEL_OCR_PRIMARY = os.getenv("OPENAI_MODEL_OCR", "gpt-4o-mini")
OPENAI_MODEL_OCR_FALLBACK = os.getenv("OPENAI_MODEL_OCR_FALLBACK", "gpt-4o")

def _text_looks_gibberish(s: str) -> bool:
    """Простая эвристика для детекта мусора."""
    if not s or len(s.strip()) < 20:
        return True
    bad_markers = ("�",)
    if any(b in s for b in bad_markers):
        return True
    # слишком мало пробелов/гласных -> похоже на «кашу»
    spaces = s.count(" ")
    if spaces < max(3, len(s)//50):
        return True
    vowels = sum(ch.lower() in "аеёиоуыэюяaeiou" for ch in s)
    if vowels < max(5, len(s)//60):
        return True
    return False

def _img_to_png_bytes(img: Image.Image, scale: float = 1.0) -> bytes:
    """Сохранить как PNG (без потерь), при необходимости масштабировать."""
    if scale != 1.0:
        w, h = img.size
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

def _tile_image(img: Image.Image, grid=(2,2), pad=4) -> list[bytes]:
    """Разбить страницу на тайлы (для сложных/плотных страниц)."""
    W, H = img.size
    cols, rows = grid
    tiles = []
    w = W // cols
    h = H // rows
    for r in range(rows):
        for c in range(cols):
            left = max(0, c*w - (pad if c else 0))
            top  = max(0, r*h - (pad if r else 0))
            right = min(W, (c+1)*w + (pad if c+1<cols else 0))
            bottom = min(H, (r+1)*h + (pad if r+1<rows else 0))
            crop = img.crop((left, top, right, bottom))
            tiles.append(_img_to_png_bytes(crop, scale=1.0))
    return tiles

async def _ocr_trial(image_bytes: bytes, *, latex_mode: bool, model: str, lang_hint: str = "русский, английский") -> str:
    """Один вызов OCR с заданным режимом/моделью."""
    instruction_latex = (
        "Распознай весь текст с изображения и верни его в LaTeX формате.\n"
        "ВАЖНО: Верни ТОЛЬКО содержимое (текст/формулы), без преамбулы и без комментариев.\n"
        "Язык: " + lang_hint + "."
    )
    instruction_plain = (
        "Распознай весь текст с изображения максимально точно, сохраняя переносы строк. "
        "Возвращай только чистый текст без комментариев. Язык: " + lang_hint + "."
    )
    instruction = instruction_latex if latex_mode else instruction_plain

    resp = chat_call_with_retry(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": as_data_url_from_bytes(image_bytes, "image/png")}}
            ]
        }],
        max_tokens=2048,
        temperature=0.0
    )
    text = resp.choices[0].message.content or ""
    if latex_mode:
        text = _strip_latex_fences(text)
    return normalize_text_unicode(text)

async def ocr_page_smart(img: Image.Image) -> str:
    """
    OCR одной страницы с ретраями:
    1) PNG @ масштаб 1.0, latex, PRIMARY
    2) PNG @ масштаб 1.0, plain,  PRIMARY
    3) PNG @ масштаб 1.25, latex,  FALLBACK
    4) PNG тайлами 2×2, plain,     FALLBACK
    Возвращает лучший из удачных вариантов, иначе — пустую строку.
    """
    trials = []

    # 1–2: обычная страница
    png_1x = _img_to_png_bytes(img, scale=1.0)
    trials.append((png_1x, True,  OPENAI_MODEL_OCR_PRIMARY))
    trials.append((png_1x, False, OPENAI_MODEL_OCR_PRIMARY))

    # 3: небольшой апскейл + более сильная модель
    png_125 = _img_to_png_bytes(img, scale=1.25)
    trials.append((png_125, True,  OPENAI_MODEL_OCR_FALLBACK))

    # Пробуем по очереди
    for image_bytes, latex_mode, model in trials:
        txt = await _ocr_trial(image_bytes, latex_mode=latex_mode, model=model)
        if not _text_looks_gibberish(txt):
            return txt

    # 4: тайлинг 2×2 (plain + сильная модель)
    parts = []
    for tile_bytes in _tile_image(img, grid=(2,2)):
        t = await _ocr_trial(tile_bytes, latex_mode=False, model=OPENAI_MODEL_OCR_FALLBACK)
        parts.append(t)
    joined = "\n".join(parts).strip()
    if not _text_looks_gibberish(joined):
        return joined

    return joined  # пусть вернёт лучшее, даже если не идеально


def as_data_url_from_bytes(data: bytes, mime_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def safe_truncate(s: str, limit=12000):
    return s if len(s) <= limit else s[:limit] + "\n...[truncated]..."


def fix_latex_commands(text: str) -> str:
    """Исправляет часто встречающиеся ошибки в LaTeX командах"""
    # Исправляем двойные обратные слеши перед буквами
    text = re.sub(r'\\\\\s+([A-Za-z])', r'\\\1', text)
    # Исправляем неправильные переносы строк в формулах
    text = re.sub(r'\\\\(\s*[a-zA-Z])', r'\\\\\n\1', text)
    return text


async def ocr_image_with_gpt(
        image_data: bytes,
        model: str = OPENAI_MODEL_OCR,
        max_tokens: int = 2048,
        latex_mode: bool = False
) -> str:
    """OCR изображения через OpenAI Vision-модель"""
    data_url = as_data_url_from_bytes(image_data)
    logger.info(f"OCR через {model} в режиме {'latex' if latex_mode else 'plain'}...")

    if latex_mode:
        instruction = (
            "Распознай весь текст с изображения и верни его в LaTeX формате.\n"
            "ВАЖНО: Возвращай ТОЛЬКО содержимое документа (текст, формулы, окружения), "
            "БЕЗ \\documentclass, \\usepackage, \\begin{document}, \\end{document}.\n"
            "Используй окружения \\begin{equation}, \\begin{align} для формул.\n"
            "Для inline формул используй $...$.\n"
            "Возвращай только LaTeX код содержимого, без комментариев."
        )
    else:
        instruction = (
            "Перепиши весь текст с изображения максимально точно, включая формулы. "
            "Возвращай только текст, без анализа и комментариев."
        )

    resp = chat_call_with_retry(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }],
        max_tokens=max_tokens,
        temperature=0.0
    )

    # Логируем usage
    try:
        if hasattr(resp, "usage"):
            prompt_tokens = getattr(resp.usage, "prompt_tokens", None)
            completion_tokens = getattr(resp.usage, "completion_tokens", None)
            total_tokens = getattr(resp.usage, "total_tokens", None)
            logger.info(f"OCR токены — prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")
    except Exception as e:
        logger.warning(f"Не удалось получить usage из ответа: {e}")

    text = resp.choices[0].message.content or ""

    # Дополнительная очистка на случай, если GPT все же вернул полный документ
    if latex_mode:
        text = _strip_latex_fences(text)
        text = extract_content_from_latex(text)

    return text.strip()


def _strip_latex_fences(s: str) -> str:
    """Убираем markdown code fences если модель такое вернула"""
    s = s.strip()
    # Убираем ```latex ... ```
    s = re.sub(r"^```(?:latex)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    return s.strip()


def validate_and_fix_latex(content: str) -> str:
    """
    Проверяет и исправляет распространенные ошибки в LaTeX коде
    """
    # Исправляем несбалансированные скобки в формулах
    content = re.sub(r'\$([^$]*)\$', lambda m: f'${m.group(1).strip()}$', content)

    # Исправляем пустые equation окружения
    content = re.sub(r'\\begin\{equation\}\s*\\end\{equation\}', '', content)

    # Добавляем пропущенные \\ в align окружениях
    def fix_align(match):
        align_content = match.group(1)
        lines = align_content.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line and not line.endswith('\\\\') and i < len(lines) - 1:
                line += ' \\\\'
            fixed_lines.append(line)
        return f'\\begin{{align}}\n{chr(10).join(fixed_lines)}\n\\end{{align}}'

    content = re.sub(r'\\begin\{align\}(.*?)\\end\{align\}', fix_align, content, flags=re.DOTALL)
    return content


CLS_INSTRUCTIONS = (
    "Определи тип фрагмента текста одним словом: "
    "определение, теорема, доказательство, пример, формула, важный_факт, дата, общий_текст.\n\n"
)


def classify_segment_with_gpt(segment: str, model: str = OPENAI_MODEL_CLS) -> str:
    prompt = f"{CLS_INSTRUCTIONS}Фрагмент:\n\"\"\"\n{segment}\n\"\"\"\nОтвет одним словом на русском:"
    resp = chat_call_with_retry(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0
    )
    return (resp.choices[0].message.content or "").strip().lower()


def convert_latex_to_html(text: str) -> str:
    """
    Конвертирует LaTeX текст в HTML с поддержкой MathJax
    """
    # Оставляем LaTeX математику как есть для MathJax
    # Заменяем только структурные элементы LaTeX на HTML

    # Заменяем LaTeX окружения на HTML с сохранением математики
    html_text = text

    # Заменяем \section{...} на <h2>...</h2>
    html_text = re.sub(r'\\section\*?\{([^}]+)\}', r'<h2>\1</h2>', html_text)

    # Заменяем \subsection{...} на <h3>...</h3>
    html_text = re.sub(r'\\subsection\*?\{([^}]+)\}', r'<h3>\1</h3>', html_text)

    # Заменяем \textbf{...} на <strong>...</strong>
    html_text = re.sub(r'\\textbf\{([^}]+)\}', r'<strong>\1</strong>', html_text)

    # Заменяем \textit{...} на <em>...</em>
    html_text = re.sub(r'\\textit\{([^}]+)\}', r'<em>\1</em>', html_text)

    # Заменяем двойные переводы строк на абзацы
    html_text = re.sub(r'\n\s*\n', '</p><p>', html_text)

    # Оборачиваем в параграфы если нужно
    if not html_text.startswith('<'):
        html_text = '<p>' + html_text
    if not html_text.endswith('</p>'):
        html_text = html_text + '</p>'

    # Заменяем одиночные переводы строк на <br>
    html_text = re.sub(r'(?<!>)\n(?!<)', '<br>', html_text)

    return html_text


class GPTNoteAnalyzer:
    def split_into_segments(self, text: str) -> List[str]:
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]

    def analyze_text(self, text: str) -> List[ClassifiedSegment]:
        logger.info("Классификация сегментов (GPT)...")
        segments = self.split_into_segments(text)
        logger.info(f"Найдено сегментов: {len(segments)}")
        results: List[ClassifiedSegment] = []
        pos = 0
        for seg in segments:
            cat = classify_segment_with_gpt(seg)
            results.append(ClassifiedSegment(
                text=seg,
                category=cat,
                confidence=1.0,
                start_pos=pos,
                end_pos=pos + len(seg)
            ))
            pos += len(seg) + 1
        return results

    def create_highlighted_html(self, text: str, segments: List[ClassifiedSegment]) -> str:
        # Сначала конвертируем LaTeX в HTML
        html_text = convert_latex_to_html(text)

        # Маппинг категорий на CSS классы
        category_classes = {
            "определение": "highlight-definition",
            "теорема": "highlight-theorem",
            "доказательство": "highlight-theorem",
            "пример": "highlight-example",
            "формула": "highlight-formula",
            "важный_факт": "highlight-important",
            "дата": "highlight-date",
            "общий_текст": ""
        }

        segments_sorted = sorted(segments, key=lambda x: x.start_pos, reverse=True)

        for segment in segments_sorted:
            css_class = category_classes.get(segment.category, "")
            if css_class:
                # Ищем текст сегмента в HTML и оборачиваем в span
                segment_text = segment.text
                if segment_text in html_text:
                    highlighted_text = f'<span class="{css_class}" title="{segment.category.title()}">{segment_text}</span>'
                    html_text = html_text.replace(segment_text, highlighted_text, 1)

        return html_text

    def summary_text(self, src_text: str, segments: List[ClassifiedSegment]) -> Dict:
        counts = Counter(s.category for s in segments)

        categories = {
            "definitions": [s for s in segments if "определение" in s.category],
            "examples": [s for s in segments if "пример" in s.category],
            "theorems": [s for s in segments if "теорема" in s.category or "доказательство" in s.category],
            "formulas": [s for s in segments if "формула" in s.category],
            "dates": [s for s in segments if "дата" in s.category],
        }

        topics = self.extract_topics(src_text)

        return {
            "total_segments": len(segments),
            "total_chars": len(src_text),
            "categories": {
                "definitions": len(categories["definitions"]),
                "examples": len(categories["examples"]),
                "theorems": len(categories["theorems"]),
                "formulas": len(categories["formulas"]),
                "dates": len(categories["dates"])
            },
            "topics": topics,
            "segments_by_category": {k: [asdict(s) for s in v] for k, v in categories.items()}
        }

    def extract_topics(self, text: str) -> List[str]:
        topics = []

        math_keywords = {
            "производная": "Производные",
            "интеграл": "Интегралы",
            "функция": "Функции",
            "уравнение": "Уравнения",
            "матрица": "Матрицы",
            "предел": "Пределы"
        }

        for keyword, topic in math_keywords.items():
            if keyword in text.lower() and topic not in topics:
                topics.append(topic)

        return topics[:5]


def _pdf_doc(out_path: str):
    return SimpleDocTemplate(
        out_path, pagesize=A4,
        rightMargin=20 * mm, leftMargin=20 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm
    )


def _pdf_styles():
    styles = getSampleStyleSheet()
    styles["BodyText"].leading = 14
    return styles


CATEGORY_TITLES_RU = {
    "определение": "Определения",
    "теорема": "Теоремы",
    "доказательство": "Доказательства",
    "пример": "Примеры",
    "формула": "Формулы",
    "важный_факт": "Важные факты",
    "дата": "Даты и события",
    "общий_текст": "Общий текст"
}


def export_study_pack_pdf(segments: List[ClassifiedSegment], out_path="study_pack.pdf") -> str:
    doc = _pdf_doc(out_path)
    styles = _pdf_styles()
    h1, h2, body = styles["Heading1"], styles["Heading2"], styles["BodyText"]

    # Группировка
    by_cat = {}
    for s in segments:
        key = s.category.lower().strip()
        by_cat.setdefault(key, []).append(s.text.strip())

    order = [
        "определение", "теорема", "доказательство",
        "пример", "формула", "важный_факт",
        "дата", "общий_текст"
    ]

    story = [Paragraph("Учебный конспект", h1), Spacer(1, 8)]

    for cat in order:
        items = by_cat.get(cat, [])
        if not items:
            continue
        story.append(Paragraph(CATEGORY_TITLES_RU.get(cat, cat.title()), h2))
        story.append(Spacer(1, 6))
        for t in items:
            story.append(Paragraph("• " + t.replace("\n", "<br/>"), body))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 8))

    doc.build(story)
    logger.info(f"PDF сохранён: {out_path}")
    return out_path


def generate_latex_document(segments: List[ClassifiedSegment], original_content: str = "") -> str:
    """Генерирует полный LaTeX документ из сегментов"""

    # Группировка сегментов по категориям
    by_cat = {}
    for s in segments:
        key = s.category.lower().strip()
        by_cat.setdefault(key, []).append(s.text.strip())

    order = [
        "определение", "теорема", "доказательство",
        "пример", "формула", "важный_факт",
        "дата", "общий_текст"
    ]

    # Начало документа
    latex_content = r"""\documentclass[12pt,a4paper]{article}
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[russian]{babel}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage[colorlinks=true, linkcolor=blue]{hyperref}
\usepackage{xcolor}

% Определяем цвета для категорий
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

    # Если есть оригинальный контент, добавляем его
    if original_content.strip():
        latex_content += r"""
\section{Оригинальный текст}
""" + original_content + r"""

\newpage
"""

    # Добавляем разделы по категориям
    latex_content += r"""
\section{Классификация по типам}

"""

    for cat in order:
        items = by_cat.get(cat, [])
        if not items:
            continue

        section_title = CATEGORY_TITLES_RU.get(cat, cat.title())
        latex_content += f"\\subsection{{{section_title}}}\n\n"

        for item in items:
            # Не экранируем LaTeX для математических формул
            if cat == "формула":
                latex_content += f"\\begin{{equation}}\n{item}\n\\end{{equation}}\n\n"
            else:
                # Экранируем только специальные символы, сохраняя LaTeX команды
                escaped_item = item.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
                latex_content += f"\\textbf{{•}} {escaped_item}\n\n"

    latex_content += r"""
\end{document}
"""
    return latex_content


async def ocr_pdf_with_gpt(pdf_input, mode: str = "latex") -> str:
    """
    Новый OCR PDF: конвертим в PNG @ 300dpi и гоняем каждую страницу через ocr_page_smart.
    """
    if isinstance(pdf_input, (bytes, bytearray)):
        data = pdf_input
    else:
        with open(pdf_input, "rb") as f:
            data = f.read()

    images = convert_from_bytes(data, dpi=300)  # dpi повыше и стабильнее
    page_texts = []

    for idx, img in enumerate(images, start=1):
        try:
            txt = await ocr_page_smart(img)
        except Exception as e:
            logger.exception("OCR page %d failed: %s", idx, e)
            txt = ""
        page_texts.append(f"% Страница {idx}\n{txt}")

    return "\n\n".join(page_texts)


def _page_is_gibberish(s: str) -> bool:
    if not s:
        return True
    # Доля «не букв/цифр/пробелов» слишком высока → мусор
    import string
    allowed = set(string.ascii_letters + string.digits + string.whitespace + ".,;:!?-–—()[]{}«»\"'/%+*=<>|\\_^$№±~")
    bad_ratio = sum(ch not in allowed for ch in s) / max(1, len(s))
    if bad_ratio > 0.35:
        return True
    # Микс латиницы и кириллицы в одном слове — характерный признак
    mixed = 0
    for w in re.findall(r"\w+", s):
        has_lat = re.search(r"[A-Za-z]", w) is not None
        has_cyr = re.search(r"[А-Яа-яЁё]", w) is not None
        if has_lat and has_cyr:
            mixed += 1
    if mixed >= 5:
        return True
    # Очень мало кириллицы при видимой длине
    cyr = sum("А" <= ch <= "я" or ch in "Ёё" for ch in s)
    if cyr < max(20, len(s)//80):
        # не всегда мусор, но часто — да
        return True
    return False

import unicodedata  # добавь к импортам сверху

# Чаще всего встречающиеся подмены символов после OCR/экстракции
_CYR_CONFUSABLES = {
    ord("ĸ"): "к",   # latin kra -> кир. к
    ord("ı"): "и",   # dotless i -> кир. и (чаще уместно для русских текстов)
    ord("’"): "'",
    ord("`"): "'",
    ord("“"): '"',
    ord("”"): '"',
}

def normalize_cyrillic_confusables(s: str) -> str:
    return s.translate(_CYR_CONFUSABLES)

def normalize_text_unicode(s: str) -> str:
    """
    Нормализует текст: убирает markdown-фенсы, приводит к NFC
    и заменяет частые псевдо-кириллические символы.
    """
    s = s or ""
    try:
        # если у тебя уже есть _strip_latex_fences — используем его;
        # если нет, можно убрать эту строку.
        s = _strip_latex_fences(s)
    except NameError:
        pass
    s = unicodedata.normalize("NFC", s)
    s = normalize_cyrillic_confusables(s)
    return s



async def extract_pdf_text_smart(file_path: str) -> str:
    """
    Стратегия:
    1) сначала пробуем PyPDF2 постранично;
    2) где страница 'мусор' — OCR только этой страницы (GPT-4o-mini).
    3) если PyPDF2 совсем пусто — делаем полный OCR.
    """
    text_pages = []
    try:
        reader = PdfReader(file_path)
        had_any_text = False
        raw_per_page = []
        for p in reader.pages:
            t = p.extract_text() or ""
            raw_per_page.append(t)
            if t.strip():
                had_any_text = True

        if not had_any_text:
            # полный OCR
            return await ocr_pdf_with_gpt(file_path, mode="latex")

        # подготовим картинки всех страниц разом, но будем брать только при необходимости
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        images = convert_from_bytes(pdf_bytes, dpi=250)  # dpi повыше, OCR качественнее

        for idx, t in enumerate(raw_per_page, start=1):
            t_norm = normalize_text_unicode(t)
            if _page_is_gibberish(t_norm):
                # OCR только этой страницы
                buf = BytesIO()
                images[idx-1].save(buf, format="JPEG")
                ocr_txt = await ocr_page_smart(images[idx - 1])
                ocr_txt = normalize_text_unicode(_strip_latex_fences(ocr_txt))
                text_pages.append(f"% Страница {idx}\n{ocr_txt}")
            else:
                text_pages.append(f"% Страница {idx}\n{t_norm}")

        return "\n\n".join(text_pages)

    except Exception as e:
        logger.exception("extract_pdf_text_smart fallback to full OCR: %s", e)
        return await ocr_pdf_with_gpt(file_path, mode="latex")


async def ask_llm_about_notes(context_text: str, question: str, model: str = OPENAI_MODEL_CLS) -> str:
    prompt = (
        "Ниже учебные заметки. Отвечай кратко и по делу, "
        "ссылайся на определения/теоремы из контекста, если уместно.\n\n"
        f"КОНТЕКСТ:\n{safe_truncate(context_text)}\n\nВОПРОС: {question}"
    )
    resp = chat_call_with_retry(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.2
    )
    return (resp.choices[0].message.content or "").strip()


def detect_mime(filename: str, declared: str | None) -> str:
    if declared:
        return declared
    guess, _ = mimetypes.guess_type(filename)
    return guess or "application/octet-stream"


@app.post("/upload")
async def upload_and_process(file: UploadFile):
    logger.info("Получен файл: %s, тип: %s, размер: %d байт",
                file.filename, file.content_type, file.size if hasattr(file, "size") else -1)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        extracted_text = ""
        if file.filename.lower().endswith(".pdf"):
            logger.info("PDF → smart extract (текстовый слой + OCR-фоллбек постранично)…")
            extracted_text = await extract_pdf_text_smart(file_path)
        else:
            extracted_text = ""

        # Выполняем классификацию текста
        analyzer = GPTNoteAnalyzer()
        segments = analyzer.analyze_text(extracted_text)
        highlighted_html = analyzer.create_highlighted_html(extracted_text, segments)
        summary = analyzer.summary_text(extracted_text, segments)

        # Обрабатываем LaTeX контент для PDF
        if extracted_text.strip().startswith('\\documentclass') or '\\begin{document}' in extracted_text:
            document_content = extract_content_from_latex(extracted_text)
        else:
            document_content = extracted_text

        # ← вот здесь как раз и находится место для вставки:
        healed = heal_latex_content(document_content)  # (1) подлатать сырой LaTeX из OCR
        healed = auto_wrap_display_math(healed)  # (2) завернуть «похожие на формулы» строки в \[...\]
        clean_content = fix_latex_commands(healed)  # (3) финальные мелкие фиксы спецсимволов

        # Полный LaTeX документ (оригинал + классификация)
        full_latex_source = generate_latex_document(segments, clean_content)

        # Минимальный документ для попытки сборки PDF (оригинал OCR)
        latex_template = r"""\documentclass[12pt,a4paper]{article}
        \usepackage{fontspec}
        \usepackage{polyglossia}
        \setdefaultlanguage{russian}
        \setotherlanguage{english}
        \defaultfontfeatures{Ligatures=TeX}
        \setmainfont{CMU Serif}
        \setsansfont{CMU Sans Serif}
        \setmonofont{CMU Typewriter Text}
        \usepackage{unicode-math}
        \setmathfont{Latin Modern Math}
        \usepackage{geometry}
        \geometry{margin=1in}
        \usepackage[colorlinks=true, linkcolor=blue]{hyperref}

        \begin{document}
        \title{Распознанный документ}
        \author{}
        \date{\today}
        \maketitle

        """ + clean_content + r"""

        \end{document}"""

        tex_path = os.path.join(tmpdir, "output.tex")
        with open(tex_path, "w", encoding='utf-8') as tex_file:
            tex_file.write(latex_template)

        # Пытаемся собрать PDF «мягко»
        pdf_path, build_err = try_build_latex_pdf(tex_path, tmpdir)

        pdf_b64 = None
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                pdf_b64 = base64.b64encode(pdf_file.read()).decode("ascii")
        else:
            # пробуем очень упрощённый шаблон, если первая сборка не удалась
            simplified_content = simplify_latex_content(clean_content)
            simplified_template = r"""\documentclass[12pt,a4paper]{article}
            \usepackage{fontspec}
            \usepackage{polyglossia}
            \setdefaultlanguage{russian}
            \defaultfontfeatures{Ligatures=TeX}
            \setmainfont{CMU Serif}
            \usepackage{geometry}
            \geometry{margin=1in}

            \begin{document}
            \section*{Распознанный текст}

            """ + simplified_content + r"""

            \end{document}"""

            with open(tex_path, "w", encoding='utf-8') as tex_file:
                tex_file.write(simplified_template)
            pdf_path2, build_err2 = try_build_latex_pdf(tex_path, tmpdir)
            if pdf_path2 and os.path.exists(pdf_path2):
                with open(pdf_path2, "rb") as pdf_file:
                    pdf_b64 = base64.b64encode(pdf_file.read()).decode("ascii")
            else:
                logger.error("LaTeX build failed, returning without pdf_base64.\nFirst error:\n%s\nSecond error:\n%s",
                             build_err, build_err2)

        return {
            "success": True,
            "extracted_text": extracted_text,
            "segments": [asdict(s) for s in segments],
            "highlighted_html": highlighted_html,
            "summary": summary,
            "pdf_base64": pdf_b64,  # может быть None, фронт умеет фоллбекнуться
            "latex_source": full_latex_source,
            "file_info": {
                "filename": file.filename,
                "original_text_length": len(extracted_text),
                "processed_text_length": len(clean_content),
                "segments_count": len(segments),
                "latex_build_ok": bool(pdf_b64)
            }
        }


# Дополнительные вспомогательные функции
def extract_content_from_latex(text: str) -> str:
    """Извлекает только содержимое из LaTeX документа"""
    content_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', text, re.DOTALL)
    if content_match:
        return content_match.group(1).strip()
    return clean_latex_for_document_body(text)


def clean_latex_for_document_body(text: str) -> str:
    """Очищает LaTeX текст для вставки в тело документа"""
    text = re.sub(r'\\documentclass(?:\[.*?\])?\{.*?\}', '', text)
    text = re.sub(r'\\usepackage(?:\[.*?\])?\{.*?\}', '', text)
    text = re.sub(r'\\geometry\{.*?\}', '', text)
    text = re.sub(r'\\begin\{document\}', '', text)
    text = re.sub(r'\\end\{document\}', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def simplify_latex_content(content: str) -> str:
    """Упрощает LaTeX контент, удаляя потенциально проблемные команды"""
    content = re.sub(r'\\begin\{align\*?\}', r'\\begin{equation}', content)
    content = re.sub(r'\\end\{align\*?\}', r'\\end{equation}', content)
    content = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}',
                     lambda m: m.group(0) if any(
                         cmd in m.group(0) for cmd in ['frac', 'sqrt', 'sum', 'int', 'text']) else '',
                     content)
    return content


def try_build_latex_pdf(tex_path: str, tmpdir: str, engines=("lualatex", "xelatex", "pdflatex"), passes=2, timeout=35):
    """
    Пытается собрать PDF несколькими движками. Возвращает (pdf_path|None, error_text).
    НЕ декодирует вывод автоматически (text=False) — сами декодируем с errors='replace'.
    """
    last_err = ""
    # Стабильная локаль для подпроцессов
    env = os.environ.copy()
    env.setdefault("LANG", "en_US.UTF-8")
    env.setdefault("LC_ALL", "en_US.UTF-8")

    for eng in engines:
        try:
            ok = True
            for i in range(passes):
                result = subprocess.run(
                    [eng, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", tex_path],
                    cwd=tmpdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,          # <-- важно: получаем байты
                    timeout=timeout,
                    check=False,
                    env=env,
                )
                # Декодируем руками, даже если там «грязные» байты
                stdout = (result.stdout or b"").decode("utf-8", errors="replace")
                stderr = (result.stderr or b"").decode("utf-8", errors="replace")

                if result.returncode != 0:
                    ok = False
                    last_err = f"[{eng} pass {i+1}] rc={result.returncode}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
                    logger.error(last_err)
                    break

            if ok:
                pdf_path = os.path.join(tmpdir, "output.pdf")
                if os.path.exists(pdf_path):
                    logger.info(f"{eng}: PDF успешно собран: {pdf_path}")
                    return pdf_path, ""

        except subprocess.TimeoutExpired as te:
            last_err = f"{eng} timeout after {timeout}s"
            logger.error(last_err)
        except FileNotFoundError:
            last_err = f"{eng} not found"
            logger.error(last_err)
            continue
        except Exception as e:
            # На всякий случай — не валимся из-за декодера
            last_err = f"{eng} unexpected error: {e}"
            logger.exception(last_err)

    return None, last_err

_MATH_TRIGGERS = (
    r"\\int", r"\\frac", r"\\sum", r"\\prod", r"\\lim", r"\\sqrt",
    r"\\sin", r"\\cos", r"\\tan", r"\\log", r"\\ln",
    r"\\alpha", r"\\beta", r"\\gamma", r"\\infty", r"\\left", r"\\right",
)

# частые «ложные» команды от OCR: \f(z), \g(x) и т.п.
def _strip_fake_single_letter_cmds(s: str) -> str:
    # \f( -> f(   \g[ -> g[   \h{ -> h{
    s = re.sub(r"\\([a-zA-Z])(?=\s*[\(\[\{])", r"\1", s)
    # одиночный \ перед лат. буквой и НЕ за которым идёт буква (не команда) -> убрать слеш
    s = re.sub(r"\\([a-zA-Z])(\b)", r"\1\2", s)
    return s

def _line_needs_math(line: str) -> bool:
    # уже в math?
    if "$" in line or r"\[" in line or r"\(" in line:
        return False
    # триггеры
    for p in _MATH_TRIGGERS:
        if re.search(p, line):
            return True
    # эвристика: ^,_ возле цифр/скобок
    if re.search(r"[\^_]\s*[\{\(]?", line):
        return True
    return False

def auto_wrap_display_math(text: str) -> str:
    r"""Строки с LaTeX-формулами, не окружённые math, оборачиваем в \[...\]."""
    lines = text.splitlines()
    out = []
    for ln in lines:
        l = ln.strip()
        if _line_needs_math(l):
            out.append(r"\[")
            out.append(ln)
            out.append(r"\]")
        else:
            out.append(ln)
    return "\n".join(out)

def heal_latex_content(s: str) -> str:
    r"""
    Быстрая «подлатка» LaTeX после OCR:
    - убираем code-fences
    - балансируем $...$
    - закрываем \[ ... \] если открытий больше
    - лечим align → equation при битых окружениях
    - вычищаем потенциально опасные команды
    - УДАЛЯЕМ мусорные \f( и подобные
    """
    s = _strip_latex_fences(s or "")

    # если число $ нечётное — убираем все $, чтобы не падал компилятор
    if s.count("$") % 2 == 1:
        s = s.replace("$", "")

    # балансируем \[ ... \]
    opens = len(re.findall(r"\\\[", s))
    closes = len(re.findall(r"\\\]", s))
    if opens > closes:
        s += ("\n" + "\\]") * (opens - closes)

    # align → equation (битые)
    s = re.sub(r'\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}', r'\\begin{equation}\1\\end{equation}', s)

    # опасные директивы
    s = re.sub(r'\\(write|input|include)\b.*', '', s)

    # НОВОЕ: убираем \f( / \g( / одиночные \x
    s = _strip_fake_single_letter_cmds(s)

    return s




@app.post("/chat")
async def chat_with_ai(message: ChatMessage):
    try:
        response = await ask_llm_about_notes(
            message.context or "",
            message.message
        )

        suggestions = [
            "Можешь объяснить это проще?",
            "Приведи еще один пример",
            "Как это связано с другими темами?",
            "Какие могут быть практические применения?"
        ]

        return ChatResponse(response=response, suggestions=suggestions)

    except Exception as e:
        logger.error(f"Ошибка чата: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка чата: {str(e)}")


@app.post("/export/pdf")
async def export_pdf(segments_data: List[Dict]):
    try:
        segments = [ClassifiedSegment(**data) for data in segments_data]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = export_study_pack_pdf(segments, tmp.name)

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename="классифицированный_конспект.pdf"
        )

    except Exception as e:
        logger.error(f"Ошибка экспорта PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка экспорта: {str(e)}")


@app.post("/export/tex")
async def export_tex(segments_data: List[Dict]):
    """Новый эндпоинт для экспорта LaTeX файла"""
    try:
        segments = [ClassifiedSegment(**data) for data in segments_data]

        # Генерируем LaTeX документ
        latex_content = generate_latex_document(segments)

        # Возвращаем как обычный текст
        return PlainTextResponse(
            latex_content,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=конспект.tex"}
        )

    except Exception as e:
        logger.error(f"Ошибка экспорта LaTeX: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка экспорта LaTeX: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0", "models": {
        "ocr": OPENAI_MODEL_OCR,
        "classification": OPENAI_MODEL_CLS
    }}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
