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
import unicodedata
import string

load_dotenv()

OPENAI_MODEL_OCR = os.getenv("OPENAI_MODEL_OCR", "gpt-4o-mini")
OPENAI_MODEL_CLS = os.getenv("OPENAI_MODEL_CLS", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

OPENAI_MODEL_OCR_PRIMARY = os.getenv("OPENAI_MODEL_OCR", "gpt-4o-mini")
OPENAI_MODEL_OCR_FALLBACK = os.getenv("OPENAI_MODEL_OCR_FALLBACK", "gpt-4o")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ClassifyMyNotes")

app = FastAPI(title="ClassifyMyNotes API", version="2.0.1")

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




def _img_to_png_bytes(img: Image.Image, scale: float = 1.0, quality: int = 95) -> bytes:
    """Сохранить как PNG с улучшенными параметрами."""
    if scale != 1.0:
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=1)
    return buf.getvalue()


def _tile_image(img: Image.Image, grid=(2, 2), overlap=10) -> List[bytes]:
    """Разбить страницу на тайлы с перекрытием."""
    W, H = img.size
    cols, rows = grid
    tiles = []
    w = W // cols
    h = H // rows

    for r in range(rows):
        for c in range(cols):
            left = max(0, c * w - (overlap if c > 0 else 0))
            top = max(0, r * h - (overlap if r > 0 else 0))
            right = min(W, (c + 1) * w + (overlap if c + 1 < cols else 0))
            bottom = min(H, (r + 1) * h + (overlap if r + 1 < rows else 0))

            crop = img.crop((left, top, right, bottom))
            tiles.append(_img_to_png_bytes(crop, scale=1.0))

    return tiles


def as_data_url_from_bytes(data: bytes, mime_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def safe_truncate(s: str, limit=12000):
    return s if len(s) <= limit else s[:limit] + "\n...[truncated]..."


def fix_latex_commands(text: str) -> str:
    """Исправляет часто встречающиеся ошибки в LaTeX командах"""
    text = re.sub(r'\\\\\s+([A-Za-z])', r'\\\1', text)
    text = re.sub(r'\\\\(\s*[a-zA-Z])', r'\\\\\n\1', text)
    return text





def _strip_latex_fences(s: str) -> str:
    """Убираем markdown code fences если модель такое вернула"""
    s = s.strip()
    s = re.sub(r"^```(?:latex)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    return s.strip()


def validate_and_fix_latex(content: str) -> str:
    """
    Проверяет и исправляет распространенные ошибки в LaTeX коде
    """
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
        return f'\\begin{{align}}\n{chr(10).join(fixed_lines)}\n\\end{{align}}'

    content = re.sub(r'\\begin\{align\}(.*?)\\end\{align\}', fix_align, content, flags=re.DOTALL)
    return content


_CYR_CONFUSABLES = {
    ord("ĸ"): "к",
    ord("ı"): "и",
    0x2019: "'",
    ord("`"): "'",
    0x201C: '"',
    0x201D: '"',
}




def _text_looks_gibberish(s: str) -> bool:
    """Более строгая проверка на мусор после OCR."""
    if not s or len(s.strip()) < 10:
        return True

    s_clean = s.strip()

    bad_markers = ("�", "□", "▯", "�")
    if any(b in s_clean for b in bad_markers):
        return True

    meaningful_chars = sum(1 for ch in s_clean if ch.isalnum() or ch.isspace() or ch in ".,!?;:-()[]{}«»\"'")
    if len(s_clean) > 0:
        meaningful_ratio = meaningful_chars / len(s_clean)
        if meaningful_ratio < 0.7:
            return True

    if len(set(s_clean)) < max(3, len(s_clean) // 20):
        return True

    weird_patterns = [
        r'[a-zA-Z]{20,}',
        r'[A-Z]{10,}',
        r'\d{15,}',
        r'[^\w\s]{10,}',
    ]

    import re
    for pattern in weird_patterns:
        if re.search(pattern, s_clean):
            return True

    mixed_script_words = 0
    words = re.findall(r'\w{3,}', s_clean)
    for word in words:
        has_latin = bool(re.search(r'[A-Za-z]', word))
        has_cyrillic = bool(re.search(r'[А-Яа-яЁё]', word))
        if has_latin and has_cyrillic:
            mixed_script_words += 1

    if len(words) > 0 and mixed_script_words / len(words) > 0.3:
        return True

    return False


OCR_MAX_TOKENS_DEFAULT = 4096
OCR_MAX_TOKENS_HIGH = 8192
OCR_MAX_TOKENS_ULTRA = 16384

OCR_TOKENS_CONFIG = {
    "basic": 4096,
    "high_quality": 8192,
    "ultra_quality": 16384,
    "tiling": 6144,
    "fallback": 12288,
}




async def ocr_page_smart(img: Image.Image, page_num: int = 1,
                                  adaptive_tokens: bool = True) -> str:
    """Улучшенная стратегия OCR с адаптивным выбором токенов."""
    logger.info(f"Начинаем умный OCR страницы {page_num} (адаптивные токены: {adaptive_tokens})...")

    img_area = img.size[0] * img.size[1]

    if adaptive_tokens:
        if img_area > 3000000:
            base_tokens = OCR_TOKENS_CONFIG["ultra_quality"]
            logger.info(f"Большое изображение ({img_area} пикселей) → {base_tokens} токенов")
        elif img_area > 1500000:
            base_tokens = OCR_TOKENS_CONFIG["high_quality"]
            logger.info(f"Среднее изображение ({img_area} пикселей) → {base_tokens} токенов")
        else:
            base_tokens = OCR_TOKENS_CONFIG["basic"]
            logger.info(f"Небольшое изображение ({img_area} пикселей) → {base_tokens} токенов")
    else:
        base_tokens = OCR_TOKENS_CONFIG["high_quality"]

    strategies = []

    png_1x = _img_to_png_bytes(img, scale=1.0)
    strategies.append((png_1x, False, OPENAI_MODEL_OCR_PRIMARY, "1.0x Plain Primary", base_tokens))

    png_13x = _img_to_png_bytes(img, scale=1.3)
    strategies.append((png_13x, False, OPENAI_MODEL_OCR_PRIMARY, "1.3x Plain Primary", base_tokens + 2048))

    strategies.append((png_1x, True, OPENAI_MODEL_OCR_PRIMARY, "1.0x LaTeX Primary", base_tokens))

    strategies.append((png_13x, True, OPENAI_MODEL_OCR_FALLBACK, "1.3x LaTeX Fallback",
                       OCR_TOKENS_CONFIG["ultra_quality"]))

    png_15x = _img_to_png_bytes(img, scale=1.5)
    strategies.append((png_15x, False, OPENAI_MODEL_OCR_FALLBACK, "1.5x Plain Fallback",
                       OCR_TOKENS_CONFIG["fallback"]))

    best_result = ""
    best_quality = 0.0

    for image_bytes, latex_mode, model, strategy_name, max_tokens in strategies:
        try:
            logger.info(f"Пробуем стратегию: {strategy_name} ({max_tokens} токенов)")
            text = await _ocr_trial(
                image_bytes,
                latex_mode=latex_mode,
                model=model,
                page_num=page_num,
                max_tokens=max_tokens
            )

            if not text.strip():
                logger.warning(f"Стратегия {strategy_name}: пустой результат")
                continue

            if _text_looks_gibberish(text):
                logger.warning(f"Стратегия {strategy_name}: результат похож на мусор")
                continue

            quality = _estimate_text_quality(text)
            logger.info(f"Стратегия {strategy_name}: качество={quality:.2f}, символов={len(text)}")

            if quality > 0.7:
                logger.info(f"✅ Отличное качество для стратегии {strategy_name}, используем её")
                return text
            elif quality > best_quality:
                best_result = text
                best_quality = quality

        except Exception as e:
            logger.error(f"Ошибка в стратегии {strategy_name}: {e}")
            continue

    if best_quality > 0.3:
        logger.info(f"✅ Используем лучший результат с качеством {best_quality:.2f}")
        return best_result

    logger.info("🔄 Все стратегии дали плохие результаты, пробуем тайлинг с большим количеством токенов...")
    try:
        tiles = _tile_image(img, grid=(2, 2), overlap=20)
        tile_parts = []

        for i, tile_bytes in enumerate(tiles):
            logger.info(f"OCR тайла {i + 1}/{len(tiles)} для страницы {page_num}")
            tile_text = await _ocr_trial(
                tile_bytes,
                latex_mode=False,
                model=OPENAI_MODEL_OCR_FALLBACK,
                page_num=f"{page_num}.{i + 1}",
                max_tokens=OCR_TOKENS_CONFIG["tiling"]
            )

            if tile_text.strip() and not _text_looks_gibberish(tile_text):
                tile_parts.append(tile_text.strip())

        if tile_parts:
            result = "\n\n".join(tile_parts)
            quality = _estimate_text_quality(result)
            logger.info(f"Тайлинг дал результат с качеством {quality:.2f}")
            if quality > 0.2:
                return result

    except Exception as e:
        logger.error(f"Ошибка тайлинга для страницы {page_num}: {e}")

    logger.warning(f"❌ Все методы OCR провалились для страницы {page_num}")
    return ""


def _estimate_text_quality(text: str) -> float:
    """Оценивает качество текста от 0 до 1."""
    if not text or len(text.strip()) < 10:
        return 0.0

    s = text.strip()

    total_chars = len(s)

    letters = sum(1 for ch in s if ch.isalpha())
    letter_ratio = letters / total_chars if total_chars > 0 else 0

    spaces = s.count(' ')
    space_ratio = spaces / total_chars if total_chars > 0 else 0
    optimal_space_ratio = 0.15
    space_score = 1.0 - abs(space_ratio - optimal_space_ratio) / optimal_space_ratio
    space_score = max(0, min(1, space_score))

    unique_chars = len(set(s.lower()))
    diversity_score = min(1.0, unique_chars / 30.0)

    import re
    words = re.findall(r'\b[а-яёА-ЯЁa-zA-Z]{3,}\b', s)
    word_count = len(words)
    word_score = min(1.0, word_count / max(1, total_chars // 10))

    weird_penalty = 0
    if re.search(r'[A-Za-z]{15,}', s):
        weird_penalty += 0.3
    if re.search(r'\d{10,}', s):
        weird_penalty += 0.2
    if re.search(r'[^\w\s]{8,}', s):
        weird_penalty += 0.3

    quality = (letter_ratio * 0.3 + space_score * 0.2 + diversity_score * 0.2 + word_score * 0.3) - weird_penalty
    return max(0.0, min(1.0, quality))





async def extract_pdf_text_smart(file_path: str) -> str:
    """Кардинально улучшенная стратегия извлечения текста из PDF."""
    logger.info(f"Начинаем улучшенное извлечение из PDF: {file_path}")

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        logger.info(f"PDF содержит {total_pages} страниц")

        text_quality_scores = []
        text_pages_data = []

        for page_idx, page in enumerate(reader.pages, 1):
            try:
                extracted = page.extract_text() or ""
                extracted_clean = normalize_text_unicode(extracted.strip())

                if extracted_clean:
                    quality = _estimate_text_quality(extracted_clean)
                    text_quality_scores.append(quality)
                    text_pages_data.append((page_idx, extracted_clean, quality))
                    logger.info(
                        f"Страница {page_idx}: текстовый слой качество={quality:.2f}, символов={len(extracted_clean)}")
                else:
                    text_quality_scores.append(0.0)
                    text_pages_data.append((page_idx, "", 0.0))
                    logger.info(f"Страница {page_idx}: текстовый слой отсутствует")

            except Exception as e:
                logger.error(f"Ошибка извлечения текста со страницы {page_idx}: {e}")
                text_quality_scores.append(0.0)
                text_pages_data.append((page_idx, "", 0.0))

        avg_text_quality = sum(text_quality_scores) / len(text_quality_scores) if text_quality_scores else 0
        good_text_pages = sum(1 for score in text_quality_scores if score > 0.5)

        logger.info(f"Средне качество текстового слоя: {avg_text_quality:.2f}")
        logger.info(f"Страниц с хорошим текстом: {good_text_pages}/{total_pages}")

        final_results = []

        if good_text_pages >= total_pages * 0.7:
            logger.info("Используем преимущественно текстовый слой")

            for page_idx, text, quality in text_pages_data:
                if quality > 0.5:
                    final_results.append((page_idx, text))
                else:
                    logger.info(f"OCR для страницы {page_idx} (плохой текстовый слой)")
                    try:
                        with open(file_path, "rb") as f:
                            pdf_bytes = f.read()


                        images = convert_from_bytes(
                            pdf_bytes,
                            dpi=300,
                            fmt='PNG',
                            first_page=page_idx,
                            last_page=page_idx
                        )

                        if images:
                            ocr_text = await ocr_page_smart(images[0], page_idx)
                            final_results.append((page_idx, ocr_text))
                        else:
                            final_results.append((page_idx, f"[Ошибка конвертации страницы {page_idx}]"))

                    except Exception as e:
                        logger.error(f"Ошибка OCR страницы {page_idx}: {e}")
                        final_results.append((page_idx, f"[Ошибка OCR страницы {page_idx}]"))

        else:
            logger.info("Текстовый слой плохого качества, переходим к полному OCR")

            try:
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()

                images = convert_from_bytes(pdf_bytes, dpi=300, fmt='PNG')

                if len(images) != total_pages:
                    logger.warning(f"Количество изображений ({len(images)}) != количеству страниц ({total_pages})")

                for page_idx in range(1, min(len(images), total_pages) + 1):
                    try:
                        img = images[page_idx - 1]
                        ocr_text = await ocr_page_smart(img, page_idx)

                        text_layer_quality = text_quality_scores[page_idx - 1] if page_idx <= len(
                            text_quality_scores) else 0
                        ocr_quality = _estimate_text_quality(ocr_text) if ocr_text else 0

                        if text_layer_quality > 0.3 and text_layer_quality > ocr_quality:
                            _, text_layer_text, _ = text_pages_data[page_idx - 1]
                            final_results.append((page_idx, text_layer_text))
                            logger.info(
                                f"Страница {page_idx}: используем текстовый слой (качество {text_layer_quality:.2f} > OCR {ocr_quality:.2f})")
                        else:
                            final_results.append((page_idx, ocr_text))
                            logger.info(
                                f"Страница {page_idx}: используем OCR (качество {ocr_quality:.2f} > текстовый слой {text_layer_quality:.2f})")

                    except Exception as e:
                        logger.error(f"Ошибка обработки страницы {page_idx}: {e}")
                        final_results.append((page_idx, f"[Ошибка обработки страницы {page_idx}]"))

            except Exception as e:
                logger.error(f"Критическая ошибка при полном OCR: {e}")
                for page_idx, text, quality in text_pages_data:
                    final_results.append(
                        (page_idx, text if quality > 0.1 else f"[Ошибка извлечения страницы {page_idx}]"))

        result_parts = []
        total_extracted_chars = 0

        for page_idx, page_text in sorted(final_results):
            if page_text and page_text.strip():
                result_parts.append(f"% Страница {page_idx}\n{page_text.strip()}")
                total_extracted_chars += len(page_text.strip())
            else:
                result_parts.append(f"% Страница {page_idx}\n[Не удалось извлечь текст]")

        final_text = "\n\n".join(result_parts)

        logger.info(f"Извлечение завершено:")
        logger.info(f"  - Всего страниц: {total_pages}")
        logger.info(f"  - Извлечено символов: {total_extracted_chars}")
        logger.info(f"  - Средний размер страницы: {total_extracted_chars // max(1, total_pages)} символов")

        return final_text

    except Exception as e:
        logger.error(f"Критическая ошибка извлечения PDF: {e}")
        try:
            logger.info("Пробуем базовый OCR как последнее средство...")
            return await ocr_pdf_with_gpt(file_path)
        except Exception as e2:
            logger.error(f"Даже базовый OCR провалился: {e2}")
            return f"Критическая ошибка обработки PDF: {str(e)}"


async def ocr_pdf_with_gpt(pdf_input, mode: str = "latex") -> str:
    logger.info("Начинаем улучшенный полный OCR PDF...")

    try:
        if isinstance(pdf_input, (bytes, bytearray)):
            data = pdf_input
        else:
            with open(pdf_input, "rb") as f:
                data = f.read()

        images = convert_from_bytes(data, dpi=300, fmt='PNG')
        logger.info(f"PDF конвертирован в {len(images)} изображений")

        page_results = []

        semaphore = asyncio.Semaphore(3)

        async def process_page(idx: int, img: Image.Image) -> Tuple[int, str]:
            async with semaphore:
                page_num = idx + 1
                try:
                    text = await ocr_page_smart(img, page_num)
                    quality = _estimate_text_quality(text) if text else 0

                    if quality > 0.3:
                        logger.info(f"Страница {page_num}: успешно обработана (качество {quality:.2f})")
                        return (page_num, text)
                    else:
                        logger.warning(f"Страница {page_num}: низкое качество ({quality:.2f}), помечаем как проблемную")
                        return (page_num, f"[Низкое качество распознавания: {len(text)} символов]")

                except Exception as e:
                    logger.error(f"Ошибка обработки страницы {page_num}: {e}")
                    return (page_num, f"[Ошибка OCR: {str(e)}]")

        tasks = [process_page(idx, img) for idx, img in enumerate(images)]
        results = await asyncio.gather(*tasks)

        results.sort(key=lambda x: x[0])

        page_texts = []
        successful_pages = 0

        for page_num, text in results:
            if text and not text.startswith("["):
                page_texts.append(f"% Страница {page_num}\n{text}")
                successful_pages += 1
            else:
                page_texts.append(f"% Страница {page_num}\n{text}")

        final_result = "\n\n".join(page_texts)

        logger.info(f"Полный OCR завершен:")
        logger.info(f"  - Всего страниц: {len(images)}")
        logger.info(f"  - Успешно обработано: {successful_pages}")
        logger.info(f"  - Итого символов: {len(final_result)}")

        return final_result

    except Exception as e:
        logger.error(f"Критическая ошибка полного OCR: {e}")
        return f"Ошибка полного OCR: {str(e)}"


def normalize_cyrillic_confusables(s: str) -> str:
    replacements = {
        "ĸ": "к",
        "ı": "и",
        "'": "'",
        "`": "'",
        """: '"',   
        """: '"',
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def normalize_text_unicode(s: str) -> str:
    """
    Нормализует текст: убирает markdown-фенсы, приводит к NFC
    и заменяет частые псевдо-кириллические символы.
    """
    s = s or ""
    try:
        s = _strip_latex_fences(s)
    except NameError:
        pass
    s = unicodedata.normalize("NFC", s)
    s = normalize_cyrillic_confusables(s)
    return s


def _page_is_gibberish(s: str) -> bool:
    """Улучшенная проверка страницы на мусор."""
    if not s or len(s.strip()) < 20:
        return True

    s_clean = s.strip()

    if len(s_clean) < 100:
        # Ищем хотя бы несколько букв подряд
        words = re.findall(r'[а-яёА-ЯЁa-zA-Z]{3,}', s_clean)
        if len(words) >= 2:
            return False

    # Проверка на слишком много "плохих" символов
    allowed = set(string.ascii_letters + string.digits + string.whitespace +
                  "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ" +
                  ".,;:!?-–—()[]{}«»\"'/%+*=<>|\\_^$№±~@#&")

    bad_chars = sum(1 for ch in s_clean if ch not in allowed)
    bad_ratio = bad_chars / max(1, len(s_clean))

    if bad_ratio > 0.4:  # было 0.35, увеличили порог
        return True

    # Проверка микса латиницы и кириллицы - более мягкая
    mixed_words = 0
    for w in re.findall(r"\w{3,}", s_clean):  # только слова длиннее 3 символов
        has_lat = bool(re.search(r"[A-Za-z]", w))
        has_cyr = bool(re.search(r"[А-Яа-яЁё]", w))
        if has_lat and has_cyr:
            mixed_words += 1

    if mixed_words >= 10:  # было 5, увеличили порог
        return True

    # Проверка количества кириллицы - для русских текстов
    cyr_chars = sum(1 for ch in s_clean if "А" <= ch <= "я" or ch in "Ёё")
    if len(s_clean) > 200 and cyr_chars < max(10, len(s_clean) // 120):  # было //80
        return True

    return False


async def ocr_page_with_fallback(img: Image.Image, page_num: int = 1) -> str:
    """OCR с fallback к альтернативным методам если GPT отказывается"""

    # Сначала пробуем основной метод
    result = await ocr_page_smart(img, page_num, adaptive_tokens=True)

    if not result or len(result.strip()) < 10:
        logger.warning(f"Основной OCR не дал результата для страницы {page_num}, пробуем альтернативы")

        # Альтернативный подход с более простой инструкцией
        try:
            png_bytes = _img_to_png_bytes(img, scale=1.0)

            simple_instruction = "Please describe what you see in this image, focusing on any text content."

            resp = chat_call_with_retry(
                model=OPENAI_MODEL_OCR_PRIMARY,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": simple_instruction},
                        {"type": "image_url", "image_url": {"url": as_data_url_from_bytes(png_bytes, "image/png")}}
                    ]
                }],
                max_tokens=4096,
                temperature=0.3
            )

            alternative_text = resp.choices[0].message.content or ""
            if alternative_text and "can't help" not in alternative_text.lower():
                logger.info(
                    f"Альтернативный OCR дал результат для страницы {page_num}: {len(alternative_text)} символов")
                return alternative_text

        except Exception as e:
            logger.error(f"Альтернативный OCR тоже не сработал для страницы {page_num}: {e}")

    return result


async def _ocr_trial(image_bytes: bytes, *, latex_mode: bool, model: str,
                              lang_hint: str = "русский, английский",
                              page_num: int = 1,
                              max_tokens: int = OCR_MAX_TOKENS_HIGH) -> str:
    """Улучшенный OCR trial с настраиваемым лимитом токенов и мягкими инструкциями."""
    try:
        if latex_mode:
            instruction = f"""Пожалуйста, помогите мне прочитать текст с этого изображения и представить его в LaTeX формате.

Что мне нужно:
- Прочитать весь видимый текст на странице {page_num}
- Вернуть только содержимое (без преамбулы документа)
- Использовать LaTeX разметку для математических формул
- Сохранить структуру: заголовки, абзацы, списки

Основные языки текста: {lang_hint}

Если некоторые символы неразборчивы, это нормально - просто пропустите их."""

        else:
            instruction = f"""Пожалуйста, помогите мне прочитать весь текст с этого изображения.

Что мне нужно:
- Извлечь весь видимый текст со страницы {page_num}
- Сохранить структуру документа и переносы строк
- Математические формулы можно оставить в LaTeX формате ($...$)
- Не добавлять комментарии или пояснения

Основные языки: {lang_hint}

Если что-то неразборчиво, просто пропустите эти части."""

        logger.info(f"OCR страницы {page_num}: используем {max_tokens} токенов (модель: {model})")

        resp = chat_call_with_retry(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": as_data_url_from_bytes(image_bytes, "image/png")}}
                ]
            }],
            max_tokens=max_tokens,
            temperature=0.1
        )

        text = resp.choices[0].message.content or ""

        # Проверяем на стандартные отказы
        refusal_patterns = [
            "не могу помочь",
            "не могу выполнить",
            "не могу обработать",
            "sorry, i can't",
            "i can't help",
            "i cannot assist",
            "против политик",
            "против правил"
        ]

        text_lower = text.lower()
        if any(pattern in text_lower for pattern in refusal_patterns):
            logger.warning(f"Модель отказалась выполнять OCR для страницы {page_num}: {text[:100]}")
            return ""

        # Логируем использование токенов
        if hasattr(resp, 'usage') and resp.usage:
            prompt_tokens = getattr(resp.usage, 'prompt_tokens', 0)
            completion_tokens = getattr(resp.usage, 'completion_tokens', 0)
            total_tokens = getattr(resp.usage, 'total_tokens', 0)
            logger.info(f"Использовано токенов: prompt={prompt_tokens}, completion={completion_tokens}, "
                        f"total={total_tokens}, лимит={max_tokens}")

            # Предупреждение если достигли лимита
            if completion_tokens >= max_tokens * 0.95:
                logger.warning(f"⚠️ Почти достигнут лимит токенов! Возможно, текст обрезан. "
                               f"Рекомендуется увеличить max_tokens для страницы {page_num}")

        if latex_mode:
            text = _strip_latex_fences(text)

        text = normalize_text_unicode(text)

        # Логируем качество результата
        quality = _estimate_text_quality(text)
        logger.info(f"OCR страницы {page_num}: качество={quality:.2f}, символов={len(text)}")

        return text

    except Exception as e:
        logger.error(f"OCR trial failed для страницы {page_num}: {e}")
        return ""



async def ocr_image_with_gpt(
        image_data: bytes,
        model: str = OPENAI_MODEL_OCR,
        max_tokens: int = OCR_TOKENS_CONFIG["high_quality"],  # ← Увеличенный лимит
        latex_mode: bool = False
) -> str:
    """OCR изображения через OpenAI Vision-модель с увеличенными токенами"""
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
                    text=False,
                    timeout=timeout,
                    check=False,
                    env=env,
                )
                stdout = (result.stdout or b"").decode("utf-8", errors="replace")
                stderr = (result.stderr or b"").decode("utf-8", errors="replace")

                if result.returncode != 0:
                    ok = False
                    last_err = f"[{eng} pass {i + 1}] rc={result.returncode}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
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
            last_err = f"{eng} unexpected error: {e}"
            logger.exception(last_err)

    return None, last_err


_MATH_TRIGGERS = (
    r"\\int", r"\\frac", r"\\sum", r"\\prod", r"\\lim", r"\\sqrt",
    r"\\sin", r"\\cos", r"\\tan", r"\\log", r"\\ln",
    r"\\alpha", r"\\beta", r"\\gamma", r"\\infty", r"\\left", r"\\right",
)


def _strip_fake_single_letter_cmds(s: str) -> str:
    s = re.sub(r"\\([a-zA-Z])(?=\s*[\(\[\{])", r"\1", s)
    s = re.sub(r"\\([a-zA-Z])(\b)", r"\1\2", s)
    return s


def _line_needs_math(line: str) -> bool:
    if "$" in line or r"\[" in line or r"\(" in line:
        return False
    for p in _MATH_TRIGGERS:
        if re.search(p, line):
            return True
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

    if s.count("$") % 2 == 1:
        s = s.replace("$", "")

    opens = len(re.findall(r"\\\[", s))
    closes = len(re.findall(r"\\\]", s))
    if opens > closes:
        s += ("\n" + "\\]") * (opens - closes)

    s = re.sub(r'\\begin\{align\*?\}([\s\S]*?)\\end\{align\*?\}', r'\\begin{equation}\1\\end{equation}', s)

    s = re.sub(r'\\(write|input|include)\b.*', '', s)

    s = _strip_fake_single_letter_cmds(s)

    return s


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
    logger.info("=== НАЧАЛО ОБРАБОТКИ ===")
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

        logger.info(f"Извлеченный текст: {len(extracted_text)} символов")
        logger.info(f"Первые 500 символов: {extracted_text[:500]}")

        analyzer = GPTNoteAnalyzer()
        segments = analyzer.analyze_text(extracted_text)
        highlighted_html = analyzer.create_highlighted_html(extracted_text, segments)
        summary = analyzer.summary_text(extracted_text, segments)

        if extracted_text.strip().startswith('\\documentclass') or '\\begin{document}' in extracted_text:
            document_content = extract_content_from_latex(extracted_text)
        else:
            document_content = extracted_text

        healed = heal_latex_content(document_content)
        healed = auto_wrap_display_math(healed)
        clean_content = fix_latex_commands(healed)

        full_latex_source = generate_latex_document(segments, clean_content)

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

        pdf_path, build_err = try_build_latex_pdf(tex_path, tmpdir)

        pdf_b64 = None
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                pdf_b64 = base64.b64encode(pdf_file.read()).decode("ascii")
        else:
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

        logger.info("=== ЗАВЕРШЕНИЕ ОБРАБОТКИ ===")
        return {
            "success": True,
            "extracted_text": extracted_text,
            "segments": [asdict(s) for s in segments],
            "highlighted_html": highlighted_html,
            "summary": summary,
            "pdf_base64": pdf_b64,
            "latex_source": full_latex_source,
            "file_info": {
                "filename": file.filename,
                "original_text_length": len(extracted_text),
                "processed_text_length": len(clean_content),
                "segments_count": len(segments),
                "latex_build_ok": bool(pdf_b64)
            }
        }


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

        latex_content = generate_latex_document(segments)

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
    return {"status": "healthy", "version": "2.0.1", "models": {
        "ocr": OPENAI_MODEL_OCR,
        "classification": OPENAI_MODEL_CLS
    }}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)