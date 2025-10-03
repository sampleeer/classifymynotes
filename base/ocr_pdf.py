
import os
import re
import base64
import unicodedata
import string
import asyncio
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from .config import logger, OPENAI_MODEL_OCR, OPENAI_MODEL_OCR_PRIMARY, OPENAI_MODEL_OCR_FALLBACK, OCR_TOKENS_CONFIG, chat_call_with_retry

def _img_to_png_bytes(img: Image.Image, scale: float = 1.0, quality: int = 95) -> bytes:
    if scale != 1.0:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=1)
    return buf.getvalue()

def _tile_image(img: Image.Image, grid=(2, 2), overlap=10) -> List[bytes]:
    W, H = img.size
    cols, rows = grid
    tiles = []
    w = W // cols; h = H // rows
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

def _strip_latex_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:latex)?\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s*```$", "", s, flags=re.MULTILINE)
    return s.strip()

def normalize_cyrillic_confusables(s: str) -> str:
    replacements = {"ĸ": "к","ı": "и","'": "'","`": "'", "\u201C": '"', "\u201D": '"'}
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s

def normalize_text_unicode(s: str) -> str:
    s = s or ""
    try:
        s = _strip_latex_fences(s)
    except NameError:
        pass
    s = unicodedata.normalize("NFC", s)
    s = normalize_cyrillic_confusables(s)
    return s

def _text_looks_gibberish(s: str) -> bool:
    if not s or len(s.strip()) < 10: return True
    s_clean = s.strip()
    bad_markers = ("�","□","▯","�")
    if any(b in s_clean for b in bad_markers): return True
    meaningful_chars = sum(1 for ch in s_clean if ch.isalnum() or ch.isspace() or ch in ".,!?;:-()[]{}«»\"'")
    if len(s_clean)>0 and meaningful_chars/len(s_clean) < 0.7: return True
    if len(set(s_clean)) < max(3, len(s_clean)//20): return True
    weird_patterns = [r'[a-zA-Z]{20,}', r'[A-Z]{10,}', r'\d{15,}', r'[^\w\s]{10,}']
    import re
    for p in weird_patterns:
        if re.search(p, s_clean): return True
    mixed_script_words = 0
    words = re.findall(r'\w{3,}', s_clean)
    for w in words:
        has_lat = bool(re.search(r'[A-Za-z]', w))
        has_cyr = bool(re.search(r'[А-Яа-яЁё]', w))
        if has_lat and has_cyr: mixed_script_words += 1
    if len(words)>0 and mixed_script_words/len(words) > 0.3: return True
    return False

def _estimate_text_quality(text: str) -> float:
    if not text or len(text.strip()) < 10: return 0.0
    s = text.strip(); total = len(s)
    letters = sum(1 for ch in s if ch.isalpha()); letter_ratio = letters/total if total>0 else 0
    spaces = s.count(' '); space_ratio = spaces/total if total>0 else 0
    optimal_space_ratio = 0.15; space_score = 1.0 - abs(space_ratio - optimal_space_ratio)/optimal_space_ratio
    space_score = max(0, min(1, space_score))
    unique_chars = len(set(s.lower())); diversity_score = min(1.0, unique_chars/30.0)
    import re
    words = re.findall(r'\b[а-яёА-ЯЁa-zA-Z]{3,}\b', s)
    word_count = len(words); word_score = min(1.0, word_count / max(1, total//10))
    weird_penalty = 0
    if re.search(r'[A-Za-z]{15,}', s): weird_penalty += 0.3
    if re.search(r'\d{10,}', s): weird_penalty += 0.2
    if re.search(r'[^\w\s]{8,}', s): weird_penalty += 0.3
    quality = (letter_ratio*0.3 + space_score*0.2 + diversity_score*0.2 + word_score*0.3) - weird_penalty
    return max(0.0, min(1.0, quality))

async def _ocr_trial(image_bytes: bytes, *, latex_mode: bool, model: str, lang_hint: str = "русский, английский", page_num: int = 1, max_tokens: int = 8192) -> str:
    try:
        if latex_mode:
            instruction = f"""Пожалуйста, помогите мне прочитать текст с этого изображения и представить его в LaTeX формате.

- Прочитать весь видимый текст на странице {page_num}
- Вернуть только содержимое (без преамбулы документа)
- Использовать LaTeX разметку для формул
- Сохранить структуру: заголовки, абзацы, списки
Языки: {lang_hint}"""
        else:
            instruction = f"""Пожалуйста, прочитай весь текст с этого изображения (страница {page_num}) без комментариев.
Сохраняй структуру и переносы. Формулы можно оставить в $...$. Языки: {lang_hint}"""
        logger.info(f"OCR страницы {page_num}: модель={model}, токенов={max_tokens}")
        resp = chat_call_with_retry(model=model,messages=[{"role": "user","content":[{"type": "text", "text": instruction},{"type": "image_url", "image_url": {"url": as_data_url_from_bytes(image_bytes, "image/png")}}]}],max_tokens=max_tokens,temperature=0.1)
        text = resp.choices[0].message.content or ""
        refusal_patterns = ["не могу помочь","не могу выполнить","sorry, i can't","i can't help","i cannot assist"]
        if any(p in (text.lower()) for p in refusal_patterns): return ""
        if latex_mode:
            text = _strip_latex_fences(text)
        text = normalize_text_unicode(text)
        quality = _estimate_text_quality(text)
        logger.info(f"OCR стр. {page_num}: качество={quality:.2f}, символов={len(text)}")
        return text
    except Exception as e:
        logger.error(f"OCR trial failed для страницы {page_num}: {e}")
        return ""

async def ocr_page_smart(img: Image.Image, page_num: int = 1, adaptive_tokens: bool = True) -> str:
    logger.info(f"Начинаем умный OCR страницы {page_num} (адаптивные токены={adaptive_tokens})")
    img_area = img.size[0] * img.size[1]
    if adaptive_tokens:
        if img_area > 3_000_000: base_tokens = OCR_TOKENS_CONFIG["ultra_quality"]
        elif img_area > 1_500_000: base_tokens = OCR_TOKENS_CONFIG["high_quality"]
        else: base_tokens = OCR_TOKENS_CONFIG["basic"]
    else:
        base_tokens = OCR_TOKENS_CONFIG["high_quality"]
    strategies = []
    png_1x = _img_to_png_bytes(img, scale=1.0)
    png_13x = _img_to_png_bytes(img, scale=1.3)
    png_15x = _img_to_png_bytes(img, scale=1.5)
    strategies.append((png_1x, False, OPENAI_MODEL_OCR_PRIMARY, "1.0x Plain Primary", base_tokens))
    strategies.append((png_13x, False, OPENAI_MODEL_OCR_PRIMARY, "1.3x Plain Primary", base_tokens + 2048))
    strategies.append((png_1x, True,  OPENAI_MODEL_OCR_PRIMARY, "1.0x LaTeX Primary", base_tokens))
    strategies.append((png_13x, True, OPENAI_MODEL_OCR_FALLBACK, "1.3x LaTeX Fallback", OCR_TOKENS_CONFIG["ultra_quality"]))
    strategies.append((png_15x, False, OPENAI_MODEL_OCR_FALLBACK, "1.5x Plain Fallback", OCR_TOKENS_CONFIG["fallback"]))
    best_result = ""; best_quality = 0.0
    for image_bytes, latex_mode, model, name, max_tokens in strategies:
        try:
            text = await _ocr_trial(image_bytes, latex_mode=latex_mode, model=model, page_num=page_num, max_tokens=max_tokens)
            if not text.strip(): continue
            if _text_looks_gibberish(text): continue
            q = _estimate_text_quality(text)
            logger.info(f"Стратегия {name}: качество={q:.2f}")
            if q > 0.7: return text
            if q > best_quality: best_result, best_quality = text, q
        except Exception as e:
            logger.error(f"Ошибка стратегии {name}: {e}"); continue
    if best_quality > 0.3: return best_result
    try:
        tiles = _tile_image(img, grid=(2,2), overlap=20); tile_parts = []
        for i, tb in enumerate(tiles):
            logger.info(f"OCR тайла {i+1}/{len(tiles)} стр. {page_num}")
            t = await _ocr_trial(tb, latex_mode=False, model=OPENAI_MODEL_OCR_FALLBACK,page_num=int(f"{page_num}{i+1}"), max_tokens=OCR_TOKENS_CONFIG["tiling"])
            if t.strip() and not _text_looks_gibberish(t): tile_parts.append(t.strip())
        if tile_parts:
            result = "\n\n".join(tile_parts); q = _estimate_text_quality(result)
            if q > 0.2: return result
    except Exception as e:
        logger.error(f"Ошибка тайлинга для страницы {page_num}: {e}")
    return ""

async def ocr_image_with_gpt(image_data: bytes, model: str = OPENAI_MODEL_OCR, max_tokens: int = 8192, latex_mode: bool = False) -> str:
    data_url = as_data_url_from_bytes(image_data)
    if latex_mode:
        instruction = ("Распознай весь текст с изображения и верни его в LaTeX формате. Возвращай ТОЛЬКО содержимое, без преамбулы.")
    else:
        instruction = "Перепиши весь текст с изображения максимально точно, без комментариев."
    resp = chat_call_with_retry(model=model,messages=[{"role":"user","content":[{"type":"text","text":instruction},{"type":"image_url","image_url":{"url": data_url}}]}],max_tokens=max_tokens, temperature=0.0)
    text = resp.choices[0].message.content or ""
    if latex_mode: text = _strip_latex_fences(text)
    return text.strip()

async def extract_pdf_text_smart(file_path: str) -> str:
    logger.info(f"Начинаем улучшенное извлечение из PDF: {file_path}")
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        logger.info(f"PDF содержит {total_pages} страниц")
        text_quality_scores = []; text_pages_data = []
        for page_idx, page in enumerate(reader.pages, 1):
            try:
                extracted = page.extract_text() or ""
                extracted_clean = normalize_text_unicode(extracted.strip())
                if extracted_clean:
                    quality = _estimate_text_quality(extracted_clean)
                    text_quality_scores.append(quality)
                    text_pages_data.append((page_idx, extracted_clean, quality))
                    logger.info(f"Стр. {page_idx}: текстовый слой качество={quality:.2f}, len={len(extracted_clean)}")
                else:
                    text_quality_scores.append(0.0); text_pages_data.append((page_idx, "", 0.0))
                    logger.info(f"Стр. {page_idx}: текстовый слой отсутствует")
            except Exception as e:
                logger.error(f"Ошибка извлечения текста со стр. {page_idx}: {e}")
                text_quality_scores.append(0.0); text_pages_data.append((page_idx, "", 0.0))
        avg_text_quality = sum(text_quality_scores)/len(text_quality_scores) if text_quality_scores else 0
        good_text_pages = sum(1 for s in text_quality_scores if s>0.5)
        logger.info(f"Среднее качество текстового слоя: {avg_text_quality:.2f}; хороших страниц: {good_text_pages}/{total_pages}")
        final_results = []
        if good_text_pages >= total_pages * 0.7:
            logger.info("Используем преимущественно текстовый слой")
            for page_idx, text, quality in text_pages_data:
                if quality > 0.5:
                    final_results.append((page_idx, text))
                else:
                    try:
                        with open(file_path, "rb") as f: pdf_bytes = f.read()
                        images = convert_from_bytes(pdf_bytes, dpi=300, fmt='PNG', first_page=page_idx, last_page=page_idx)
                        if images:
                            ocr_text = await ocr_page_smart(images[0], page_idx)
                            final_results.append((page_idx, ocr_text))
                        else:
                            final_results.append((page_idx, f"[Ошибка конвертации стр. {page_idx}]"))
                    except Exception as e:
                        logger.error(f"Ошибка OCR стр. {page_idx}: {e}")
                        final_results.append((page_idx, f"[Ошибка OCR стр. {page_idx}]"))
        else:
            logger.info("Текстовый слой плохого качества, переходим к полному OCR")
            try:
                with open(file_path, "rb") as f: pdf_bytes = f.read()
                images = convert_from_bytes(pdf_bytes, dpi=300, fmt='PNG')
                if len(images) != total_pages:
                    logger.warning(f"Количество изображений ({len(images)}) != страниц ({total_pages})")
                for page_idx in range(1, min(len(images), total_pages)+1):
                    try:
                        img = images[page_idx-1]
                        ocr_text = await ocr_page_smart(img, page_idx)
                        text_layer_quality = text_quality_scores[page_idx-1] if page_idx<=len(text_quality_scores) else 0
                        ocr_quality = _estimate_text_quality(ocr_text) if ocr_text else 0
                        if text_layer_quality > 0.3 and text_layer_quality > ocr_quality:
                            _, text_layer_text, _ = text_pages_data[page_idx-1]
                            final_results.append((page_idx, text_layer_text))
                        else:
                            final_results.append((page_idx, ocr_text))
                    except Exception as e:
                        logger.error(f"Ошибка обработки стр. {page_idx}: {e}")
                        final_results.append((page_idx, f"[Ошибка обработки стр. {page_idx}]"))
            except Exception as e:
                logger.error(f"Критическая ошибка при полном OCR: {e}")
                for page_idx, text, quality in text_pages_data:
                    final_results.append((page_idx, text if quality>0.1 else f"[Ошибка извлечения стр. {page_idx}]"))
        parts = []
        for page_idx, page_text in sorted(final_results):
            parts.append(f"% Страница {page_idx}\n{page_text.strip() if page_text and page_text.strip() else '[Не удалось извлечь текст]'}")
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"Критическая ошибка извлечения PDF: {e}")
        try:
            return await ocr_pdf_with_gpt(file_path)
        except Exception as e2:
            logger.error(f"Даже базовый OCR провалился: {e2}")
            return f"Критическая ошибка обработки PDF: {str(e)}"

async def ocr_pdf_with_gpt(pdf_input, mode: str = "latex") -> str:
    logger.info("Начинаем улучшенный полный OCR PDF...")
    try:
        if isinstance(pdf_input, (bytes, bytearray)): data = pdf_input
        else:
            with open(pdf_input, "rb") as f: data = f.read()
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
                        return (page_num, f"[Низкое качество распознавания: {len(text)} символов]")
                except Exception as e:
                    logger.error(f"Ошибка обработки страницы {page_num}: {e}")
                    return (page_num, f"[Ошибка OCR: {str(e)}]")
        tasks = [process_page(idx, img) for idx, img in enumerate(images)]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda x: x[0])
        page_texts = []
        for page_num, text in results:
            page_texts.append(f"% Страница {page_num}\n{text}")
        return "\n\n".join(page_texts)
    except Exception as e:
        logger.error(f"Критическая ошибка полного OCR: {e}")
        return f"Ошибка полного OCR: {str(e)}"

async def ask_llm_about_notes(context_text: str, question: str, model: str = OPENAI_MODEL_OCR) -> str:
    from .config import OPENAI_MODEL_CLS
    prompt = ("Ниже учебные заметки. Отвечай кратко и по делу, "
        "ссылайся на определения/теоремы из контекста, если уместно.\n\n"
        f"КОНТЕКСТ:\n{safe_truncate(context_text)}\n\nВОПРОС: {question}")
    resp = chat_call_with_retry(model=OPENAI_MODEL_CLS,messages=[{"role": "user", "content": prompt}],max_tokens=600,temperature=0.2)
    return (resp.choices[0].message.content or "").strip()
