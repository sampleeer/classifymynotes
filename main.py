import os
import re
import time
import json
import base64
import mimetypes
import logging
import asyncio
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from io import BytesIO
import tempfile

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


def as_data_url_from_bytes(data: bytes, mime_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def safe_truncate(s: str, limit=12000):
    return s if len(s) <= limit else s[:limit] + "\n...[truncated]..."



async def ocr_image_with_gpt(image_data: bytes, model: str = OPENAI_MODEL_OCR, max_tokens: int = 2048) -> str:
    data_url = as_data_url_from_bytes(image_data)
    logger.info(f"OCR через {model}...")

    resp = chat_call_with_retry(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Перепиши весь текст с изображения максимально точно, включая формулы. Без анализа, только текст."},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }],
        max_tokens=max_tokens,
        temperature=0.0
    )
    text = resp.choices[0].message.content or ""
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
        html_text = text

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
                highlighted_text = f'<span class="{css_class}" title="{segment.category.title()}">{segment.text}</span>'
            else:
                highlighted_text = segment.text

            # Заменяем текст сегмента на подсвеченный
            if segment.text in html_text:
                html_text = html_text.replace(segment.text, highlighted_text, 1)

        return html_text.replace('\n', '<br>')

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
async def upload_and_process(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Пустой файл")

        mime = (file.content_type or "").lower()
        fname = (file.filename or "").lower()
        logger.info(f"Получен файл: {file.filename}, тип: {mime}, размер: {len(content)} байт")

        if mime.startswith("image/"):
            extracted_text = await ocr_image_with_gpt(content)

        elif mime == "application/pdf" or fname.endswith(".pdf"):
            extracted_text = ""

            try:
                with pdfplumber.open(BytesIO(content)) as pdf:
                    pages_text = [page.extract_text() or "" for page in pdf.pages]
                    extracted_text = "\n\n".join(t.strip() for t in pages_text if t and t.strip())
            except Exception as e:
                logger.warning(f"Не удалось извлечь текст из PDF: {e}")

            if not extracted_text.strip():
                logger.info("PDF без текстового слоя — делаем OCR по страницам…")
                images = convert_from_bytes(content, dpi=200)
                page_texts = []
                for idx, img in enumerate(images, start=1):
                    buf = BytesIO()
                    img.save(buf, format="JPEG")
                    buf.seek(0)
                    ocr_txt = await ocr_image_with_gpt(buf.getvalue())
                    page_texts.append(f"[Страница {idx}]\n{ocr_txt}")
                extracted_text = "\n\n".join(page_texts)

        elif mime == "text/plain" or fname.endswith(".txt"):
            extracted_text = content.decode("utf-8", errors="replace")

        else:
            raise HTTPException(
                status_code=400,
                detail="Поддерживаются изображения (JPEG/PNG), PDF и текстовые файлы (.txt)"
            )

        analyzer = GPTNoteAnalyzer()
        segments = analyzer.analyze_text(extracted_text)
        highlighted_html = analyzer.create_highlighted_html(extracted_text, segments)
        summary = analyzer.summary_text(extracted_text, segments)

        return ProcessingResult(
            success=True,
            extracted_text=extracted_text,
            segments=[asdict(s) for s in segments],
            highlighted_html=highlighted_html,
            summary=summary,
            file_info={"filename": file.filename, "size": len(content), "type": mime},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обработки файла: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


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


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0", "models": {
        "ocr": OPENAI_MODEL_OCR,
        "classification": OPENAI_MODEL_CLS
    }}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
