
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from collections import Counter
import spacy
import spacy.cli
from pydantic import BaseModel
from .config import logger, OPENAI_MODEL_CLS, chat_call_with_retry

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

def load_spacy_ru():
    try:
        return spacy.load("ru_core_news_sm")
    except OSError:
        logger.info("Скачиваю spacy ru_core_news_sm...")
        spacy.cli.download("ru_core_news_sm")
        return spacy.load("ru_core_news_sm")

nlp = load_spacy_ru()
CLS_INSTRUCTIONS = ("Определи тип фрагмента текста одним словом: "
    "определение, теорема, доказательство, пример, формула, важный_факт, дата, общий_текст.\n\n")

def classify_segment_with_gpt(segment: str, model: str = OPENAI_MODEL_CLS) -> str:
    prompt = f"{CLS_INSTRUCTIONS}Фрагмент:\n\"\"\"\n{segment}\n\"\"\"\nОтвет одним словом на русском:"
    resp = chat_call_with_retry(model=model,messages=[{"role": "user", "content": prompt}],max_tokens=10,temperature=0.0)
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
            results.append(ClassifiedSegment(text=seg,category=cat,confidence=1.0,start_pos=pos,end_pos=pos + len(seg)))
            pos += len(seg) + 1
        return results

    def extract_topics(self, text: str) -> List[str]:
        topics = []
        math_keywords = {"производная": "Производные","интеграл": "Интегралы","функция": "Функции","уравнение": "Уравнения","матрица": "Матрицы","предел": "Пределы"}
        for keyword, topic in math_keywords.items():
            if keyword in text.lower() and topic not in topics:
                topics.append(topic)
        return topics[:5]

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
        return {"total_segments": len(segments),"total_chars": len(src_text),"categories": {"definitions": len(categories["definitions"]),"examples": len(categories["examples"]),"theorems": len(categories["theorems"]),"formulas": len(categories["formulas"]),"dates": len(categories["dates"])}, "topics": topics,"segments_by_category": {k: [asdict(s) for s in v] for k, v in categories.items()}}
