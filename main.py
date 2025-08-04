!pip install torch torchvision torchaudio
!pip install transformers
!pip install pillow
!pip install opencv-python
!pip install matplotlib
!pip install "numpy<2.0"



from openai import OpenAI
import base64
import spacy
from collections import Counter
from dataclasses import dataclass
import logging
from typing import List, Tuple

client = OpenAI(api_key="") 
nlp = spacy.load("ru_core_news_sm")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClassifiedSegment:
    text: str
    category: str
    confidence: float
    start_pos: int
    end_pos: int

def classify_segment_with_gpt(segment: str) -> str:
    prompt = f"""Определи, что это за фрагмент текста: определение, теорема, пример, формула, важный факт, дата/событие или общий текст.

Фрагмент:
\"{segment}\"

Ответ одним словом на русском:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()

class GPTNoteAnalyzer:
    def __init__(self):
        pass

    def split_into_segments(self, text: str) -> List[str]:
        return [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]

    def analyze_text(self, text: str) -> List[ClassifiedSegment]:
        logger.info("Анализ текста через GPT...")
        segments = self.split_into_segments(text)
        logger.info(f"Найдено {len(segments)} сегментов")
        results = []
        pos = 0
        for seg in segments:
            category = classify_segment_with_gpt(seg)
            results.append(ClassifiedSegment(
                text=seg,
                category=category,
                confidence=1.0,
                start_pos=pos,
                end_pos=pos + len(seg)
            ))
            pos += len(seg) + 1
        return results

    def print_summary(self, segments: List[ClassifiedSegment]):
        print("\n📊 Сводка:")
        counts = Counter([s.category for s in segments])
        for cat, cnt in counts.items():
            print(f"{cat}: {cnt}")
        print("\nПримеры:")
        for s in segments[:3]:
            print(f"[{s.category} | {s.confidence:.2f}] {s.text[:100]}...")

def process_image(image_path: str):
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Что написано на этом изображении? Перепиши весь текст, включая формулы, без анализа."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
    )
    extracted_text = response.choices[0].message.content
    print("\nПолученный текст:\n", extracted_text)

    analyzer = GPTNoteAnalyzer()
    segments = analyzer.analyze_text(extracted_text)
    analyzer.print_summary(segments)

if __name__ == "__main__":
    process_image("komplan.jpg")
