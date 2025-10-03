
import os
import re
import time
import base64
import logging
from typing import Optional, Dict, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_MODEL_OCR = os.getenv("OPENAI_MODEL_OCR", "gpt-4o-mini")
OPENAI_MODEL_CLS = os.getenv("OPENAI_MODEL_CLS", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
OPENAI_MODEL_OCR_PRIMARY = os.getenv("OPENAI_MODEL_OCR", "gpt-4o-mini")
OPENAI_MODEL_OCR_FALLBACK = os.getenv("OPENAI_MODEL_OCR_FALLBACK", "gpt-4o")
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ClassifyMyNotes")
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
                import time as _t
                _t.sleep(sleep_s)
                continue
            raise

OCR_MAX_TOKENS_DEFAULT = 4096
OCR_MAX_TOKENS_HIGH = 8192
OCR_MAX_TOKENS_ULTRA = 16384
OCR_TOKENS_CONFIG = {"basic": 4096,"high_quality": 8192,"ultra_quality": 16384,"tiling": 6144,"fallback": 12288}
