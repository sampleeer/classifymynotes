
import os
import tempfile
import base64
from typing import List, Dict
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse, PlainTextResponse
from .config import logger, OPENAI_MODEL_OCR, OPENAI_MODEL_CLS
from .analysis_cls import ClassifiedSegment, ChatMessage, ChatResponse, GPTNoteAnalyzer
from .latex_tools import convert_latex_to_html, extract_content_from_latex, heal_latex_content, auto_wrap_display_math, validate_and_fix_latex, generate_latex_document, try_build_latex_pdf, simplify_latex_content, export_study_pack_pdf
from .ocr_pdf import extract_pdf_text_smart, ask_llm_about_notes

app = FastAPI(title="ClassifyMyNotes API", version="2.0.1")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

if os.path.isdir("static"):
    app.mount("/web", StaticFiles(directory="static", html=True), name="web")

@app.get("/")
def root():
    if os.path.isdir("static"):
        return RedirectResponse(url="/web/")
    return JSONResponse({"message": "ClassifyMyNotes API is running"})

@app.post("/upload")
async def upload_and_process(file: UploadFile):
    logger.info("=== НАЧАЛО ОБРАБОТКИ ===")
    logger.info(f"Получен файл: {file.filename}, тип: {file.content_type}")
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        extracted_text = ""
        if file.filename.lower().endswith(".pdf"):
            logger.info("PDF → smart extract (текстовый слой + OCR-фоллбек постранично)…")
            extracted_text = await extract_pdf_text_smart(file_path)
        logger.info(f"Извлеченный текст: {len(extracted_text)} символов")
        analyzer = GPTNoteAnalyzer()
        segments = analyzer.analyze_text(extracted_text)
        highlighted_html = convert_latex_to_html(extracted_text)
        summary = analyzer.summary_text(extracted_text, segments)
        if extracted_text.strip().startswith('\\documentclass') or '\\begin{document}' in extracted_text:
            document_content = extract_content_from_latex(extracted_text)
        else:
            document_content = extracted_text
        healed = heal_latex_content(document_content)
        healed = auto_wrap_display_math(healed)
        clean_content = validate_and_fix_latex(healed)
        full_latex_source = generate_latex_document(segments, clean_content)
        tex_path = os.path.join(tmpdir, "output.tex")
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
        with open(tex_path, "w", encoding="utf-8") as tex_file:
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
            with open(tex_path, "w", encoding="utf-8") as tex_file:
                tex_file.write(simplified_template)
            pdf_path2, build_err2 = try_build_latex_pdf(tex_path, tmpdir)
            if pdf_path2 and os.path.exists(pdf_path2):
                with open(pdf_path2, "rb") as pdf_file:
                    pdf_b64 = base64.b64encode(pdf_file.read()).decode("ascii")
            else:
                logger.error("LaTeX build failed, returning without pdf_base64.\nFirst error:\n%s\nSecond error:\n%s",build_err, build_err2)
        logger.info("=== ЗАВЕРШЕНИЕ ОБРАБОТКИ ===")
        return {"success": True,"extracted_text": extracted_text,"segments": [s.__dict__ for s in segments],"highlighted_html": highlighted_html,"summary": summary,"pdf_base64": pdf_b64,"latex_source": full_latex_source,"file_info": {"filename": file.filename,"original_text_length": len(extracted_text),"processed_text_length": len(clean_content),"segments_count": len(segments),"latex_build_ok": bool(pdf_b64)}}

@app.post("/chat")
async def chat_with_ai(message: ChatMessage):
    try:
        response = await ask_llm_about_notes(message.context or "", message.message)
        suggestions = ["Можешь объяснить это проще?","Приведи еще один пример","Как это связано с другими темами?","Какие могут быть практические применения?"]
        return ChatResponse(response=response, suggestions=suggestions)
    except Exception as e:
        logger.error(f"Ошибка чата: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка чата: {str(e)}")

@app.post("/export/pdf")
async def export_pdf(segments_data: List[Dict]):
    try:
        segments = [ClassifiedSegment(**data) for data in segments_data]
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = export_study_pack_pdf(segments, tmp.name)
        return FileResponse(pdf_path, media_type="application/pdf", filename="классифицированный_конспект.pdf")
    except Exception as e:
        logger.error(f"Ошибка экспорта PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка экспорта: {str(e)}")

@app.post("/export/tex")
async def export_tex(segments_data: List[Dict]):
    try:
        segments = [ClassifiedSegment(**data) for data in segments_data]
        latex_content = generate_latex_document(segments)
        return PlainTextResponse(latex_content, media_type="text/plain; charset=utf-8",headers={"Content-Disposition": "attachment; filename=конспект.tex"})
    except Exception as e:
        logger.error(f"Ошибка экспорта LaTeX: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка экспорта LaTeX: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.1", "models": {"ocr": OPENAI_MODEL_OCR, "classification": OPENAI_MODEL_CLS}}

def start():
    import uvicorn
    uvicorn.run("split_project.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()
