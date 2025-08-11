let currentStep = 1;
let analysisData = {};

document.addEventListener('DOMContentLoaded', function () {
  const fileInput = document.getElementById('fileInput');
  if (fileInput) {
    fileInput.addEventListener('change', handleFileUpload);
  }

  const uploadArea = document.querySelector('.upload-area');
  if (uploadArea) {
    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
      uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.classList.remove('dragover');
      const files = e.dataTransfer.files;
      if (files && files.length > 0) processFile(files[0]);
    });
  }

  const cards = document.querySelectorAll('.analysis-card');
  cards.forEach(card => {
    card.addEventListener('mouseenter', () => {
      card.style.transform = 'translateY(-5px)';
      card.style.boxShadow = '0 8px 25px rgba(0,0,0,0.1)';
    });
    card.addEventListener('mouseleave', () => {
      card.style.transform = 'translateY(0)';
      card.style.boxShadow = 'none';
    });
  });
});

function handleFileUpload(event) {
  const file = event.target.files && event.target.files[0];
  if (file) processFile(file);
}

async function processFile(file) {
  try {
    hideError();
    showLoading();
    updateProgress(2);
    updateLoadingText("Извлекаем текст из документа...");

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://localhost:8000/upload', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    updateProgress(3);
    updateLoadingText("Классифицируем содержимое...");

    const result = await response.json();

    updateProgress(4);
    updateLoadingText("Создаем структурированный анализ...");

    analysisData = result;

    updateAnalysisResults(result.summary || { categories: {} });
    await updateHighlightedText(result.highlighted_html || "");

    updateProgress(5);
    updateLoadingText("Готово! Теперь вы можете общаться с AI...");

    setTimeout(() => {
      hideLoading();
      showResults();
    }, 300);

  } catch (error) {
    console.error('Ошибка обработки файла:', error);
    hideLoading();
    showError('Ошибка обработки файла: ' + (error?.message || error));
    resetProgress();
  }
}

function updateProgress(step) {
  for (let i = 1; i < step; i++) {
    const el = document.getElementById('step' + i);
    if (!el) continue;
    el.classList.add('completed');
    el.classList.remove('active');
  }
  const cur = document.getElementById('step' + step);
  if (cur) cur.classList.add('active');
  currentStep = step;
}

function resetProgress() {
  for (let i = 1; i <= 5; i++) {
    const step = document.getElementById('step' + i);
    if (step) step.classList.remove('completed', 'active');
  }
  const s1 = document.getElementById('step1');
  if (s1) s1.classList.add('active');
  currentStep = 1;
}

function showLoading() {
  const el = document.getElementById('loadingSection');
  if (el) el.style.display = 'block';
}

function hideLoading() {
  const el = document.getElementById('loadingSection');
  if (el) el.style.display = 'none';
}

function updateLoadingText(text) {
  const el = document.getElementById('loadingText');
  if (el) el.textContent = text;
}

function showError(message) {
  const txt = document.getElementById('errorText');
  if (txt) txt.textContent = message;
  const sec = document.getElementById('errorSection');
  if (sec) sec.style.display = 'block';
}

function hideError() {
  const sec = document.getElementById('errorSection');
  if (sec) sec.style.display = 'none';
}

function updateAnalysisResults(summary) {
  if (!summary || !summary.categories) return;
  const c = summary.categories;

  const set = (id, txt) => {
    const el = document.getElementById(id);
    if (el) el.textContent = txt;
  };

  set('definitionsCount', 'Найдено ' + (c.definitions ?? 0) + ' определений');
  set('examplesCount', 'Найдено ' + (c.examples ?? 0) + ' примеров');
  set('theoremsCount', 'Найдено ' + (c.theorems ?? 0) + ' теорем');
  set('datesCount', 'Найдено ' + (c.dates ?? 0) + ' дат');
  set('formulasCount', 'Найдено ' + (c.formulas ?? 0) + ' формул');

  const topicTags = document.getElementById('topicTags');
  if (topicTags) {
    topicTags.innerHTML = '';
    if (Array.isArray(summary.topics)) {
      summary.topics.forEach(topic => {
        const tag = document.createElement('span');
        tag.className = 'topic-tag';
        tag.textContent = topic;
        topicTags.appendChild(tag);
      });
    }
  }
}

async function updateHighlightedText(highlightedHTML) {
  const container = document.getElementById('highlightedText');
  if (!container) return;
  container.innerHTML = highlightedHTML;

  if (window.MathJax && window.MathJax.typesetPromise) {
    try {
      await window.MathJax.typesetPromise([container]);
    } catch (error) {
      console.error('Ошибка рендеринга MathJax:', error);
    }
  }
}

function showResults() {
  const show = (id) => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'block';
  };
  show('analysisSection');
  show('downloadSection');
  show('chatSection');

  const chatMessages = document.getElementById('chatMessages');
  if (chatMessages) {
    const total = analysisData.summary ? analysisData.summary.total_segments : 0;
    chatMessages.innerHTML =
      '<div class="message ai">Отлично! Я проанализировал ваш конспект. Найдено ' + total +
      ' сегментов текста. Теперь могу ответить на любые вопросы по содержимому! 🤖</div>';
  }

  setTimeout(() => {
    const sec = document.getElementById('analysisSection');
    if (sec) sec.scrollIntoView({ behavior: 'smooth' });
  }, 300);
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function downloadFile(format) {
  if (format === 'tex') {
    if (analysisData.latex_source) {
      const blob = new Blob([analysisData.latex_source], { type: 'text/plain; charset=utf-8' });
      downloadBlob(blob, 'конспект.tex');
      return;
    }
    try {
      const response = await fetch('http://localhost:8000/export/tex', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(analysisData.segments || [])
      });
      if (response.ok) {
        const latexContent = await response.text();
        const blob = new Blob([latexContent], { type: 'text/plain; charset=utf-8' });
        downloadBlob(blob, 'конспект.tex');
        return;
      }
      throw new Error(`HTTP ${response.status}`);
    } catch (e) {
      console.error('Ошибка получения LaTeX:', e);
      alert('Не удалось получить LaTeX файл');
      return;
    }
  }

  if (format === 'pdf' && analysisData.segments) {
    try {
      const response = await fetch('http://localhost:8000/export/pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(analysisData.segments)
      });
      if (response.ok) {
        const blob = await response.blob();
        downloadBlob(blob, 'классифицированный_конспект.pdf');
        return;
      }
      throw new Error(`HTTP ${response.status}`);
    } catch (e) {
      console.error('Ошибка экспорта PDF:', e);
    }

    if (analysisData.pdf_base64) {
      const byteCharacters = atob(analysisData.pdf_base64);
      const byteNumbers = new Uint8Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const blob = new Blob([byteNumbers], { type: 'application/pdf' });
      downloadBlob(blob, 'обработанный_конспект.pdf');
      return;
    }
  }

  if (format === 'html') {
    const totalSegments = analysisData.summary?.total_segments ?? 0;
    const definitions   = analysisData.summary?.categories?.definitions ?? 0;
    const examples      = analysisData.summary?.categories?.examples ?? 0;
    const theorems      = analysisData.summary?.categories?.theorems ?? 0;
    const formulas      = analysisData.summary?.categories?.formulas ?? 0;
    const dates         = analysisData.summary?.categories?.dates ?? 0;
    const contentHTML   = document.getElementById('highlightedText')?.innerHTML ?? '';

    const htmlTemplate = `<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>Классифицированный конспект</title>

  <!-- Конфиг MathJax должен быть ДО подключения скрипта -->
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [["$", "$"], ["\\\\(", "\\\\)"]],
        displayMath: [["$$", "$$"], ["\\\\[", "\\\\]"]],
        processEscapes: true,
        processEnvironments: true
      },
      options: {
        skipHtmlTags: ['script','noscript','style','textarea','pre']
      }
    };
  </script>
  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 24px; line-height: 1.6; }
    h1 { margin: 0 0 16px 0; }
    .stats { background: #f8fafc; padding: 16px 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #e2e8f0; }
    .stats p { margin: 4px 0; }
    .content { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; }
    .highlight-definition { background: #dcfce7; padding: 2px 4px; border-radius: 3px; }
    .highlight-example { background: #fef3c7; padding: 2px 4px; border-radius: 3px; }
    .highlight-theorem { background: #fee2e2; padding: 2px 4px; border-radius: 3px; }
    .highlight-date { background: #e9d5ff; padding: 2px 4px; border-radius: 3px; }
    .highlight-formula { background: #cffafe; padding: 2px 4px; border-radius: 3px; }
    mjx-container[jax="CHTML"] { line-height: 1.2; }
  </style>
</head>
<body>
  <h1>📚 Классифицированный конспект</h1>

  <div class="stats">
    <h2>📊 Статистика</h2>
    <p><strong>Всего сегментов:</strong> ${totalSegments}</p>
    <p><strong>Определения:</strong> ${definitions}</p>
    <p><strong>Примеры:</strong> ${examples}</p>
    <p><strong>Теоремы:</strong> ${theorems}</p>
    <p><strong>Формулы:</strong> ${formulas}</p>
    <p><strong>Даты:</strong> ${dates}</p>
  </div>

  <div class="content">
    ${contentHTML}
  </div>
</body>
</html>`;

    const blob = new Blob([htmlTemplate], { type: 'text/html; charset=utf-8' });
    downloadBlob(blob, 'классифицированный_конспект.html');
  }
}

function playAudio() {
  alert('Функция аудио воспроизведения будет добавлена в будущих версиях!');
}

function handleChatKeyPress(event) {
  if (event.key === 'Enter') sendMessage();
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const message = (input?.value || '').trim();
  if (!message) return;

  addMessage(message, 'user');
  input.value = '';

  const loadingMessage = addMessage('Думаю...', 'ai');

  try {
    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        segments: analysisData.segments || []
      })
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const result = await response.json();
    loadingMessage.textContent = result.response || 'Готово.';

  } catch (error) {
    console.error('Ошибка чата:', error);
    loadingMessage.textContent = 'Извините, произошла ошибка при обработке вашего вопроса.';
  }
}

function addMessage(text, sender) {
  const chatMessages = document.getElementById('chatMessages');
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${sender}`;
  messageDiv.textContent = text;
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return messageDiv;
}
