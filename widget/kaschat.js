/**
 * KAS Archive Assistant widget.
 *
 * Embed on any page with:
 *   <link rel="stylesheet" href="http://localhost:8000/widget/kaschat.css">
 *   <script src="http://localhost:8000/widget/kaschat.js" data-api="http://localhost:8000"></script>
 *
 * Optional data-api attribute overrides the API base URL. Defaults to same-origin.
 */
(function () {
  'use strict';

  const scriptTag = document.currentScript;
  const API_BASE =
    (scriptTag && scriptTag.getAttribute('data-api')) ||
    window.KASCHAT_API_BASE ||
    '';

  const WELCOME =
    "Hi! I can help you explore the Korean American Story Legacy Project — hundreds of oral-history interviews plus census data from 1910-2020. Ask about a topic, an era, a person, or demographic trends, and I'll help you find what you're looking for.";

  const SUGGESTIONS = [
    'Stories from the Korean War',
    'Immigrating to America',
    'Growing up Korean American',
    'Korean American population trends',
    'Korean adoptee experiences',
  ];

  let isSending = false;

  /* ---------- icons ---------- */
  const chatIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`;
  const closeIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
  const sendIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`;

  /* ---------- DOM ---------- */
  const launcher = document.createElement('button');
  launcher.className = 'kaschat-launcher';
  launcher.type = 'button';
  launcher.setAttribute('aria-label', 'Open archive assistant');
  launcher.setAttribute('aria-expanded', 'false');
  launcher.innerHTML = chatIcon;

  const panel = document.createElement('div');
  panel.className = 'kaschat-panel';
  panel.setAttribute('role', 'dialog');
  panel.setAttribute('aria-label', 'KAS Archive Assistant');
  panel.innerHTML = `
    <div class="kaschat-header">
      <div class="kaschat-header-text">
        <div class="kaschat-header-title">Archive Assistant</div>
        <div class="kaschat-header-sub">Korean American Story — Legacy Project</div>
      </div>
      <button class="kaschat-close" type="button" aria-label="Close">${closeIcon}</button>
    </div>
    <div class="kaschat-messages" role="log" aria-live="polite"></div>
    <form class="kaschat-form" autocomplete="off">
      <input class="kaschat-input" type="text" placeholder="Ask about an interview…" aria-label="Message" maxlength="2000" />
      <button class="kaschat-send" type="submit" aria-label="Send">${sendIcon}</button>
    </form>
  `;

  document.body.appendChild(launcher);
  document.body.appendChild(panel);

  const messagesEl = panel.querySelector('.kaschat-messages');
  const form = panel.querySelector('.kaschat-form');
  const input = panel.querySelector('.kaschat-input');
  const sendBtn = panel.querySelector('.kaschat-send');
  const closeBtn = panel.querySelector('.kaschat-close');

  /* ---------- helpers ---------- */
  function scrollToEnd() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
  }

  function renderAnswerWithCites(text) {
    return escapeHtml(text).replace(
      /\[(\d+)\]/g,
      '<cite-mark>[$1]</cite-mark>'
    );
  }

  function addUserMessage(text) {
    const el = document.createElement('div');
    el.className = 'kaschat-msg user';
    el.textContent = text;
    messagesEl.appendChild(el);
    scrollToEnd();
  }

  function formatTimestamp(seconds) {
    if (!seconds || seconds < 0) return '';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  function renderCitation(c) {
    const hasUrl = Boolean(c.youtube_url);
    const node = document.createElement(hasUrl ? 'a' : 'div');
    node.className = 'kaschat-citation' + (hasUrl ? '' : ' no-url');
    if (hasUrl) {
      node.href = c.youtube_url;
      node.target = '_blank';
      node.rel = 'noopener noreferrer';
    }

    const ts = c.start_seconds && c.start_seconds > 0
      ? `<span class="kaschat-citation-timestamp">· ${formatTimestamp(c.start_seconds)}</span>`
      : '';
    const metaParts = [c.interviewee, c.date].filter(Boolean).map(escapeHtml);
    const metaLine = metaParts.length
      ? `<div class="kaschat-citation-meta">${metaParts.join(' · ')}${ts}</div>`
      : (ts ? `<div class="kaschat-citation-meta">${ts}</div>` : '');

    const thumb = c.thumbnail_url
      ? `<img class="kaschat-citation-thumb" src="${escapeHtml(c.thumbnail_url)}" alt="" loading="lazy" onerror="this.outerHTML='<div class=\\'kaschat-citation-thumb-fallback\\'>KAS</div>'" />`
      : `<div class="kaschat-citation-thumb-fallback">KAS</div>`;

    node.innerHTML = `
      ${thumb}
      <div class="kaschat-citation-body">
        <div class="kaschat-citation-title">
          <span class="kaschat-citation-index">[${c.index}]</span>${escapeHtml(c.title)}
        </div>
        ${metaLine}
      </div>
    `;
    return node;
  }

  function addAssistantMessage(text, citations) {
    const msg = document.createElement('div');
    msg.className = 'kaschat-msg assistant';
    msg.innerHTML = renderAnswerWithCites(text);
    messagesEl.appendChild(msg);

    if (citations && citations.length) {
      const wrap = document.createElement('div');
      wrap.className = 'kaschat-citations';
      const MAX_VISIBLE = 3;
      citations.forEach((c, i) => {
        const node = renderCitation(c);
        if (i >= MAX_VISIBLE) node.classList.add('hidden');
        wrap.appendChild(node);
      });
      if (citations.length > MAX_VISIBLE) {
        const hiddenCount = citations.length - MAX_VISIBLE;
        const moreLabel = `See ${hiddenCount} more video${hiddenCount > 1 ? 's' : ''}`;
        const lessLabel = 'See less';
        const btn = document.createElement('button');
        btn.className = 'kaschat-more';
        btn.type = 'button';
        btn.textContent = moreLabel;
        let expanded = false;
        btn.addEventListener('click', () => {
          expanded = !expanded;
          const extras = wrap.querySelectorAll('.kaschat-citation');
          extras.forEach((el, i) => {
            if (i >= MAX_VISIBLE) el.classList.toggle('hidden', !expanded);
          });
          btn.textContent = expanded ? lessLabel : moreLabel;
          if (expanded) scrollToEnd();
        });
        wrap.appendChild(btn);
      }
      messagesEl.appendChild(wrap);
    }
    scrollToEnd();
  }

  function addTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'kaschat-typing';
    el.innerHTML = '<span></span><span></span><span></span>';
    messagesEl.appendChild(el);
    scrollToEnd();
    return el;
  }

  function addError(text) {
    const el = document.createElement('div');
    el.className = 'kaschat-error';
    el.textContent = text;
    messagesEl.appendChild(el);
    scrollToEnd();
  }

  function addSuggestions() {
    const wrap = document.createElement('div');
    wrap.className = 'kaschat-suggestions';
    SUGGESTIONS.forEach((q) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'kaschat-suggestion';
      btn.textContent = q;
      btn.addEventListener('click', () => {
        sendMessage(q);
      });
      wrap.appendChild(btn);
    });
    messagesEl.appendChild(wrap);
    scrollToEnd();
  }

  function openPanel() {
    panel.classList.add('open');
    launcher.setAttribute('aria-expanded', 'true');
    if (!messagesEl.children.length) {
      addAssistantMessage(WELCOME, []);
      addSuggestions();
    }
    setTimeout(() => input.focus(), 100);
  }

  function closePanel() {
    panel.classList.remove('open');
    launcher.setAttribute('aria-expanded', 'false');
  }

  /* ---------- API ---------- */
  async function sendMessage(text) {
    if (isSending) return;
    isSending = true;
    sendBtn.disabled = true;

    const suggestionsEl = messagesEl.querySelector('.kaschat-suggestions');
    if (suggestionsEl) suggestionsEl.remove();

    addUserMessage(text);
    const typing = addTypingIndicator();

    try {
      const resp = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      typing.remove();

      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        addError(detail.detail || `Error ${resp.status}`);
        return;
      }

      const data = await resp.json();
      addAssistantMessage(data.answer, data.citations);
    } catch (e) {
      typing.remove();
      addError('Could not reach the server. Is it running?');
    } finally {
      isSending = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  /* ---------- events ---------- */
  launcher.addEventListener('click', openPanel);
  closeBtn.addEventListener('click', closePanel);
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    sendMessage(text);
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && panel.classList.contains('open')) closePanel();
  });
})();
