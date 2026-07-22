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
    "Hi! I can help you explore the Korean American Story Legacy Project — hundreds of oral-history interviews plus census data from 1910-2020. Ask about a topic, an era, a person, or demographic trends.";

  let isSending = false;

  /* ---------- icons ---------- */
  const chatIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`;
  const closeIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
  const sendIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`;
  const chevronIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>`;
  const backIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"></polyline></svg>`;

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
    <div class="kaschat-quick-explore" id="quickExplore">
      <button class="kaschat-quick-toggle" type="button" id="quickToggle">
        <span class="kaschat-quick-label">Quick Explore</span>
        <span class="kaschat-quick-icon">${chevronIcon}</span>
      </button>
      <div class="kaschat-quick-content">
        <div class="kaschat-quick-group">
          <div class="kaschat-quick-group-label">Census Data</div>
          <div class="kaschat-quick-buttons">
            <button class="kaschat-quick-btn census" type="button" id="byYearBtn">By Year</button>
            <button class="kaschat-quick-btn census" type="button" id="byStateBtn">By State</button>
            <button class="kaschat-quick-btn census" type="button" id="trendsBtn">Population Trends</button>
          </div>
        </div>
      </div>
    </div>
    <div class="kaschat-form-wrapper">
      <div class="kaschat-selector" id="yearSelector">
        <div class="kaschat-selector-header">
          <span class="kaschat-selector-title">Select a census year</span>
          <button class="kaschat-back-btn" type="button" data-back>${backIcon} Back</button>
        </div>
        <div class="kaschat-selector-grid">
          <button class="kaschat-selector-btn" type="button" data-year="2020">2020</button>
          <button class="kaschat-selector-btn" type="button" data-year="2010">2010</button>
          <button class="kaschat-selector-btn" type="button" data-year="2000">2000</button>
          <button class="kaschat-selector-btn" type="button" data-year="1990">1990</button>
          <button class="kaschat-selector-btn" type="button" data-year="1980">1980</button>
          <button class="kaschat-selector-btn" type="button" data-year="1970">1970</button>
          <button class="kaschat-selector-btn" type="button" data-year="1960">1960</button>
          <button class="kaschat-selector-btn" type="button" data-year="1950">1950</button>
          <button class="kaschat-selector-btn" type="button" data-year="1940">1940</button>
          <button class="kaschat-selector-btn" type="button" data-year="1930">1930</button>
          <button class="kaschat-selector-btn" type="button" data-year="1920">1920</button>
          <button class="kaschat-selector-btn" type="button" data-year="1910">1910</button>
        </div>
      </div>
      <div class="kaschat-selector" id="stateSelector">
        <div class="kaschat-selector-header">
          <span class="kaschat-selector-title">Select a state</span>
          <button class="kaschat-back-btn" type="button" data-back>${backIcon} Back</button>
        </div>
        <div class="kaschat-selector-grid">
          <button class="kaschat-selector-btn" type="button" data-state="California">California</button>
          <button class="kaschat-selector-btn" type="button" data-state="New York">New York</button>
          <button class="kaschat-selector-btn" type="button" data-state="Texas">Texas</button>
          <button class="kaschat-selector-btn" type="button" data-state="New Jersey">New Jersey</button>
          <button class="kaschat-selector-btn" type="button" data-state="Virginia">Virginia</button>
          <button class="kaschat-selector-btn" type="button" data-state="Washington">Washington</button>
          <button class="kaschat-selector-btn" type="button" data-state="Illinois">Illinois</button>
          <button class="kaschat-selector-btn" type="button" data-state="Georgia">Georgia</button>
          <button class="kaschat-selector-btn" type="button" data-state="Maryland">Maryland</button>
          <button class="kaschat-selector-btn" type="button" data-state="Hawaii">Hawaii</button>
          <button class="kaschat-selector-btn" type="button" data-state="Pennsylvania">Pennsylvania</button>
          <button class="kaschat-selector-btn" type="button" data-state="Florida">Florida</button>
        </div>
      </div>
      <div class="kaschat-selector" id="trendsSelector">
        <div class="kaschat-selector-header">
          <span class="kaschat-selector-title">Select a trend to explore</span>
          <button class="kaschat-back-btn" type="button" data-back>${backIcon} Back</button>
        </div>
        <div class="kaschat-selector-grid two-col">
          <button class="kaschat-selector-btn trend" type="button" data-trend="How has the Korean American population changed over time?">Overall Growth<small>1910–2020 nationwide</small></button>
          <button class="kaschat-selector-btn trend" type="button" data-trend="Tell me about Korean American immigration after the Korean War">Post-War Immigration<small>1950s–1970s surge</small></button>
          <button class="kaschat-selector-btn trend" type="button" data-trend="Which states have the most Korean Americans?">Top States<small>Where Korean Americans live</small></button>
          <button class="kaschat-selector-btn trend" type="button" data-trend="How did the Korean American population change between 2010 and 2020?">Recent Growth<small>2010–2020 changes</small></button>
        </div>
      </div>
      <form class="kaschat-form" autocomplete="off">
        <input class="kaschat-input" type="text" placeholder="Ask about an interview…" aria-label="Message" maxlength="2000" />
        <button class="kaschat-send" type="submit" aria-label="Send">${sendIcon}</button>
      </form>
    </div>
  `;

  document.body.appendChild(launcher);
  document.body.appendChild(panel);

  const messagesEl = panel.querySelector('.kaschat-messages');
  const form = panel.querySelector('.kaschat-form');
  const input = panel.querySelector('.kaschat-input');
  const sendBtn = panel.querySelector('.kaschat-send');
  const closeBtn = panel.querySelector('.kaschat-close');
  const quickExplore = panel.querySelector('#quickExplore');
  const quickToggle = panel.querySelector('#quickToggle');
  const yearSelector = panel.querySelector('#yearSelector');
  const stateSelector = panel.querySelector('#stateSelector');
  const trendsSelector = panel.querySelector('#trendsSelector');

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

  function hideAllSelectors() {
    yearSelector.classList.remove('show');
    stateSelector.classList.remove('show');
    trendsSelector.classList.remove('show');
  }

  function openPanel() {
    panel.classList.add('open');
    launcher.setAttribute('aria-expanded', 'true');
    if (!messagesEl.children.length) {
      addAssistantMessage(WELCOME, []);
    }
    setTimeout(() => input.focus(), 100);
  }

  function closePanel() {
    panel.classList.remove('open');
    launcher.setAttribute('aria-expanded', 'false');
    hideAllSelectors();
  }

  /* ---------- API ---------- */
  async function sendMessage(text) {
    if (isSending) return;
    isSending = true;
    sendBtn.disabled = true;
    hideAllSelectors();

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

  // Quick explore toggle
  quickToggle.addEventListener('click', () => {
    quickExplore.classList.toggle('expanded');
    hideAllSelectors();
  });

  // Interview topic buttons
  panel.querySelectorAll('.kaschat-quick-btn[data-query]').forEach(btn => {
    btn.addEventListener('click', () => {
      sendMessage(btn.dataset.query);
    });
  });

  // Census selector buttons
  panel.querySelector('#byYearBtn').addEventListener('click', () => {
    hideAllSelectors();
    yearSelector.classList.add('show');
  });
  panel.querySelector('#byStateBtn').addEventListener('click', () => {
    hideAllSelectors();
    stateSelector.classList.add('show');
  });
  panel.querySelector('#trendsBtn').addEventListener('click', () => {
    hideAllSelectors();
    trendsSelector.classList.add('show');
  });

  // Back buttons
  panel.querySelectorAll('[data-back]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      hideAllSelectors();
    });
  });

  // Year selection
  panel.querySelectorAll('[data-year]').forEach(btn => {
    btn.addEventListener('click', () => {
      const year = btn.dataset.year;
      sendMessage(`Korean American population in ${year}`);
    });
  });

  // State selection
  panel.querySelectorAll('[data-state]').forEach(btn => {
    btn.addEventListener('click', () => {
      const state = btn.dataset.state;
      sendMessage(`Korean American population in ${state}`);
    });
  });

  // Trend selection
  panel.querySelectorAll('[data-trend]').forEach(btn => {
    btn.addEventListener('click', () => {
      sendMessage(btn.dataset.trend);
    });
  });

  // Close selectors when clicking elsewhere
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.kaschat-quick-btn.census') && !e.target.closest('.kaschat-selector')) {
      hideAllSelectors();
    }
  });
})();
