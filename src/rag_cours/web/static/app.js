"use strict";

// --- Icônes (Lucide, comme dans Obsidian) ---------------------------------

const ICONS = {
  pencil: '<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/>',
  clipboard: '<rect width="8" height="4" x="8" y="2" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
  circleCheck: '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
  flame: '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  help: '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
  alert: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  zap: '<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
  bug: '<path d="m8 2 1.88 1.88"/><path d="M14.12 3.88 16 2"/><path d="M9 7.13v-1a3.003 3.003 0 1 1 6 0v1"/><path d="M12 20c-3.3 0-6-2.7-6-6v-3a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v3c0 3.3-2.7 6-6 6"/><path d="M12 20v-9"/><path d="M6 13H2"/><path d="M22 13h-4"/>',
  list: '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
  quote: '<path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V20c0 1 0 1 1 1z"/><path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 4v3c0 1 0 1 1 1z"/>',
  code: '<path d="m16 18 6-6-6-6"/><path d="m8 6-6 6 6 6"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
};

const CALLOUT_ICONS = {
  note: "pencil",
  abstract: "clipboard", summary: "clipboard", tldr: "clipboard",
  info: "info",
  todo: "circleCheck",
  tip: "flame", hint: "flame", important: "flame",
  success: "check", check: "check", done: "check",
  question: "help", help: "help", faq: "help",
  warning: "alert", caution: "alert", attention: "alert",
  failure: "x", fail: "x", missing: "x",
  danger: "zap", error: "zap",
  bug: "bug",
  example: "list",
  quote: "quote", cite: "quote",
  code: "code",
};

function svgIcon(name) {
  return `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ICONS.pencil}</svg>`;
}

// --- Rendu markdown façon Obsidian ---------------------------------------

let vaultName = "Programming_travel";

function obsidianUrl(file) {
  return `obsidian://open?vault=${encodeURIComponent(vaultName)}&file=${encodeURIComponent(file)}`;
}

// Transforme `> [!type]+/- Titre` (blockquote rendu par marked) en callout Obsidian
function transformCallouts(root) {
  // Ordre inverse : les callouts imbriqués sont traités avant leur parent
  const quotes = Array.from(root.querySelectorAll("blockquote")).reverse();
  for (const quote of quotes) {
    const first = quote.firstElementChild;
    if (!first || first.tagName !== "P") continue;
    const head = first.firstChild;
    if (!head || head.nodeType !== Node.TEXT_NODE) continue;
    const match = head.textContent.match(/^\s*\[!([\w-]+)\]([+-]?)[ \t]*/);
    if (!match) continue;

    const type = match[1].toLowerCase();
    const fold = match[2];
    head.textContent = head.textContent.slice(match[0].length);

    // Le titre s'arrête au premier saut de ligne du paragraphe
    const title = document.createDocumentFragment();
    while (first.firstChild) {
      const node = first.firstChild;
      if (node.nodeName === "BR") { node.remove(); break; }
      if (node.nodeType === Node.TEXT_NODE) {
        const nl = node.textContent.indexOf("\n");
        if (nl !== -1) {
          title.append(node.textContent.slice(0, nl));
          node.textContent = node.textContent.slice(nl + 1);
          break;
        }
      }
      title.append(node);
    }
    if (!first.textContent.trim() && !first.querySelector("img, code")) first.remove();

    const callout = document.createElement(fold ? "details" : "div");
    callout.className = "callout";
    callout.dataset.callout = type;
    if (fold === "+") callout.open = true;

    const titleEl = document.createElement(fold ? "summary" : "div");
    titleEl.className = "callout-title";
    const icon = document.createElement("span");
    icon.className = "callout-icon";
    icon.innerHTML = svgIcon(CALLOUT_ICONS[type]);
    const inner = document.createElement("span");
    inner.className = "callout-title-inner";
    if (title.textContent.trim()) inner.append(title);
    else inner.textContent = type.charAt(0).toUpperCase() + type.slice(1);
    titleEl.append(icon, inner);
    if (fold) {
      const chevron = document.createElement("span");
      chevron.className = "callout-fold";
      chevron.innerHTML = svgIcon("chevron");
      titleEl.append(chevron);
    }

    const content = document.createElement("div");
    content.className = "callout-content";
    content.append(...quote.childNodes);

    callout.append(titleEl);
    if (content.textContent.trim() || content.querySelector("img, pre, table")) callout.append(content);
    quote.replaceWith(callout);
  }
}

// Wikilinks, références [Source N], repères [image] et #tags (hors code et liens)
const TEXT_PATTERN = new RegExp(
  [
    String.raw`\[\[([^\]|\n]+)(?:\|([^\]\n]+))?\]\]`,
    String.raw`\[Sources?\s+(\d+(?:\s*(?:,|et|&)\s*(?:Source\s+)?\d+)*)\]`,
    String.raw`(\[image\])`,
    String.raw`(^|[\s(])#([\p{L}\p{N}_/-]*[\p{L}_/-][\p{L}\p{N}_/-]*)`,
  ].join("|"),
  "gu",
);

function decorateText(root, { sourceRefs }) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentElement.closest("pre, code, a, .callout-title")
        ? NodeFilter.FILTER_REJECT
        : NodeFilter.FILTER_ACCEPT,
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);

  for (const node of nodes) {
    const text = node.textContent;
    TEXT_PATTERN.lastIndex = 0;
    if (!TEXT_PATTERN.test(text)) continue;
    TEXT_PATTERN.lastIndex = 0;

    const frag = document.createDocumentFragment();
    let last = 0;
    for (const m of text.matchAll(TEXT_PATTERN)) {
      frag.append(text.slice(last, m.index));
      last = m.index + m[0].length;

      if (m[1] !== undefined) {
        const target = m[1].trim();
        const link = document.createElement("a");
        link.className = "internal-link";
        link.href = obsidianUrl(target.split("#")[0]);
        link.textContent = (m[2] || target).trim();
        frag.append(link);
      } else if (m[3] !== undefined) {
        const nums = m[3].match(/\d+/g);
        if (!sourceRefs) { frag.append(m[0]); continue; }
        nums.forEach((n, i) => {
          if (i > 0) frag.append(" ");
          const ref = document.createElement("a");
          ref.className = "source-ref";
          ref.href = `#source-${n}`;
          ref.textContent = n;
          ref.title = `Source ${n}`;
          ref.addEventListener("click", (e) => { e.preventDefault(); focusSource(n); });
          frag.append(ref);
        });
      } else if (m[4] !== undefined) {
        const img = document.createElement("span");
        img.className = "image-placeholder";
        img.textContent = "image";
        frag.append(img);
      } else {
        frag.append(m[5]);
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = `#${m[6]}`;
        frag.append(tag);
      }
    }
    frag.append(text.slice(last));
    node.replaceWith(frag);
  }
}

function renderMarkdown(markdown, container, { sourceRefs = false } = {}) {
  if (!window.marked || !window.DOMPurify) {
    // CDN injoignable : on affiche au moins le texte brut
    container.textContent = markdown;
    container.style.whiteSpace = "pre-wrap";
    return;
  }
  container.innerHTML = DOMPurify.sanitize(marked.parse(markdown, { gfm: true, breaks: true }));
  transformCallouts(container);
  decorateText(container, { sourceRefs });
  if (window.hljs) {
    container.querySelectorAll("pre code").forEach((el) => {
      el.className = el.className.toLowerCase();
      hljs.highlightElement(el);
    });
  }
}

// --- Interface ------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const form = $("ask-form");
const questionEl = $("question");
const matiereEl = $("matiere");
const backendEl = $("backend");
const topKEl = $("top-k");
const submitEl = $("submit");
const statusEl = $("status");
const answerEl = $("answer");
const sourcesEl = $("sources");

function storageGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}

function storageSet(key, value) {
  try { localStorage.setItem(key, value); } catch { /* stockage indisponible */ }
}

function addOptions(select, values) {
  for (const value of values) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    select.append(opt);
  }
}

async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    const config = await res.json();
    vaultName = config.vault || vaultName;
    addOptions(matiereEl, config.matieres || []);
    addOptions(backendEl, config.backends || []);
    const saved = storageGet("rag.backend");
    backendEl.value = (config.backends || []).includes(saved) ? saved : config.backend;
    const savedMatiere = storageGet("rag.matiere");
    if ((config.matieres || []).includes(savedMatiere)) matiereEl.value = savedMatiere;
  } catch (err) {
    showError(`Configuration indisponible : ${err.message}`);
  }
}

function showStatus(text) {
  statusEl.hidden = false;
  statusEl.className = "status";
  statusEl.replaceChildren();
  const spinner = document.createElement("div");
  spinner.className = "spinner";
  const label = document.createElement("span");
  label.textContent = text;
  statusEl.append(spinner, label);
  return label;
}

function showError(message) {
  statusEl.hidden = false;
  statusEl.className = "status error";
  statusEl.textContent = message;
}

function focusSource(n) {
  const card = document.getElementById(`source-${n}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.remove("highlight");
  void card.offsetWidth;
  card.classList.add("highlight");
  setTimeout(() => card.classList.remove("highlight"), 1600);
}

function noteName(path) {
  return path.split("/").pop().replace(/\.md$/, "");
}

function renderSources(sources) {
  const list = $("sources-list");
  list.replaceChildren();
  for (const s of sources) {
    const li = document.createElement("li");
    li.className = "source";
    li.id = `source-${s.index}`;

    const head = document.createElement("div");
    head.className = "source-head";
    const num = document.createElement("span");
    num.className = "source-num";
    num.textContent = `[${s.index}]`;

    const title = document.createElement("div");
    title.className = "source-title";
    const link = document.createElement("a");
    link.className = "source-link";
    link.href = s.obsidian_url;
    link.textContent = noteName(s.fichier);
    link.title = "Ouvrir dans Obsidian";
    title.append(link);
    if (s.section) {
      const section = document.createElement("span");
      section.className = "source-section";
      // Les titres peuvent contenir du markdown inline (`code`, **gras**)
      section.append(" › ");
      const inline = document.createElement("span");
      if (window.marked && window.DOMPurify) inline.innerHTML = DOMPurify.sanitize(marked.parseInline(s.section));
      else inline.textContent = s.section;
      section.append(inline);
      title.append(section);
    }
    const meta = document.createElement("div");
    meta.className = "source-meta";
    const parts = [s.matiere, s.fichier];
    if (typeof s.distance === "number") parts.push(`pertinence ${Math.round((1 - s.distance) * 100)} %`);
    meta.textContent = parts.filter(Boolean).join(" · ");
    title.append(meta);
    head.append(num, title);

    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Voir l'extrait";
    const excerpt = document.createElement("div");
    excerpt.className = "markdown-rendered";
    details.append(summary, excerpt);
    details.addEventListener("toggle", () => {
      if (details.open && !excerpt.hasChildNodes()) renderMarkdown(s.texte, excerpt);
    }, { once: false });

    li.append(head, details);
    list.append(li);
  }
  sourcesEl.hidden = sources.length === 0;
}

async function ask(event) {
  event.preventDefault();
  const question = questionEl.value.trim();
  if (!question) return;

  storageSet("rag.backend", backendEl.value);
  storageSet("rag.matiere", matiereEl.value);
  submitEl.disabled = true;
  answerEl.hidden = true;
  sourcesEl.hidden = true;

  const started = performance.now();
  const elapsed = () => Math.round((performance.now() - started) / 1000);
  const label = showStatus("Recherche dans les notes et génération de la réponse…");
  const timer = setInterval(() => {
    label.textContent = `Recherche dans les notes et génération de la réponse… ${elapsed()} s`;
  }, 1000);

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        matiere: matiereEl.value || null,
        backend: backendEl.value || null,
        top_k: Number(topKEl.value) || 5,
      }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join(" ; ") : data.detail;
      throw new Error(detail || `Erreur HTTP ${res.status}`);
    }

    statusEl.hidden = true;
    const model = data.modele ? ` (${data.modele})` : "";
    $("answer-meta").textContent =
      `Généré par ${data.backend}${model} · ${data.sources.length} source(s) · ${elapsed()} s`;
    renderMarkdown(data.reponse, $("answer-body"), { sourceRefs: true });
    answerEl.hidden = false;
    renderSources(data.sources);
  } catch (err) {
    showError(err.message);
  } finally {
    clearInterval(timer);
    submitEl.disabled = false;
  }
}

questionEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    form.requestSubmit();
  }
});
form.addEventListener("submit", ask);
if (window.hljs) hljs.configure({ ignoreUnescapedHTML: true });
loadConfig();
