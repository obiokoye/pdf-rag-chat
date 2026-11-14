// PDF RAG Chat front end: plain JavaScript, no build step.
const $ = (id) => document.getElementById(id);
const sessionId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2);
let busy = false;

const escapeHtml = (s) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

function shortName(doc) {
  return doc.replace(/\.pdf$/i, "").replace(/[-_]+/g, " ");
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail?.[0]?.msg || body.detail || `Request failed (${res.status})`);
  return body;
}

async function loadStatus() {
  const s = await api("/api/status");
  $("engine").innerHTML = s.demo
    ? `Model: <code>scripted demo</code><br>Embeddings: <code>${s.embedder}</code> · Vector DB: <code>Chroma</code>`
    : `Model: <code>${escapeHtml(s.model)}</code> (${s.provider})<br>Embeddings: <code>${s.embedder}</code> · Vector DB: <code>Chroma</code>`;
  if (s.demo) {
    const n = $("notice");
    n.hidden = false;
    n.innerHTML = `<strong>Demo mode: scripted AI replies.</strong> Retrieval and citations are real; answers are scripted for the sample PDFs (${escapeHtml(s.demo_reason)}).`;
  }
}

async function loadDocs(highlight) {
  const docs = await api("/api/documents");
  $("docs").innerHTML = docs.length
    ? docs
        .map(
          (d) => `<li class="${d.name === highlight ? "new" : ""}"><span class="name">📄 ${escapeHtml(d.name)}</span>
          <span class="meta">${d.pages} pages · ${d.chunks} chunks</span></li>`,
        )
        .join("")
    : `<li class="muted">No PDFs yet. Add one below.</li>`;
  const pages = docs.reduce((n, d) => n + d.pages, 0);
  $("doc-count").textContent = `${docs.length} PDFs · ${pages} pages indexed`;
}

function addMessage(role, html, extraClass = "") {
  $("empty")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role} ${extraClass}`;
  el.innerHTML = html;
  $("messages").appendChild(el);
  const box = $("messages");
  box.scrollTo({ top: box.scrollHeight, behavior: "smooth" });
  return el;
}

function renderAnswer(data, question) {
  const byN = Object.fromEntries(data.citations.map((c) => [c.n, c]));
  const text = escapeHtml(data.answer).replace(/\[(\d+)\]/g, (m, n) =>
    byN[n] ? `<button class="cite" data-n="${n}" title="${escapeHtml(byN[n].doc)}, page ${byN[n].page}">${n}</button>` : "",
  );
  const chips = data.citations
    .map((c) => `<button class="chip" data-n="${c.n}"><b>${c.n}</b>${escapeHtml(shortName(c.doc))} · p.${c.page}</button>`)
    .join("");
  let status;
  if (data.grounded) {
    status = `<div class="status">✓ Grounded in ${data.citations.length} cited passage${data.citations.length > 1 ? "s" : ""}</div>`;
  } else if (!data.called_llm) {
    status = `<div class="status">No passage scored above the relevance threshold, so the model was not called.</div>`;
  } else {
    status = `<div class="status warn">⚠ No source cited. Check this answer yourself.</div>`;
  }
  const el = addMessage(
    "bot",
    `<div class="bubble">${text}</div>${chips ? `<div class="chips">${chips}</div>` : ""}${status}`,
    data.grounded ? "" : "ungrounded",
  );
  el.querySelectorAll("[data-n]").forEach((b) =>
    b.addEventListener("click", () => {
      document.querySelectorAll(".cite.active").forEach((x) => x.classList.remove("active"));
      el.querySelectorAll(`.cite[data-n="${b.dataset.n}"]`).forEach((x) => x.classList.add("active"));
      showSource(byN[b.dataset.n], question);
    }),
  );
}

function highlight(text, question) {
  const words = [...new Set(question.toLowerCase().match(/[a-z0-9.]{4,}/g) || [])];
  let html = escapeHtml(text);
  for (const w of words) {
    html = html.replace(new RegExp(`\\b(${w.replace(/[.]/g, "\\.")}\\w*)`, "gi"), "<mark>$1</mark>");
  }
  return html;
}

function showSource(c, question) {
  $("source-body").innerHTML = `
    <div class="src-doc">📄 ${escapeHtml(c.doc)}</div>
    <div class="src-meta"><span class="pill">Page ${c.page}</span><span class="pill">Citation [${c.n}]</span>
      <span class="pill score">Match ${c.score.toFixed(2)}</span></div>
    <div class="src-text">${highlight(c.snippet, question)}</div>
    <a class="open-pdf" href="/files/${encodeURIComponent(c.doc)}#page=${c.page}" target="_blank" rel="noopener">Open page ${c.page} in the PDF ↗</a>`;
  $("source").hidden = false;
  document.querySelector(".app").classList.add("with-source");
}

function closeSource() {
  $("source").hidden = true;
  document.querySelector(".app").classList.remove("with-source");
}

async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  $("send").disabled = true;
  $("sidebar").classList.remove("open");
  addMessage("user", `<div class="bubble">${escapeHtml(question)}</div>`);
  const typing = addMessage("bot", `<div class="bubble typing"><span></span><span></span><span></span></div>`);
  try {
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: question }),
    });
    typing.remove();
    renderAnswer(data, question);
  } catch (err) {
    typing.remove();
    addMessage("bot", `<div class="bubble">Something went wrong: ${escapeHtml(err.message)}</div>`, "ungrounded");
  } finally {
    busy = false;
    $("send").disabled = false;
    $("input").focus();
  }
}

async function upload(file) {
  if (!file) return;
  const hint = $("upload-hint");
  hint.textContent = `Indexing ${file.name}…`;
  const form = new FormData();
  form.append("file", file);
  try {
    const r = await api("/api/documents", { method: "POST", body: form });
    hint.textContent = `Added ${r.name}: ${r.pages} pages → ${r.chunks} chunks`;
    await loadDocs(r.name);
  } catch (err) {
    hint.textContent = `✗ ${err.message}`;
  }
}

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("input").value;
  $("input").value = "";
  ask(q);
});
$("suggestions").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") ask(e.target.textContent);
});
$("file").addEventListener("change", (e) => upload(e.target.files[0]));
const drop = $("upload");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("drag"); upload(e.dataTransfer.files[0]); });
$("menu").addEventListener("click", () => $("sidebar").classList.toggle("open"));
$("close-source").addEventListener("click", closeSource);
$("reset").addEventListener("click", async () => {
  await api("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message: "reset" }),
  });
  $("messages").innerHTML = `<div class="empty" id="empty"><div class="empty-icon">🔎</div><h3>New chat</h3><p>Ask anything about your PDFs.</p></div>`;
  closeSource();
});

loadStatus();
loadDocs();
