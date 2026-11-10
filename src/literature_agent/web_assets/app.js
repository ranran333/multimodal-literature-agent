"use strict";

const $ = (id) => document.getElementById(id);
const state = {token: "", documents: [], selected: null, busy: false, page: null};
let toastTimer;

function notify(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 4500);
}

function setStatus(message, kind = "ready") {
  const status = $("status");
  status.className = "status" + (kind === "ready" ? "" : ` ${kind}`);
  status.querySelector("span").textContent = message;
}

function refreshControls() {
  $("send-button").disabled = state.busy || !state.token || !state.documents.length || !$("question").value.trim();
  $("upload-trigger").disabled = state.busy || !state.token;
  $("file-input").disabled = state.busy || !state.token;
  $("visual-index").disabled = state.busy || !state.token;
  $("answer-mode").disabled = state.busy;
}

function setBusy(busy, message = "就绪") {
  state.busy = busy;
  setStatus(message, busy ? "busy" : "ready");
  refreshControls();
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  let result;
  try { result = await response.json(); }
  catch { throw new Error("服务没有返回有效结果，请检查服务端状态。"); }
  if (!response.ok) throw new Error(result.error || `请求失败（${response.status}）`);
  return result;
}

function docById(id) { return state.documents.find((document) => document.id === id); }

function chooseDocument(id, preview = true) {
  state.selected = id;
  const doc = id ? docById(id) : null;
  $("scope-label").textContent = doc ? `当前文献：${doc.name}` :
    state.documents.length ? `当前范围：全部 ${state.documents.length} 篇文献` : "上传论文后开始提问";
  renderDocuments();
  if (doc && preview) showPage(doc.id, 1);
}

function renderDocuments() {
  $("doc-count").textContent = String(state.documents.length);
  $("all-docs-detail").textContent = state.documents.length ? `${state.documents.length} 篇文献` : "尚未添加文献";
  const all = $("all-docs");
  all.classList.toggle("selected", !state.selected);
  all.setAttribute("aria-pressed", String(!state.selected));
  const list = $("document-list");
  list.replaceChildren();
  for (const doc of state.documents) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "doc-item" + (state.selected === doc.id ? " selected" : "");
    button.setAttribute("aria-pressed", String(state.selected === doc.id));
    button.title = doc.name;
    const icon = document.createElement("span"); icon.className = "doc-icon"; icon.textContent = "▧";
    const info = document.createElement("span"); info.className = "doc-name";
    const name = document.createElement("b"); name.textContent = doc.name;
    const detail = document.createElement("small");
    detail.textContent = `${doc.pages} 页${doc.visual_indexed ? " · 已增强图表检索" : ""}`;
    info.append(name, detail); button.append(icon, info);
    button.addEventListener("click", () => chooseDocument(doc.id));
    list.append(button);
  }
  refreshControls();
}

async function loadDocuments() {
  const data = await request("/api/documents");
  state.documents = data.documents;
  if (state.selected && !docById(state.selected)) state.selected = null;
  const selected = state.selected ? docById(state.selected) : null;
  $("scope-label").textContent = selected ? `当前文献：${selected.name}` :
    state.documents.length ? `当前范围：全部 ${state.documents.length} 篇文献` : "上传论文后开始提问";
  renderDocuments();
}

async function waitForJob(jobId, onProgress = () => {}) {
  while (true) {
    const job = await request(`/api/jobs/${encodeURIComponent(jobId)}`);
    onProgress(job.progress || "正在处理…");
    if (job.state === "done") return job.result;
    if (job.state === "error") throw new Error(job.error || "处理失败，请重试。");
    await new Promise((resolve) => setTimeout(resolve, 850));
  }
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length || state.busy) return;
  setBusy(true, "正在上传");
  try {
    for (const file of files) {
      if (!file.name.toLowerCase().endsWith(".pdf")) throw new Error(`“${file.name}”不是 PDF 文件。`);
      if (file.size > 50 * 1024 * 1024) throw new Error(`“${file.name}”超过 50 MB。`);
      const params = new URLSearchParams({filename: file.name, describe: $("visual-index").checked ? "1" : "0"});
      const response = await request(`/api/documents?${params}`, {
        method: "POST", headers: {"Content-Type": "application/pdf", "X-Local-Token": state.token}, body: file,
      });
      const result = await waitForJob(response.job_id, (progress) => setStatus(progress, "busy"));
      await loadDocuments();
      chooseDocument(result.document_id, false);
      notify(`已添加 ${result.name}`);
    }
  } catch (error) { notify(error.message); }
  finally { setBusy(false); $("file-input").value = ""; }
}

function scrollConversation() {
  const pane = $("conversation");
  requestAnimationFrame(() => { pane.scrollTop = pane.scrollHeight; });
}

function addUserMessage(question) {
  $("welcome").hidden = true;
  const message = document.createElement("div"); message.className = "message message-user";
  const bubble = document.createElement("div"); bubble.className = "user-bubble"; bubble.textContent = question;
  message.append(bubble); $("messages").append(message); scrollConversation();
}

function createAssistantMessage() {
  const message = document.createElement("div"); message.className = "message message-assistant";
  const heading = document.createElement("div"); heading.className = "assistant-heading";
  const mark = document.createElement("span"); mark.className = "assistant-mark"; mark.textContent = "文";
  const name = document.createElement("span"); name.textContent = "文献问答";
  heading.append(mark, name);
  const body = document.createElement("div"); body.className = "assistant-body";
  message.append(heading, body); $("messages").append(message); scrollConversation();
  return body;
}

function renderAnswer(body, result) {
  body.replaceChildren();
  const answer = document.createElement("div"); answer.className = "answer-text";
  const citations = new Map((result.citations || []).map((citation) => [citation.label, citation]));
  const raw = result.answer || "没有生成回答。";
  const pattern = /\[(E\d+)\]/g;
  let match, offset = 0;
  while ((match = pattern.exec(raw))) {
    answer.append(document.createTextNode(raw.slice(offset, match.index)));
    const citation = citations.get(match[1]);
    if (citation) {
      const button = document.createElement("button"); button.type = "button";
      button.className = "inline-cite"; button.textContent = match[0];
      button.addEventListener("click", () => showPage(citation.document_id, citation.page));
      answer.append(button);
    } else answer.append(document.createTextNode(match[0]));
    offset = pattern.lastIndex;
  }
  answer.append(document.createTextNode(raw.slice(offset))); body.append(answer);
  if (citations.size) {
    const label = document.createElement("div"); label.className = "citation-label"; label.textContent = "引用来源";
    const list = document.createElement("div"); list.className = "citation-list";
    for (const citation of citations.values()) {
      const button = document.createElement("button"); button.type = "button"; button.className = "citation";
      const number = document.createElement("b"); number.textContent = `[${citation.label}]`;
      const text = document.createElement("span"); text.textContent = `${citation.document} · 第 ${citation.page} 页`;
      button.append(number, text); button.addEventListener("click", () => showPage(citation.document_id, citation.page));
      list.append(button);
    }
    body.append(label, list);
  }
  const trace = result.trace || [];
  const route = result.route === "vision" ? "读取页面图像" : result.route === "text" ? "文字问答" : "未找到证据";
  const count = trace.filter((step) => step.tool === "read_page_image").length;
  const summary = document.createElement("div"); summary.className = "trace";
  summary.textContent = count ? `${route} · 已读取 ${count} 页图像` : route;
  body.append(summary); scrollConversation();
}

async function submitQuestion(question) {
  question = question.trim();
  if (!question || state.busy) return;
  if (!state.documents.length) { notify("请先上传一篇论文。"); return; }
  const documentId = state.selected;
  const mode = $("answer-mode").value;
  addUserMessage(question);
  $("question").value = ""; $("question").style.height = "";
  const body = createAssistantMessage();
  const waiting = document.createElement("div"); waiting.className = "waiting";
  const spinner = document.createElement("span"); spinner.className = "spinner";
  const label = document.createElement("span"); label.textContent = "正在查找相关内容…";
  waiting.append(spinner, label); body.append(waiting); setBusy(true, "正在思考");
  try {
    const data = await request("/api/ask", {
      method: "POST", headers: {"Content-Type": "application/json", "X-Local-Token": state.token},
      body: JSON.stringify({question, document_id: documentId, mode}),
    });
    const result = await waitForJob(data.job_id, (progress) => { label.textContent = progress; setStatus("正在思考", "busy"); });
    renderAnswer(body, result);
  } catch (error) {
    body.replaceChildren();
    const problem = document.createElement("div"); problem.className = "answer-text"; problem.textContent = error.message;
    body.append(problem); notify("本次问答未完成。");
  } finally { setBusy(false); scrollConversation(); }
}

function showPage(id, page) {
  const doc = docById(id);
  if (!doc) { notify("未找到引用的文献，请刷新页面。"); return; }
  page = Math.min(Math.max(1, Number(page) || 1), doc.pages);
  state.page = {id, number: page};
  $("source-panel").classList.add("open");
  $("source-empty").hidden = true; $("source-content").hidden = false;
  $("source-name").textContent = doc.name;
  $("page-label").textContent = `第 ${page} / ${doc.pages} 页`;
  $("prev-page").disabled = page <= 1; $("next-page").disabled = page >= doc.pages;
  $("open-pdf").href = `/api/documents/${encodeURIComponent(id)}/file#page=${page}`;
  $("page-error").hidden = true;
  const image = $("page-image");
  image.hidden = false; image.alt = `${doc.name} 第 ${page} 页`;
  image.src = `/api/documents/${encodeURIComponent(id)}/pages/${page}.png`;
}

async function initialize() {
  try {
    const config = await request("/api/config");
    state.token = config.token; state.documents = config.documents;
    $("upload-hint").textContent = `单份不超过 ${config.max_pdf_mb} MB、${config.max_pages} 页`;
    renderDocuments();
    if (config.busy) {
      setBusy(true, "正在处理");
      notify("服务正在处理先前的任务，请稍候。");
      const check = setInterval(async () => {
        try {
          const next = await request("/api/config");
          if (!next.busy) { clearInterval(check); await loadDocuments(); setBusy(false); }
        } catch { clearInterval(check); setStatus("连接失败", "error"); }
      }, 1500);
    }
    else setBusy(false);
    if (state.documents.length) $("scope-label").textContent = `当前范围：全部 ${state.documents.length} 篇文献`;
  } catch (error) { setStatus("连接失败", "error"); notify(error.message); }
}

$("upload-trigger").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) => uploadFiles(event.target.files));
$("all-docs").addEventListener("click", () => chooseDocument(null));
$("question-form").addEventListener("submit", (event) => { event.preventDefault(); submitQuestion($("question").value); });
$("question").addEventListener("input", (event) => {
  event.target.style.height = "auto"; event.target.style.height = `${Math.min(event.target.scrollHeight, 135)}px`;
  refreshControls();
});
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault(); submitQuestion(event.target.value);
  }
});
for (const button of document.querySelectorAll("[data-prompt]")) {
  button.addEventListener("click", () => { $("question").value = button.dataset.prompt; $("question").focus(); refreshControls(); });
}
$("clear-chat").addEventListener("click", () => { $("messages").replaceChildren(); $("welcome").hidden = false; });
$("open-tech").addEventListener("click", () => $("tech-dialog").showModal());
$("close-tech").addEventListener("click", () => $("tech-dialog").close());
$("tech-dialog").addEventListener("click", (event) => { if (event.target === $("tech-dialog")) $("tech-dialog").close(); });
$("close-preview").addEventListener("click", () => $("source-panel").classList.remove("open"));
$("prev-page").addEventListener("click", () => state.page && showPage(state.page.id, state.page.number - 1));
$("next-page").addEventListener("click", () => state.page && showPage(state.page.id, state.page.number + 1));
$("page-image").addEventListener("error", () => { $("page-image").hidden = true; $("page-error").textContent = "无法加载页面，请确认原 PDF 未被移动或修改。"; $("page-error").hidden = false; });
const zone = $("upload-zone");
for (const eventName of ["dragenter", "dragover"]) zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.add("dragging"); });
for (const eventName of ["dragleave", "drop"]) zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.remove("dragging"); });
zone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));
initialize();
