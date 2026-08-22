/**
 * DevDocs AI — Frontend Application
 *
 * Anonymous, session-based app: no sign-in wall. Every browser tab/profile
 * gets a private, persistent session id (localStorage `dd_session`) sent as
 * `X-Session-Id` on every backend request. Free daily questions are capped
 * server-side; users can optionally paste their own Gemini API key (BYOK,
 * localStorage `dd_api_key`) to bypass that limit.
 *
 * Handles streaming chat with per-session conversation history, source
 * ingestion (repo / docs / PDF / topic search), source management, and
 * public metrics.
 */

// ── State ───────────────────────────────────────────────────
const API = window.location.origin;
let sessionId = null;
let apiKey = null;
let isStreaming = false;
let currentAbortController = null;
let currentConversationId = null;
let lastQuestion = null; // remembered so a BYOK save can retry the blocked question
let usageState = null; // { used, limit, remaining } from GET /usage

// ── DOM refs ────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const messagesEl = $("#messages");
const questionInput = $("#question-input");
const sendBtn = $("#send-btn");
const stopBtn = $("#stop-btn");
const sidebar = $("#sidebar");
const menuToggle = $("#menu-toggle");
const sidebarBackdrop = $("#sidebar-backdrop");
const confirmModal = $("#confirm-modal");
const confirmModalTitle = $("#confirm-modal-title");
const confirmModalBody = $("#confirm-modal-body");
const apiKeyModal = $("#api-key-modal");

// The welcome-message markup, captured once at load so startNewChat() can
// restore #messages to its pristine state.
const WELCOME_HTML = messagesEl.innerHTML;

// ── Init ────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    sessionId = getOrCreateSessionId();
    apiKey = localStorage.getItem("dd_api_key") || null;
    updateKeyStatusBadge();
    setupEventListeners();
    loadUsage();
});

// ── Anonymous session identity ─────────────────────────────
function generateUUID() {
    if (window.crypto && typeof crypto.randomUUID === "function") return crypto.randomUUID();
    // Fallback for contexts without crypto.randomUUID (e.g. non-HTTPS) —
    // RFC4122-ish v4 UUID built from getRandomValues.
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0"));
    return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10, 16).join("")}`;
}

function getOrCreateSessionId() {
    let id = localStorage.getItem("dd_session");
    if (!id) {
        id = generateUUID();
        localStorage.setItem("dd_session", id);
    }
    return id;
}

/** Headers sent on every backend fetch. */
function sessionHeaders(extra = {}) {
    return { "X-Session-Id": sessionId, ...extra };
}

// ── Event Listeners ─────────────────────────────────────────
function setupEventListeners() {
    // Chat input
    questionInput.addEventListener("input", () => {
        sendBtn.disabled = !questionInput.value.trim() || isStreaming;
        autoResize(questionInput);
    });

    questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled) sendMessage();
        }
    });

    sendBtn.addEventListener("click", sendMessage);
    stopBtn.addEventListener("click", () => currentAbortController?.abort());
    $("#new-chat-btn").addEventListener("click", startNewChat);
    $("#new-chat-from-history").addEventListener("click", () => {
        startNewChat();
        switchView("chat");
        closeSidebar();
    });

    // Quick action buttons (use closest() so a click on the icon/text inside
    // the button — not just the button element itself — still registers)
    document.addEventListener("click", (e) => {
        const btn = e.target.closest(".quick-btn");
        if (btn) {
            questionInput.value = btn.dataset.question;
            sendBtn.disabled = false;
            sendMessage();
        }
    });

    // Nav items
    $$(".nav-item").forEach((item) => {
        item.addEventListener("click", () => {
            switchView(item.dataset.view);
            closeSidebar();
        });
    });

    // API key (BYOK) settings
    $("#api-key-btn").addEventListener("click", openApiKeyModal);
    $("#api-key-cancel").addEventListener("click", closeApiKeyModal);
    $("#api-key-clear").addEventListener("click", () => {
        clearApiKey();
        closeApiKeyModal();
    });
    $("#api-key-save").addEventListener("click", saveApiKeyFromModal);
    apiKeyModal.addEventListener("click", (e) => {
        if (e.target === apiKeyModal) closeApiKeyModal();
    });

    // Ingest: source-type tabs
    $$(".ingest-tab").forEach((tab) => {
        tab.addEventListener("click", () => {
            const target = tab.dataset.sourceTab;
            $$(".ingest-tab").forEach((t) => {
                t.classList.toggle("active", t === tab);
                t.setAttribute("aria-pressed", t === tab ? "true" : "false");
            });
            $$(".ingest-panel").forEach((p) => {
                p.classList.toggle("hidden", p.dataset.panel !== target);
            });
            const statusEl = $("#ingest-status");
            statusEl.classList.add("hidden");
        });
    });

    // Ingest forms
    $("#ingest-github-form").addEventListener("submit", handleGithubIngest);
    $("#ingest-docs-form").addEventListener("submit", handleDocsIngest);
    $("#ingest-topic-form").addEventListener("submit", handleTopicSearch);
    setupPdfDropzone();

    // My Sources
    $("#refresh-sources").addEventListener("click", loadMySources);

    // Refresh metrics
    $("#refresh-metrics").addEventListener("click", loadMetrics);

    // Delete-conversation modal
    $("#confirm-cancel").addEventListener("click", () => closeConfirmModal("cancel"));
    $("#confirm-history-only").addEventListener("click", () => closeConfirmModal("history"));
    $("#confirm-history-docs").addEventListener("click", () => closeConfirmModal("history+docs"));
    confirmModal.addEventListener("click", (e) => {
        if (e.target === confirmModal) closeConfirmModal("cancel");
    });

    // Mobile sidebar (hamburger + backdrop + Escape) and modal Escape/Tab trap
    menuToggle.addEventListener("click", () => {
        if (sidebar.classList.contains("open")) closeSidebar();
        else openSidebar();
    });
    sidebarBackdrop.addEventListener("click", closeSidebar);
    document.addEventListener("keydown", (e) => {
        if (!confirmModal.classList.contains("hidden")) {
            if (e.key === "Escape") {
                closeConfirmModal("cancel");
                return;
            }
            if (e.key === "Tab") trapModalFocus(confirmModal, e);
            return;
        }
        if (!apiKeyModal.classList.contains("hidden")) {
            if (e.key === "Escape") {
                closeApiKeyModal();
                return;
            }
            if (e.key === "Tab") trapModalFocus(apiKeyModal, e);
            return;
        }
        if (e.key === "Escape" && sidebar.classList.contains("open")) closeSidebar();
    });
}

function openSidebar() {
    sidebar.classList.add("open");
    sidebarBackdrop.classList.add("open");
    menuToggle.setAttribute("aria-expanded", "true");
}

function closeSidebar() {
    sidebar.classList.remove("open");
    sidebarBackdrop.classList.remove("open");
    menuToggle.setAttribute("aria-expanded", "false");
}

// ── BYOK: API key settings ─────────────────────────────────
let apiKeyModalPrevFocus = null;

function updateKeyStatusBadge() {
    const el = $("#key-status-badge");
    if (el) el.textContent = apiKey ? "API key set" : "No API key";
}

function setApiKey(key) {
    apiKey = key;
    localStorage.setItem("dd_api_key", key);
    updateKeyStatusBadge();
    renderUsageIndicator();
}

function clearApiKey() {
    apiKey = null;
    localStorage.removeItem("dd_api_key");
    updateKeyStatusBadge();
    renderUsageIndicator();
}

function openApiKeyModal() {
    $("#api-key-input").value = apiKey || "";
    $("#api-key-modal-error").classList.add("hidden");
    apiKeyModalPrevFocus = document.activeElement;
    apiKeyModal.classList.remove("hidden");
    $("#api-key-input").focus();
}

function closeApiKeyModal() {
    apiKeyModal.classList.add("hidden");
    if (apiKeyModalPrevFocus && typeof apiKeyModalPrevFocus.focus === "function") {
        apiKeyModalPrevFocus.focus();
    }
    apiKeyModalPrevFocus = null;
}

function saveApiKeyFromModal() {
    const key = $("#api-key-input").value.trim();
    if (!key) {
        const err = $("#api-key-modal-error");
        err.textContent = "Enter a key, or use Clear key to remove the saved one.";
        err.classList.remove("hidden");
        return;
    }
    setApiKey(key);
    closeApiKeyModal();
}

function trapModalFocus(modalEl, e) {
    const focusables = Array.from(modalEl.querySelectorAll("button, input"));
    if (!focusables.length) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
    }
}

// ── Usage (free daily questions) ───────────────────────────
async function loadUsage() {
    try {
        const res = await fetch(`${API}/usage`, { headers: sessionHeaders() });
        if (!res.ok) throw new Error("Failed to load usage");
        usageState = await res.json();
    } catch {
        usageState = null;
    }
    renderUsageIndicator();
}

function renderUsageIndicator() {
    const el = $("#usage-indicator");
    if (!el) return;
    if (apiKey) {
        el.textContent = "Using your own API key — no daily limit.";
        el.classList.remove("warning");
        return;
    }
    if (!usageState || typeof usageState.remaining !== "number") {
        el.textContent = "";
        el.classList.remove("warning");
        return;
    }
    const { remaining, limit } = usageState;
    el.textContent = `${remaining} of ${limit} free questions left today`;
    el.classList.toggle("warning", remaining <= 0);
}

// ── App Shell ───────────────────────────────────────────────
function switchView(viewName) {
    $$(".view").forEach((v) => v.classList.remove("active"));
    $$(".nav-item").forEach((n) => n.classList.remove("active"));

    const view = $(`#view-${viewName}`);
    const nav = $(`[data-view="${viewName}"]`);
    if (view) view.classList.add("active");
    if (nav) nav.classList.add("active");

    // Load data for the relevant view
    if (viewName === "metrics") loadMetrics();
    if (viewName === "ingest") loadMySources();
    if (viewName === "history") loadHistory();
}

// ── Chat ────────────────────────────────────────────────────
function updateChatTitle(title) {
    $("#chat-conversation-title").textContent = title || "New conversation";
}

function startNewChat() {
    if (isStreaming) currentAbortController?.abort();
    currentConversationId = null;
    messagesEl.innerHTML = WELCOME_HTML;
    updateChatTitle(null);
    questionInput.value = "";
    sendBtn.disabled = true;
    autoResize(questionInput);
}

async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question || isStreaming) return;

    // Hide welcome
    const welcome = $(".welcome-message");
    if (welcome) welcome.remove();

    // Add user message
    appendMessage("user", question, "You");

    // Clear input
    questionInput.value = "";
    sendBtn.disabled = true;
    autoResize(questionInput);
    lastQuestion = question;

    // Add assistant message with typing indicator
    const msgEl = appendMessage("assistant", "", "DevDocs AI");
    const contentEl = msgEl.querySelector(".message-content");
    contentEl.innerHTML = `<div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    </div>`;

    isStreaming = true;
    sendBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");
    const controller = new AbortController();
    currentAbortController = controller;

    // Throttle rendering to one animation frame — re-parsing markdown on every
    // chunk is O(n^2) DOM work that janks on long answers.
    let fullText = "";
    let sources = [];
    let renderQueued = false;
    let firstToken = true;

    const paint = () => {
        renderQueued = false;
        contentEl.innerHTML = renderMarkdown(fullText) + renderSources(sources);
        scrollToBottom();
    };
    const schedulePaint = () => {
        if (renderQueued) return;
        renderQueued = true;
        requestAnimationFrame(paint);
    };

    try {
        const headers = {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
            ...sessionHeaders(),
        };
        if (apiKey) headers["X-Api-Key"] = apiKey;

        const res = await fetch(`${API}/ask`, {
            method: "POST",
            headers,
            body: JSON.stringify({
                question,
                k: 5,
                ...(currentConversationId ? { conversation_id: currentConversationId } : {}),
            }),
            signal: controller.signal,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Request failed" }));
            throw new Error(err.detail || `Error ${res.status}`);
        }

        // ── Server-Sent Events ──────────────────────────────
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let streamError = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // { stream: true } holds an incomplete multi-byte UTF-8 character
            // (an emoji, an accent) split across two network chunks until the
            // next read — without it you get intermittent replacement chars.
            buffer += decoder.decode(value, { stream: true });

            // Events are separated by a blank line; the trailing partial event
            // stays in the buffer for the next iteration.
            const frames = buffer.split("\n\n");
            buffer = frames.pop() ?? "";

            for (const frame of frames) {
                const evt = parseSSE(frame);
                if (!evt) continue;

                if (evt.event === "meta") {
                    if (evt.data.conversation_id) {
                        currentConversationId = evt.data.conversation_id;
                        updateChatTitle(question.length > 60 ? `${question.slice(0, 60)}…` : question);
                        loadHistory().catch(() => {}); // refresh in the background
                    }
                } else if (evt.event === "token") {
                    if (firstToken) {
                        contentEl.innerHTML = "";
                        firstToken = false;
                    }
                    fullText += evt.data.text ?? "";
                    schedulePaint();
                } else if (evt.event === "sources") {
                    sources = evt.data.sources ?? [];
                    schedulePaint();
                } else if (evt.event === "error") {
                    streamError = evt.data || {};
                }
                // "done" carries no payload we need — it just precedes stream close.
            }
        }

        paint(); // final synchronous render — never leave a queued frame unpainted

        if (streamError) {
            if (streamError.code === "limit_reached" || streamError.code === "quota") {
                appendLimitPrompt(contentEl, question);
            } else {
                contentEl.innerHTML +=
                    `<div class="status-message error">⚠️ ${escapeHtml(streamError.message || "Stream failed")}</div>`;
            }
            scrollToBottom();
        }
        attachMessageActions(msgEl, fullText, question);
    } catch (err) {
        if (err.name === "AbortError") {
            if (firstToken) contentEl.innerHTML = "";
            contentEl.innerHTML += `<div class="status-message info">⏹ Stopped.</div>`;
        } else {
            contentEl.innerHTML = `<span style="color: var(--error)">⚠️ ${escapeHtml(err.message)}</span>`;
        }
        attachMessageActions(msgEl, fullText, question);
    } finally {
        isStreaming = false;
        currentAbortController = null;
        stopBtn.classList.add("hidden");
        sendBtn.classList.remove("hidden");
        sendBtn.disabled = !questionInput.value.trim();
        scrollToBottom();
        loadUsage();
    }
}

function retryQuestion(question) {
    if (isStreaming) return;
    questionInput.value = question;
    sendBtn.disabled = false;
    sendMessage();
}

/**
 * Inline prompt shown inside an assistant message when the stream reports
 * `limit_reached` or `quota` — lets the user paste a Gemini API key right
 * there and immediately retry the question that got blocked.
 */
function appendLimitPrompt(contentEl, question) {
    const wrap = document.createElement("div");
    wrap.className = "limit-prompt";

    const msg = document.createElement("p");
    msg.textContent = "You've used your free questions today — add your Gemini API key to continue.";
    wrap.appendChild(msg);

    const form = document.createElement("form");
    form.className = "limit-prompt-form";

    const input = document.createElement("input");
    input.type = "password";
    input.placeholder = "Paste your Gemini API key";
    input.setAttribute("aria-label", "Gemini API key");
    input.autocomplete = "off";
    input.spellcheck = false;
    form.appendChild(input);

    const btn = document.createElement("button");
    btn.type = "submit";
    btn.className = "btn-primary";
    btn.textContent = "Save key & retry";
    form.appendChild(btn);

    const err = document.createElement("div");
    err.className = "status-message error hidden";
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        const key = input.value.trim();
        if (!key) {
            err.textContent = "Enter a key first.";
            err.classList.remove("hidden");
            return;
        }
        setApiKey(key);
        wrap.remove();
        retryQuestion(question);
    });

    wrap.appendChild(form);
    wrap.appendChild(err);
    contentEl.appendChild(wrap);
}

/**
 * Add a Copy button (and, for the most recent answer, a Retry button) below
 * an assistant message once its final text is known.
 */
function attachMessageActions(msgEl, finalText, question) {
    const actions = msgEl.querySelector(".message-actions");
    if (!actions) return;
    actions.innerHTML = "";
    actions.classList.remove("hidden");

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "message-action-btn";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText(finalText || "");
            copyBtn.textContent = "Copied!";
        } catch {
            copyBtn.textContent = "Copy failed";
        } finally {
            setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
        }
    });
    actions.appendChild(copyBtn);

    if (question) {
        // Only the latest answer should offer Retry.
        $$(".message-action-btn.retry-btn").forEach((b) => b.remove());
        const retryBtn = document.createElement("button");
        retryBtn.type = "button";
        retryBtn.className = "message-action-btn retry-btn";
        retryBtn.textContent = "Retry";
        retryBtn.addEventListener("click", () => retryQuestion(question));
        actions.appendChild(retryBtn);
    }
}

/**
 * Parse one SSE frame into { event, data }. Returns null for frames we can't
 * use (comments, malformed JSON) rather than throwing mid-stream.
 */
function parseSSE(frame) {
    let event = "message";
    const dataLines = [];
    for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return null;
    try {
        return { event, data: JSON.parse(dataLines.join("\n")) };
    } catch {
        return null;
    }
}

function renderSources(sources) {
    if (!sources || !sources.length) return "";
    return `<div class="sources-panel">
        <div class="sources-label">📄 Sources</div>
        ${sources.map((s) => `<div class="source-item">${escapeHtml(s)}</div>`).join("")}
    </div>`;
}

function appendMessage(role, content, sender) {
    const avatarContent =
        role === "user"
            ? sender[0].toUpperCase()
            : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>`;

    const el = document.createElement("div");
    el.className = `message ${role}`;
    el.innerHTML = `
        <div class="message-avatar">${avatarContent}</div>
        <div class="message-body">
            <div class="message-sender">${escapeHtml(sender)}</div>
            <div class="message-content">${content ? renderMarkdown(content) : ""}</div>
            ${role === "assistant" ? '<div class="message-actions hidden"></div>' : ""}
        </div>
    `;

    messagesEl.appendChild(el);
    scrollToBottom();
    return el;
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Chat History (conversations) ───────────────────────────
async function loadHistory() {
    const list = $("#history-list");
    list.innerHTML = `<div class="empty-state">Loading…</div>`;
    try {
        const res = await fetch(`${API}/conversations`, { headers: sessionHeaders() });
        if (!res.ok) throw new Error("Failed to load conversations");
        const data = await res.json();
        const convs = (data.conversations || []).slice().sort((a, b) => {
            const at = new Date(a.updated || a.created || 0).getTime();
            const bt = new Date(b.updated || b.created || 0).getTime();
            return bt - at;
        });

        if (!convs.length) {
            list.innerHTML = `<div class="empty-state">No conversations yet. Start a new chat to see it here.</div>`;
            return;
        }

        list.innerHTML = "";
        convs.forEach((c) => list.appendChild(renderHistoryRow(c)));
    } catch (err) {
        list.innerHTML = `<div class="empty-state">Failed to load history.</div>`;
    }
}

function renderHistoryRow(conv) {
    const row = document.createElement("div");
    row.className = "history-row" + (conv.id === currentConversationId ? " active" : "");

    const title = conv.title || "Untitled conversation";
    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "history-row-main";
    openBtn.innerHTML = `
        <span class="history-row-title">${escapeHtml(title)}</span>
        <span class="history-row-meta">${escapeHtml(conv.updated || conv.created || "")}</span>
    `;
    openBtn.addEventListener("click", async () => {
        await loadConversation(conv.id);
        switchView("chat");
        closeSidebar();
    });

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "btn-ghost history-delete-btn";
    delBtn.setAttribute("aria-label", `Delete conversation "${title}"`);
    delBtn.title = "Delete conversation";
    delBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>`;
    delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteConversation(conv.id, title);
    });

    row.appendChild(openBtn);
    row.appendChild(delBtn);
    return row;
}

async function loadConversation(id) {
    try {
        const res = await fetch(`${API}/conversations/${id}`, { headers: sessionHeaders() });
        if (!res.ok) throw new Error("Failed to load conversation");
        const data = await res.json();

        currentConversationId = data.id;
        messagesEl.innerHTML = "";

        const messages = data.messages || [];
        messages.forEach((m, i) => {
            const isUser = m.role === "user";
            const sender = isUser ? "You" : "DevDocs AI";
            const el = appendMessage(isUser ? "user" : "assistant", "", sender);
            const contentEl = el.querySelector(".message-content");
            contentEl.innerHTML = renderMarkdown(m.content) + renderSources(m.sources);

            if (!isUser) {
                const precedingQuestion = messages[i - 1]?.role === "user" ? messages[i - 1].content : null;
                const isLast = i === messages.length - 1;
                attachMessageActions(el, m.content, isLast ? precedingQuestion : null);
            }
        });

        updateChatTitle(data.title);
        scrollToBottom();
    } catch (err) {
        alert(`Failed to load conversation: ${err.message}`);
    }
}

async function deleteConversation(id, title) {
    const choice = await openDeleteConversationModal(title);
    if (choice === "cancel" || !choice) return;

    try {
        const res = await fetch(`${API}/conversations/${id}`, {
            method: "DELETE",
            headers: sessionHeaders(),
        });
        if (!res.ok) throw new Error("Delete failed");

        if (choice === "history+docs") await deleteMyUploadedDocuments();
        if (currentConversationId === id) startNewChat();
        await loadHistory();
    } catch (err) {
        alert(`Failed to delete conversation: ${err.message}`);
    }
}

/**
 * Best-effort deletion of every source ingested under *this session* — used
 * by the "history + my documents" delete option. Failures for individual
 * sources are swallowed so one bad source doesn't block the rest;
 * loadMySources()/loadHistory() reflect whatever actually succeeded.
 */
async function deleteMyUploadedDocuments() {
    try {
        const res = await fetch(`${API}/sources/mine`, { headers: sessionHeaders() });
        if (!res.ok) return;
        const data = await res.json();
        const uploaded = (data.sources || []).filter((s) => {
            const kind = (s.kind || "").toLowerCase();
            return kind.includes("pdf") || kind.includes("upload");
        });
        await Promise.all(
            uploaded.map((s) =>
                fetch(`${API}/sources`, {
                    method: "DELETE",
                    headers: sessionHeaders({ "Content-Type": "application/json" }),
                    body: JSON.stringify({ source: s.source }),
                }).catch(() => {})
            )
        );
    } catch {
        // Best-effort — the conversation delete above already succeeded.
    } finally {
        if ($("#view-ingest").classList.contains("active")) loadMySources();
    }
}

// ── Delete-conversation confirmation modal ─────────────────
let modalResolve = null;
let modalPrevFocus = null;

function openDeleteConversationModal(title) {
    confirmModalTitle.textContent = "Delete conversation";
    confirmModalBody.textContent =
        `Delete "${title || "this conversation"}"? Choose whether to also remove documents you personally uploaded. This can't be undone.`;
    modalPrevFocus = document.activeElement;
    confirmModal.classList.remove("hidden");
    $("#confirm-history-only").focus();
    return new Promise((resolve) => {
        modalResolve = resolve;
    });
}

function closeConfirmModal(choice) {
    confirmModal.classList.add("hidden");
    if (modalResolve) modalResolve(choice);
    modalResolve = null;
    if (modalPrevFocus && typeof modalPrevFocus.focus === "function") modalPrevFocus.focus();
    modalPrevFocus = null;
}

// ── Ingest: GitHub repo / Docs URL ─────────────────────────
// /ingest now returns 202 + a job_id immediately, because cloning + chunking +
// embedding a real repository takes minutes and load balancers cut idle
// connections long before that. We poll for the outcome instead.
async function ingestSource(source, show) {
    show("⏳ Queued — cloning and embedding. This can take a few minutes...", "info");

    const res = await fetch(`${API}/ingest`, {
        method: "POST",
        headers: sessionHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ source }),
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Ingestion failed");
    }

    const { job_id } = await res.json();
    const job = await pollIngestJob(job_id, (state) =>
        show(`⏳ ${state}... (job ${job_id})`, "info")
    );

    if (job.status !== "succeeded") {
        throw new Error(job.error || "Ingest failed");
    }

    show(`✅ Ingested ${job.chunks_added} chunks — ${job.total_chunks} total in DB`, "success");
    loadMySources().catch(() => {});
    return job;
}

async function submitIngestForm(form, source) {
    const statusEl = $("#ingest-status");
    const btn = form.querySelector(".btn-primary");

    setLoading(btn, true);
    statusEl.classList.add("hidden");

    const show = (text, kind) => {
        statusEl.textContent = text;
        statusEl.className = `status-message ${kind}`;
        statusEl.classList.remove("hidden");
    };

    try {
        await ingestSource(source, show);
        form.reset();
    } catch (err) {
        show(`❌ ${err.message}`, "error");
    } finally {
        setLoading(btn, false);
    }
}

async function handleGithubIngest(e) {
    e.preventDefault();
    const source = $("#ingest-github-url").value.trim();
    if (!source) return;
    await submitIngestForm(e.target, source);
}

async function handleDocsIngest(e) {
    e.preventDefault();
    const source = $("#ingest-docs-url").value.trim();
    if (!source) return;
    await submitIngestForm(e.target, source);
}

async function pollIngestJob(jobId, onProgress, { intervalMs = 2000, timeoutMs = 900000 } = {}) {
    const deadline = Date.now() + timeoutMs;
    let last = "";
    while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, intervalMs));
        const res = await fetch(`${API}/ingest/${jobId}`, { headers: sessionHeaders() });
        if (!res.ok) throw new Error("Lost track of the ingest job");
        const job = await res.json();
        if (job.status === "succeeded" || job.status === "failed") return job;
        if (job.status !== last) {
            last = job.status;
            onProgress?.(job.status);
        }
    }
    throw new Error("Timed out waiting for the ingest job");
}

// ── Ingest: PDF upload (drag & drop) ───────────────────────
const PDF_MAX_BYTES = 20 * 1024 * 1024;

function setupPdfDropzone() {
    const dropzone = $("#pdf-dropzone");
    const fileInput = $("#pdf-file-input");

    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInput.click();
        }
    });

    ["dragenter", "dragover"].forEach((evt) =>
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.add("dragover");
        })
    );
    ["dragleave", "drop"].forEach((evt) =>
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            dropzone.classList.remove("dragover");
        })
    );
    dropzone.addEventListener("drop", (e) => {
        const file = e.dataTransfer?.files?.[0];
        if (file) handlePdfFile(file);
    });

    fileInput.addEventListener("change", () => {
        const file = fileInput.files?.[0];
        if (file) handlePdfFile(file);
    });
}

function showPdfStatus(text, kind) {
    const statusEl = $("#pdf-status");
    statusEl.textContent = text;
    statusEl.className = `status-message ${kind}`;
    statusEl.classList.remove("hidden");
}

function handlePdfFile(file) {
    $("#pdf-status").classList.add("hidden");

    // Client-side validation before we ever hit the network.
    const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
    if (!isPdf) {
        $("#pdf-file-info").classList.add("hidden");
        showPdfStatus("❌ Only PDF files are supported.", "error");
        $("#pdf-file-input").value = "";
        return;
    }
    if (file.size === 0) {
        $("#pdf-file-info").classList.add("hidden");
        showPdfStatus("❌ File is empty.", "error");
        $("#pdf-file-input").value = "";
        return;
    }
    if (file.size > PDF_MAX_BYTES) {
        $("#pdf-file-info").classList.add("hidden");
        showPdfStatus("❌ File is larger than 20MB.", "error");
        $("#pdf-file-input").value = "";
        return;
    }

    const info = $("#pdf-file-info");
    info.textContent = `${file.name} — ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    info.classList.remove("hidden");
    uploadPdf(file);
}

async function uploadPdf(file) {
    showPdfStatus("⏳ Uploading…", "info");
    const formData = new FormData();
    formData.append("file", file);

    try {
        // No Content-Type header here — the browser sets multipart/form-data
        // with the correct boundary itself; overriding it breaks parsing.
        const res = await fetch(`${API}/ingest/upload`, {
            method: "POST",
            headers: sessionHeaders(),
            body: formData,
        });

        if (res.status === 415) throw new Error("Only PDF files are accepted.");
        if (res.status === 413) throw new Error("File is too large (max 20MB).");
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || "Upload failed");
        }

        const { job_id } = await res.json();
        showPdfStatus("⏳ Queued — extracting and embedding. This can take a few minutes...", "info");
        const job = await pollIngestJob(job_id, (state) =>
            showPdfStatus(`⏳ ${state}... (job ${job_id})`, "info")
        );

        if (job.status === "succeeded") {
            showPdfStatus(`✅ Ingested ${job.chunks_added} chunks — ${job.total_chunks} total in DB`, "success");
            loadMySources().catch(() => {});
        } else {
            showPdfStatus(`❌ Ingest failed: ${job.error || "unknown error"}`, "error");
        }
    } catch (err) {
        showPdfStatus(`❌ ${err.message}`, "error");
    } finally {
        $("#pdf-file-input").value = "";
    }
}

// ── Ingest: Topic search ────────────────────────────────────
async function handleTopicSearch(e) {
    e.preventDefault();
    const query = $("#ingest-topic-query").value.trim();
    if (!query) return;

    const resultsEl = $("#topic-results");
    const btn = e.target.querySelector("button[type=submit]");

    setLoading(btn, true);
    resultsEl.classList.remove("hidden");
    resultsEl.innerHTML = `<div class="empty-state">Searching…</div>`;

    try {
        const res = await fetch(`${API}/search-sources`, {
            method: "POST",
            headers: sessionHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ query }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || "Search failed");
        }

        const data = await res.json();
        if (!data.results || !data.results.length) {
            resultsEl.innerHTML = `<div class="empty-state">${escapeHtml(data.message || "No candidates found.")}</div>`;
            return;
        }

        resultsEl.innerHTML = "";
        data.results.forEach((r) => resultsEl.appendChild(renderTopicResultRow(r)));
    } catch (err) {
        resultsEl.innerHTML = `<div class="status-message error">❌ ${escapeHtml(err.message)}</div>`;
    } finally {
        setLoading(btn, false);
    }
}

function renderTopicResultRow(result) {
    const row = document.createElement("div");
    row.className = "topic-result-row";
    row.innerHTML = `
        <div class="topic-result-main">
            <span class="topic-result-title">${escapeHtml(result.title || result.url)}</span>
            <span class="topic-result-url">${escapeHtml(result.url)}</span>
        </div>
    `;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-secondary topic-ingest-btn";
    btn.textContent = "Ingest";
    row.appendChild(btn);

    const statusEl = document.createElement("div");
    statusEl.className = "status-message hidden topic-result-status";
    statusEl.setAttribute("role", "status");
    statusEl.setAttribute("aria-live", "polite");
    row.appendChild(statusEl);

    const show = (text, kind) => {
        statusEl.textContent = text;
        statusEl.className = `status-message topic-result-status ${kind}`;
        statusEl.classList.remove("hidden");
    };

    btn.addEventListener("click", async () => {
        btn.disabled = true;
        try {
            await ingestSource(result.url, show);
            btn.textContent = "Ingested ✓";
        } catch (err) {
            show(`❌ ${err.message}`, "error");
            btn.disabled = false;
            btn.textContent = "Retry";
        }
    });

    return row;
}

// ── My Sources ──────────────────────────────────────────────
async function loadMySources() {
    const list = $("#my-sources-list");
    list.innerHTML = `<div class="empty-state">Loading…</div>`;
    try {
        const res = await fetch(`${API}/sources/mine`, { headers: sessionHeaders() });
        if (!res.ok) throw new Error("Failed to load sources");
        const data = await res.json();

        if (!data.sources || !data.sources.length) {
            list.innerHTML = `<div class="empty-state">No sources ingested yet.</div>`;
            return;
        }

        list.innerHTML = "";
        data.sources.forEach((s) => list.appendChild(renderSourceRow(s)));
    } catch (err) {
        list.innerHTML = `<div class="empty-state">Failed to load sources.</div>`;
    }
}

function renderSourceRow(s) {
    const row = document.createElement("div");
    row.className = "source-row";
    row.innerHTML = `
        <div class="source-row-main">
            <span class="source-row-title">${escapeHtml(s.source)}</span>
            <span class="source-row-meta">${escapeHtml(s.kind || "")} · ${s.chunks ?? 0} chunks · ${escapeHtml(s.created || "")}</span>
        </div>
    `;

    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.className = "btn-ghost source-delete-btn";
    delBtn.setAttribute("aria-label", `Delete source ${s.source}`);
    delBtn.title = "Delete source";
    delBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>`;
    delBtn.addEventListener("click", () => deleteSource(s.source));
    row.appendChild(delBtn);

    return row;
}

async function deleteSource(source) {
    if (!confirm(`Delete "${source}" and all its ingested chunks? This can't be undone.`)) return;
    try {
        const res = await fetch(`${API}/sources`, {
            method: "DELETE",
            headers: sessionHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify({ source }),
        });
        if (!res.ok) throw new Error("Delete failed");
        await loadMySources();
    } catch (err) {
        alert(`Failed to delete source: ${err.message}`);
    }
}

// ── Metrics ─────────────────────────────────────────────────
async function loadMetrics() {
    try {
        const res = await fetch(`${API}/metrics`, { headers: sessionHeaders() });
        if (!res.ok) throw new Error("Failed to load metrics");
        const data = await res.json();

        const set = (id, val) => {
            const el = $(`#${id}`);
            if (el) el.textContent = val;
        };

        set("m-total-requests", data.total_requests ?? 0);
        set("m-ask-requests", data.ask_requests ?? 0);
        set("m-p95-latency", data.p95_latency_ms ?? 0);
        set("m-avg-latency", data.avg_latency_ms ?? 0);
        set("m-cache-hit-rate", ((data.embedding_cache_hit_rate ?? 0) * 100).toFixed(1));
        set("m-cache-size", data.embedding_cache_size ?? 0);
        set("m-llm-calls", data.llm_calls ?? 0);
        set("m-llm-avg", data.llm_avg_ms ?? 0);
        set("m-errors", data.errors ?? 0);
    } catch (err) {
        console.error("Metrics load failed:", err);
    }
}

// ── Utilities ───────────────────────────────────────────────
function setLoading(btn, loading) {
    const text = btn.querySelector(".btn-text");
    const loader = btn.querySelector(".btn-loader");
    if (loading) {
        btn.disabled = true;
        if (text) text.classList.add("hidden");
        if (loader) loader.classList.remove("hidden");
    } else {
        btn.disabled = false;
        if (text) text.classList.remove("hidden");
        if (loader) loader.classList.add("hidden");
    }
}

function autoResize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Simple markdown → HTML renderer.
 * Handles: code blocks, inline code, bold, italic, lists, headings, paragraphs.
 *
 * ⚠️ SECURITY — THE ORDERING BELOW IS LOAD-BEARING. escapeHtml() MUST run
 * first, before any regex reintroduces tags. Because the input is escaped up
 * front, a <script> in LLM output is already &lt;script&gt; by the time the
 * replacements run, so the only tags in the result are the ones we emit.
 * Moving escapeHtml() later — or dropping it — turns model output (which is
 * derived from arbitrary ingested documents) into stored XSS.
 *
 * If this grows any further, replace it with marked + DOMPurify rather than
 * extending the regex chain.
 */
function renderMarkdown(text) {
    if (!text) return "";

    let html = escapeHtml(text); // ← must stay first; see note above

    // Code blocks (```...```). The newline after the opening fence is
    // optional — ```js foo()``` on one line is valid too.
    html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code (`...`)
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold (**...**)
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

    // Italic (*...*)
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

    // Headings
    html = html.replace(/^### (.+)$/gm, "<h4>$1</h4>");
    html = html.replace(/^## (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^# (.+)$/gm, "<h2>$1</h2>");

    // Lists — tag each line as ordered/unordered <li> first, THEN wrap
    // consecutive runs of the same type, so numbered lists don't render as bare <li>s.
    html = html.replace(/^[*-] (.+)$/gm, '<li data-list-type="ul">$1</li>');
    html = html.replace(/^\d+\. (.+)$/gm, '<li data-list-type="ol">$1</li>');
    html = html.replace(
        /<li data-list-type="(ul|ol)">.*<\/li>(?:\n?<li data-list-type="\1">.*<\/li>)*\n?/g,
        (match, type) => {
            const items = match.replace(/ data-list-type="(?:ul|ol)"/g, "");
            return `<${type}>${items}</${type}>`;
        }
    );

    // Line breaks → paragraphs
    html = html
        .split("\n\n")
        .map((block) => {
            block = block.trim();
            if (!block) return "";
            if (
                block.startsWith("<pre>") ||
                block.startsWith("<h") ||
                block.startsWith("<ul>") ||
                block.startsWith("<ol>") ||
                block.startsWith("<li>")
            ) {
                return block;
            }
            return `<p>${block.replace(/\n/g, "<br>")}</p>`;
        })
        .join("\n");

    return html;
}
