/**
 * DevDocs AI — Frontend Application
 *
 * Handles auth flow (JWT), RBAC-aware UI, streaming chat,
 * admin panels (ingest, metrics, users).
 */

// ── State ───────────────────────────────────────────────────
const API = window.location.origin;
let token = localStorage.getItem("dd_token");
let currentUser = null;
let isStreaming = false;

// ── DOM refs ────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const authScreen = $("#auth-screen");
const appScreen = $("#app-screen");
const authForm = $("#auth-form");
const authError = $("#auth-error");
const authTabs = $$(".auth-tab");
const authSubmit = $("#auth-submit");
const messagesEl = $("#messages");
const questionInput = $("#question-input");
const sendBtn = $("#send-btn");
const sidebar = $("#sidebar");
const menuToggle = $("#menu-toggle");
const sidebarBackdrop = $("#sidebar-backdrop");

let authMode = "login"; // or "register"

// The welcome-message markup, captured once at load so logout() can restore
// #messages to its pristine state instead of leaking the previous user's
// conversation into the next login on the same tab.
const WELCOME_HTML = messagesEl.innerHTML;

// ── Init ────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
    if (token) {
        try {
            await fetchCurrentUser();
            showApp();
        } catch {
            logout();
        }
    }
    setupEventListeners();
});

// ── Event Listeners ─────────────────────────────────────────
function setupEventListeners() {
    // Auth tabs
    authTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            authMode = tab.dataset.tab;
            authTabs.forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            authSubmit.querySelector(".btn-text").textContent =
                authMode === "login" ? "Sign In" : "Create Account";
            // Chrome/Firefox use the autocomplete token to decide whether to
            // offer to *save* a new password vs *fill* an existing one.
            $("#password").autocomplete =
                authMode === "login" ? "current-password" : "new-password";
            authError.classList.add("hidden");
        });
    });

    // Auth form
    authForm.addEventListener("submit", handleAuth);

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

    // Logout
    $("#logout-btn").addEventListener("click", () => logout(true));

    // Ingest form
    $("#ingest-form").addEventListener("submit", handleIngest);

    // Refresh metrics
    $("#refresh-metrics").addEventListener("click", loadMetrics);

    // Mobile sidebar (hamburger + backdrop + Escape)
    menuToggle.addEventListener("click", () => {
        if (sidebar.classList.contains("open")) closeSidebar();
        else openSidebar();
    });
    sidebarBackdrop.addEventListener("click", closeSidebar);
    document.addEventListener("keydown", (e) => {
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

// ── Auth ────────────────────────────────────────────────────
async function handleAuth(e) {
    e.preventDefault();
    const username = $("#username").value.trim();
    const password = $("#password").value;

    setLoading(authSubmit, true);
    authError.classList.add("hidden");

    try {
        const endpoint = authMode === "login" ? "/auth/login" : "/auth/register";
        const res = await fetch(`${API}${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || "Authentication failed");
        }

        const data = await res.json();
        token = data.token;
        localStorage.setItem("dd_token", token);
        currentUser = { username: data.username, role: data.role };
        showApp();
    } catch (err) {
        authError.textContent = err.message;
        authError.classList.remove("hidden");
    } finally {
        setLoading(authSubmit, false);
    }
}

async function fetchCurrentUser() {
    const res = await fetch(`${API}/auth/me`, {
        headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) throw new Error("Invalid session");
    currentUser = await res.json();
}

/**
 * Clear local session state. `revoke` sends the token to /auth/logout so the
 * server denylists its jti — a stateless JWT is otherwise valid until it
 * expires, which would make "log out" purely cosmetic.
 */
async function logout(revoke = false) {
    if (revoke && token) {
        try {
            await fetch(`${API}/auth/logout`, {
                method: "POST",
                headers: { Authorization: `Bearer ${token}` },
            });
        } catch {
            // Network failure must not trap the user in a logged-in UI.
        }
    }
    token = null;
    currentUser = null;
    isStreaming = false; // abandon any in-flight assistant response
    localStorage.removeItem("dd_token");
    authScreen.classList.add("active");
    appScreen.classList.remove("active");
    authForm.reset();
    authError.classList.add("hidden");

    // Reset app-screen state so the next user to sign in on this tab never
    // sees the previous user's conversation or ingest status.
    messagesEl.innerHTML = WELCOME_HTML;
    const ingestStatus = $("#ingest-status");
    ingestStatus.textContent = "";
    ingestStatus.className = "status-message hidden";
    questionInput.value = "";
    sendBtn.disabled = true;
    autoResize(questionInput);
    closeSidebar();
    switchView("chat");
}

// ── App Shell ───────────────────────────────────────────────
function showApp() {
    authScreen.classList.remove("active");
    appScreen.classList.add("active");

    // Update user info
    $("#user-name").textContent = currentUser.username;
    $("#user-role").textContent = currentUser.role;
    $("#user-avatar").textContent = currentUser.username[0].toUpperCase();

    // Show/hide admin nav items
    const isAdmin = currentUser.role === "admin";
    $$(".admin-only").forEach((el) => {
        el.style.display = isAdmin ? "flex" : "none";
    });

    // Default to chat view
    switchView("chat");
}

function switchView(viewName) {
    $$(".view").forEach((v) => v.classList.remove("active"));
    $$(".nav-item").forEach((n) => n.classList.remove("active"));

    const view = $(`#view-${viewName}`);
    const nav = $(`[data-view="${viewName}"]`);
    if (view) view.classList.add("active");
    if (nav) nav.classList.add("active");

    // Load data for admin views
    if (viewName === "metrics") loadMetrics();
    if (viewName === "users") loadUsers();
}

// ── Chat ────────────────────────────────────────────────────
async function sendMessage() {
    const question = questionInput.value.trim();
    if (!question || isStreaming) return;

    // Hide welcome
    const welcome = $(".welcome-message");
    if (welcome) welcome.remove();

    // Add user message
    appendMessage("user", question, currentUser.username);

    // Clear input
    questionInput.value = "";
    sendBtn.disabled = true;
    autoResize(questionInput);

    // Add assistant message with typing indicator
    const msgEl = appendMessage("assistant", "", "DevDocs AI");
    const contentEl = msgEl.querySelector(".message-content");
    contentEl.innerHTML = `<div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    </div>`;

    isStreaming = true;

    // Rendering is throttled to one animation frame. Re-parsing the whole
    // markdown answer and rewriting innerHTML on *every* network chunk is
    // O(n^2) DOM work that visibly janks on long answers.
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
        const res = await fetch(`${API}/ask`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "text/event-stream",
                Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({ question, k: 5 }),
        });

        if (res.status === 401) {
            logout();
            return;
        }

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: "Request failed" }));
            throw new Error(err.detail || `Error ${res.status}`);
        }

        // ── Server-Sent Events ──────────────────────────────
        // Replaces the old "\n\n||SOURCES||" sentinel, which was ambiguous (an
        // LLM emitting that literal string corrupted the split) and untyped.
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

                if (evt.event === "token") {
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
                    streamError = evt.data.message || "Stream failed";
                }
            }
        }

        paint(); // final synchronous render — never leave a queued frame unpainted

        if (streamError) {
            contentEl.innerHTML +=
                `<div class="status-message error">⚠️ ${escapeHtml(streamError)}</div>`;
            scrollToBottom();
        }
    } catch (err) {
        contentEl.innerHTML = `<span style="color: var(--error)">⚠️ ${escapeHtml(err.message)}</span>`;
    } finally {
        isStreaming = false;
        sendBtn.disabled = !questionInput.value.trim();
        scrollToBottom();
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
        </div>
    `;

    messagesEl.appendChild(el);
    scrollToBottom();
    return el;
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Admin: Ingest ───────────────────────────────────────────
// /ingest now returns 202 + a job_id immediately, because cloning + chunking +
// embedding a real repository takes minutes and load balancers cut idle
// connections long before that. We poll for the outcome instead.
async function handleIngest(e) {
    e.preventDefault();
    const source = $("#ingest-source").value.trim();
    const statusEl = $("#ingest-status");
    const btn = e.target.querySelector(".btn-primary");

    setLoading(btn, true);
    statusEl.classList.add("hidden");

    const show = (text, kind) => {
        statusEl.textContent = text;
        statusEl.className = `status-message ${kind}`;
        statusEl.classList.remove("hidden");
    };

    try {
        const res = await fetch(`${API}/ingest`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({ source }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || "Ingestion failed");
        }

        const { job_id } = await res.json();
        show("⏳ Queued — cloning and embedding. This can take a few minutes...", "info");
        $("#ingest-source").value = "";

        const job = await pollIngestJob(job_id, (state) =>
            show(`⏳ ${state}... (job ${job_id})`, "info")
        );

        if (job.status === "succeeded") {
            show(
                `✅ Ingested ${job.chunks_added} chunks — ${job.total_chunks} total in DB`,
                "success"
            );
        } else {
            show(`❌ Ingest failed: ${job.error || "unknown error"}`, "error");
        }
    } catch (err) {
        show(`❌ ${err.message}`, "error");
    } finally {
        setLoading(btn, false);
    }
}

async function pollIngestJob(jobId, onProgress, { intervalMs = 2000, timeoutMs = 900000 } = {}) {
    const deadline = Date.now() + timeoutMs;
    let last = "";
    while (Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, intervalMs));
        const res = await fetch(`${API}/ingest/${jobId}`, {
            headers: { Authorization: `Bearer ${token}` },
        });
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

// ── Admin: Metrics ──────────────────────────────────────────
async function loadMetrics() {
    try {
        const res = await fetch(`${API}/metrics`, {
            headers: { Authorization: `Bearer ${token}` },
        });
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

// ── Admin: Users ────────────────────────────────────────────
async function loadUsers() {
    const tbody = $("#users-tbody");
    try {
        const res = await fetch(`${API}/admin/users`, {
            headers: { Authorization: `Bearer ${token}` },
        });
        if (!res.ok) throw new Error("Failed to load users");
        const data = await res.json();

        if (!data.users || data.users.length === 0) {
            tbody.innerHTML = `<tr><td colspan="4" class="empty-state">No users found</td></tr>`;
            return;
        }

        tbody.innerHTML = data.users
            .map(
                (u) => `<tr>
                <td>${u.id}</td>
                <td>${escapeHtml(u.username)}</td>
                <td><span class="role-badge ${u.role}">${u.role}</span></td>
                <td>${u.created || "—"}</td>
            </tr>`
            )
            .join("");
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="4" class="empty-state">Failed to load users</td></tr>`;
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
    // consecutive runs of the *same* type in one pass. Numbered lines must be
    // converted to <li> before the wrap regex runs, otherwise (as before) the
    // <ul> wrap already happened and numbered lists render as bare <li>s.
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
