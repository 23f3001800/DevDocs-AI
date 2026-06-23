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

let authMode = "login"; // or "register"

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

    // Quick action buttons
    document.addEventListener("click", (e) => {
        if (e.target.classList.contains("quick-btn")) {
            questionInput.value = e.target.dataset.question;
            sendBtn.disabled = false;
            sendMessage();
        }
    });

    // Nav items
    $$(".nav-item").forEach((item) => {
        item.addEventListener("click", () => switchView(item.dataset.view));
    });

    // Logout
    $("#logout-btn").addEventListener("click", logout);

    // Ingest form
    $("#ingest-form").addEventListener("submit", handleIngest);

    // Refresh metrics
    $("#refresh-metrics").addEventListener("click", loadMetrics);
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

function logout() {
    token = null;
    currentUser = null;
    localStorage.removeItem("dd_token");
    authScreen.classList.add("active");
    appScreen.classList.remove("active");
    authForm.reset();
    authError.classList.add("hidden");
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

    try {
        const res = await fetch(`${API}/ask`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
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

        // Stream the response
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullText = "";
        contentEl.innerHTML = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            fullText += chunk;

            // Check for sources delimiter
            const sourceSplit = fullText.split("\n\n||SOURCES||");
            const answerText = sourceSplit[0];

            contentEl.innerHTML = renderMarkdown(answerText);
            scrollToBottom();
        }

        // Parse sources if present
        const sourceSplit = fullText.split("\n\n||SOURCES||");
        if (sourceSplit.length > 1) {
            try {
                const sources = JSON.parse(sourceSplit[1]);
                if (sources.length > 0) {
                    const sourcesHtml = `<div class="sources-panel">
                        <div class="sources-label">📄 Sources</div>
                        ${sources.map((s) => `<div class="source-item">${escapeHtml(s)}</div>`).join("")}
                    </div>`;
                    contentEl.innerHTML = renderMarkdown(sourceSplit[0]) + sourcesHtml;
                }
            } catch {
                // Ignore JSON parse errors for sources
            }
        }
    } catch (err) {
        contentEl.innerHTML = `<span style="color: var(--error)">⚠️ ${escapeHtml(err.message)}</span>`;
    } finally {
        isStreaming = false;
        sendBtn.disabled = !questionInput.value.trim();
        scrollToBottom();
    }
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
async function handleIngest(e) {
    e.preventDefault();
    const source = $("#ingest-source").value.trim();
    const statusEl = $("#ingest-status");
    const btn = e.target.querySelector(".btn-primary");

    setLoading(btn, true);
    statusEl.classList.add("hidden");

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

        const data = await res.json();
        statusEl.textContent = `✅ Ingested successfully — ${data.total_chunks} total chunks in DB`;
        statusEl.className = "status-message success";
        statusEl.classList.remove("hidden");
        $("#ingest-source").value = "";
    } catch (err) {
        statusEl.textContent = `❌ ${err.message}`;
        statusEl.className = "status-message error";
        statusEl.classList.remove("hidden");
    } finally {
        setLoading(btn, false);
    }
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
 * Simple markdown → HTML renderer
 * Handles: code blocks, inline code, bold, italic, lists, headings, paragraphs
 */
function renderMarkdown(text) {
    if (!text) return "";

    let html = escapeHtml(text);

    // Code blocks (```...```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
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

    // Unordered lists
    html = html.replace(/^[*-] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>");

    // Numbered lists
    html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

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
