(function () {
  "use strict";

  const STORAGE_KEYS = {
    prompt: "copilot.systemPrompt",
    settings: "copilot.chatSettings",
  };

  const elements = {
    chatWindow: document.getElementById("chat-window"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    promptTextarea: document.getElementById("system-prompt"),
    savePrompt: document.getElementById("save-prompt"),
    resetPrompt: document.getElementById("reset-prompt"),
    settingsModal: document.getElementById("settings-modal"),
    openSettings: document.getElementById("open-settings"),
    closeSettings: document.getElementById("close-settings"),
    cancelSettings: document.getElementById("cancel-settings"),
    settingsForm: document.getElementById("settings-form"),
    settingsApiKey: document.getElementById("settings-api-key"),
    settingsModel: document.getElementById("settings-model"),
    settingsTemperature: document.getElementById("settings-temperature"),
    settingsTopP: document.getElementById("settings-top-p"),
    settingsApiBase: document.getElementById("settings-api-base"),
    toast: document.getElementById("toast"),
    refreshCheckAll: document.getElementById("refresh-check-all"),
    createTestEnv: document.getElementById("create-test-env"),
    checkAllOutput: document.getElementById("check-all-output"),
    testStart: document.getElementById("test-start"),
    testEnd: document.getElementById("test-end"),
    requestTrade: document.getElementById("request-trade"),
    tradeOutput: document.getElementById("trade-output"),
  };

  const state = {
    defaultPrompt: "",
    systemPrompt: "",
    settings: {
      model: "",
      temperature: 0.2,
      top_p: null,
      api_base: null,
      api_key: "",
    },
    chat: [],
    loadingChat: false,
    loadingCheckAll: false,
    loadingTrade: false,
  };

  function showToast(message, variant = "info") {
    if (!elements.toast) return;
    elements.toast.textContent = message;
    elements.toast.dataset.variant = variant;
    elements.toast.classList.add("visible");
    window.setTimeout(() => {
      elements.toast.classList.remove("visible");
    }, 2800);
  }

  async function fetchJSON(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail = "";
      try {
        const payload = await response.json();
        detail = payload?.detail || payload?.message || JSON.stringify(payload);
      } catch (error) {
        detail = await response.text();
      }
      throw new Error(detail || `Запрос завершился с ошибкой ${response.status}`);
    }
    if (response.status === 204) return null;
    return response.json();
  }

  function loadSettings() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEYS.settings);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object") {
        state.settings.model = typeof parsed.model === "string" ? parsed.model : state.settings.model;
        state.settings.temperature = Number.isFinite(parsed.temperature)
          ? Number(parsed.temperature)
          : state.settings.temperature;
        state.settings.top_p = Number.isFinite(parsed.top_p) ? Number(parsed.top_p) : null;
        state.settings.api_base = typeof parsed.api_base === "string" && parsed.api_base ? parsed.api_base : null;
        state.settings.api_key = typeof parsed.api_key === "string" ? parsed.api_key : "";
      }
    } catch (error) {
      console.warn("Не удалось прочитать сохранённые настройки", error);
    }
  }

  function persistSettings() {
    const payload = {
      model: state.settings.model,
      temperature: state.settings.temperature,
      top_p: state.settings.top_p,
      api_base: state.settings.api_base,
      api_key: state.settings.api_key,
    };
    try {
      window.localStorage.setItem(STORAGE_KEYS.settings, JSON.stringify(payload));
    } catch (error) {
      console.warn("Не удалось сохранить настройки", error);
    }
  }

  function loadPrompt() {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEYS.prompt);
      if (typeof saved === "string" && saved.trim()) {
        state.systemPrompt = saved;
      }
    } catch (error) {
      console.warn("Не удалось прочитать промпт", error);
    }
  }

  function persistPrompt() {
    try {
      window.localStorage.setItem(STORAGE_KEYS.prompt, state.systemPrompt);
    } catch (error) {
      console.warn("Не удалось сохранить промпт", error);
    }
  }

  function renderPrompt() {
    if (elements.promptTextarea) {
      elements.promptTextarea.value = state.systemPrompt;
    }
  }

  function renderChat() {
    if (!elements.chatWindow) return;
    elements.chatWindow.innerHTML = "";
    if (!state.chat.length) {
      const empty = document.createElement("p");
      empty.className = "chat-placeholder";
      empty.textContent = "Сообщений пока нет";
      elements.chatWindow.append(empty);
      return;
    }
    for (const message of state.chat) {
      const node = document.createElement("div");
      node.className = `chat-message ${message.role}`;
      const role = document.createElement("div");
      role.className = "role";
      role.textContent = message.role === "user" ? "Вы" : "Модель";
      node.append(role);

      const content = document.createElement("div");
      if (message.isJson) {
        const pre = document.createElement("pre");
        pre.textContent = message.content;
        content.append(pre);
      } else {
        const paragraph = document.createElement("p");
        paragraph.textContent = message.content;
        content.append(paragraph);
      }
      node.append(content);
      if (message.pending) {
        node.classList.add("pending");
      }
      elements.chatWindow.append(node);
    }
    elements.chatWindow.scrollTop = elements.chatWindow.scrollHeight;
  }

  function renderSettingsForm() {
    if (!elements.settingsForm) return;
    elements.settingsApiKey.value = state.settings.api_key;
    elements.settingsModel.value = state.settings.model;
    elements.settingsTemperature.value = state.settings.temperature;
    elements.settingsTopP.value = state.settings.top_p ?? "";
    elements.settingsApiBase.value = state.settings.api_base ?? "";
  }

  function formatJson(input) {
    if (!input) return "";
    try {
      const parsed = typeof input === "string" ? JSON.parse(input) : input;
      return JSON.stringify(parsed, null, 2);
    } catch (error) {
      return typeof input === "string" ? input : JSON.stringify(input);
    }
  }

  function parseDateTimeValue(value) {
    if (!value) return null;
    const dt = new Date(value);
    if (Number.isNaN(dt.getTime())) return null;
    return dt.toISOString();
  }

  function setLoading(target, loading) {
    const button = elements[target];
    if (!button) return;
    button.disabled = loading;
    button.classList.toggle("is-loading", loading);
  }

  async function initialiseDefaults() {
    try {
      const defaults = await fetchJSON("/api/chat/defaults");
      if (defaults) {
        state.defaultPrompt = defaults.system_prompt || "";
        if (!state.systemPrompt) {
          state.systemPrompt = state.defaultPrompt;
        }
        if (defaults.settings) {
          state.settings.model = defaults.settings.model || state.settings.model;
          if (typeof defaults.settings.temperature === "number") {
            state.settings.temperature = defaults.settings.temperature;
          }
          if (typeof defaults.settings.top_p === "number") {
            state.settings.top_p = defaults.settings.top_p;
          }
        }
      }
    } catch (error) {
      console.warn("Не удалось загрузить настройки по умолчанию", error);
    }
  }

  function addMessage(role, content, { pending = false } = {}) {
    const isJson = (() => {
      if (typeof content !== "string") return false;
      const trimmed = content.trim();
      return trimmed.startsWith("{") || trimmed.startsWith("[");
    })();
    state.chat.push({ role, content, pending, isJson });
    renderChat();
  }

  function updatePendingMessage(content) {
    for (let idx = state.chat.length - 1; idx >= 0; idx -= 1) {
      const pending = state.chat[idx];
      if (pending && pending.pending) {
        pending.pending = false;
        pending.content = content;
        const trimmed = String(content || "").trim();
        pending.isJson = trimmed.startsWith("{") || trimmed.startsWith("[");
        break;
      }
    }
    renderChat();
  }

  async function sendChatMessage(text) {
    if (state.loadingChat) return;
    const trimmed = text.trim();
    if (!trimmed) return;
    if (!state.settings.api_key) {
      showToast("Укажите OpenAI API key в настройках", "warning");
      return;
    }
    state.loadingChat = true;
    addMessage("user", trimmed);
    addMessage("assistant", "Ожидание ответа...", { pending: true });
    renderChat();

    const payload = {
      system_prompt: state.systemPrompt,
      messages: state.chat.filter((item) => !item.pending).map((item) => ({
        role: item.role,
        content: item.content,
      })),
      settings: {
        model: state.settings.model,
        temperature: state.settings.temperature,
        top_p: state.settings.top_p,
        api_base: state.settings.api_base,
      },
      api_key: state.settings.api_key,
    };

    try {
      const data = await fetchJSON("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      updatePendingMessage(data?.reply || "Ответ без содержимого");
    } catch (error) {
      updatePendingMessage(`Ошибка: ${error.message || error}`);
    } finally {
      state.loadingChat = false;
    }
  }

  async function refreshCheckAll() {
    if (state.loadingCheckAll) return;
    state.loadingCheckAll = true;
    setLoading("refreshCheckAll", true);
    try {
      const data = await fetchJSON("/api/check-all");
      const formatted = formatJson(data?.check_all);
      elements.checkAllOutput.textContent = formatted || "Нет данных";
    } catch (error) {
      elements.checkAllOutput.textContent = `Ошибка: ${error.message || error}`;
    } finally {
      state.loadingCheckAll = false;
      setLoading("refreshCheckAll", false);
    }
  }

  async function createTestEnvironment() {
    if (state.loadingCheckAll) return;
    const payload = {
      start: parseDateTimeValue(elements.testStart?.value),
      end: parseDateTimeValue(elements.testEnd?.value),
    };
    state.loadingCheckAll = true;
    setLoading("createTestEnv", true);
    try {
      const data = await fetchJSON("/api/test-environment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const formatted = formatJson(data?.check_all);
      elements.checkAllOutput.textContent = formatted || "Нет данных";
      if (data?.rendered_html) {
        const popup = window.open("", "_blank", "noopener");
        if (popup) {
          popup.document.write(data.rendered_html);
          popup.document.close();
        } else {
          showToast("Браузер заблокировал новое окно", "warning");
        }
      }
    } catch (error) {
      elements.checkAllOutput.textContent = `Ошибка: ${error.message || error}`;
    } finally {
      state.loadingCheckAll = false;
      setLoading("createTestEnv", false);
    }
  }

  async function requestTrade() {
    if (state.loadingTrade) return;
    if (!state.settings.api_key) {
      showToast("Укажите OpenAI API key в настройках", "warning");
      return;
    }
    state.loadingTrade = true;
    setLoading("requestTrade", true);
    try {
      const payload = {
        api_key: state.settings.api_key,
        model: state.settings.model,
        system_prompt: state.systemPrompt,
        api_base: state.settings.api_base,
      };
      const data = await fetchJSON("/api/trade-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      elements.tradeOutput.textContent = formatJson(data);
      if (data?.trade_json) {
        addMessage("assistant", JSON.stringify(data.trade_json, null, 2));
      }
    } catch (error) {
      elements.tradeOutput.textContent = `Ошибка: ${error.message || error}`;
    } finally {
      state.loadingTrade = false;
      setLoading("requestTrade", false);
    }
  }

  function openSettingsModal() {
    elements.settingsModal?.classList.remove("hidden");
    renderSettingsForm();
  }

  function closeSettingsModal() {
    elements.settingsModal?.classList.add("hidden");
  }

  async function init() {
    await initialiseDefaults();
    loadSettings();
    loadPrompt();
    if (!state.systemPrompt) {
      state.systemPrompt = state.defaultPrompt;
    }
    renderPrompt();
    renderSettingsForm();
    renderChat();
    refreshCheckAll();
  }

  if (elements.savePrompt) {
    elements.savePrompt.addEventListener("click", () => {
      state.systemPrompt = elements.promptTextarea?.value || "";
      persistPrompt();
      showToast("Промпт сохранён");
    });
  }

  if (elements.resetPrompt) {
    elements.resetPrompt.addEventListener("click", () => {
      state.systemPrompt = state.defaultPrompt;
      renderPrompt();
      persistPrompt();
      showToast("Промпт сброшен к значениям по умолчанию");
    });
  }

  if (elements.chatForm && elements.chatInput) {
    elements.chatForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const value = elements.chatInput.value;
      elements.chatInput.value = "";
      sendChatMessage(value);
    });
  }

  if (elements.openSettings) {
    elements.openSettings.addEventListener("click", () => {
      renderSettingsForm();
      openSettingsModal();
    });
  }

  if (elements.closeSettings) {
    elements.closeSettings.addEventListener("click", closeSettingsModal);
  }

  if (elements.cancelSettings) {
    elements.cancelSettings.addEventListener("click", closeSettingsModal);
  }

  if (elements.settingsForm) {
    elements.settingsForm.addEventListener("submit", (event) => {
      event.preventDefault();
      state.settings.api_key = elements.settingsApiKey.value.trim();
      state.settings.model = elements.settingsModel.value.trim() || state.settings.model;
      state.settings.temperature = Number(elements.settingsTemperature.value) || state.settings.temperature;
      const topPValue = elements.settingsTopP.value;
      state.settings.top_p = topPValue === "" ? null : Number(topPValue);
      const apiBaseValue = elements.settingsApiBase.value.trim();
      state.settings.api_base = apiBaseValue ? apiBaseValue : null;
      persistSettings();
      closeSettingsModal();
      showToast("Настройки обновлены");
    });
  }

  if (elements.refreshCheckAll) {
    elements.refreshCheckAll.addEventListener("click", refreshCheckAll);
  }

  if (elements.createTestEnv) {
    elements.createTestEnv.addEventListener("click", createTestEnvironment);
  }

  if (elements.requestTrade) {
    elements.requestTrade.addEventListener("click", requestTrade);
  }

  window.addEventListener("DOMContentLoaded", init);
})();
