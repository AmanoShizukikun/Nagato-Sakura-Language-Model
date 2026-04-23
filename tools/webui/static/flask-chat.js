const bootstrap = JSON.parse(document.getElementById("bootstrap-data").textContent || "{}");
const STORAGE_KEY = "ns_flask_webui_conversations_v1";
const SIDEBAR_STATE_KEY = "ns_flask_webui_sidebar_collapsed_v1";
const SETTINGS_STORAGE_KEY = "ns_flask_webui_settings_v2";
const SETTINGS_SCHEMA_VERSION = 1;
const DEFAULT_ACTIVE_SETTINGS_SECTION = "audioVideo";
const TIME_PREVIEW_UPDATE_MS = 30000;
const CHAT_UI_MODES = ["bubbleOnly", "avatarBubble", "discord"];
const CHAT_BOTTOM_THRESHOLD_PX = 24;
const SETTINGS_SECTIONS = ["profile", "audioVideo", "theme", "display", "shortcuts", "languageTime", "toolRegistry"];
const DEFAULT_USER_NAME = "主人";
const PROFILE_AVATAR_MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024;
const PROFILE_AVATAR_MAX_DATA_URL_LENGTH = 2800000;

const TXT = {
    defaultChatTitle: "\u65b0\u5c0d\u8a71",
    tempChatTitle: "\u81e8\u6642\u5c0d\u8a71",
    normalChatPrefix: "\u4e00\u822c\u5c0d\u8a71",
    tempChatPrefix: "\u81e8\u6642\u5c0d\u8a71",
    emptyResponse: "\uff08\u7a7a\u56de\u61c9\uff09",
    noSearchResults: "\u627e\u4e0d\u5230\u7b26\u5408\u689d\u4ef6\u7684\u5c0d\u8a71",
    noHistoryYet: "\u5c1a\u7121\u904e\u53bb\u804a\u5929\u7d00\u9304",
    noMessageYet: "\u5c1a\u7121\u8a0a\u606f",
    toastNewChat: "\u5df2\u5efa\u7acb\u65b0\u5c0d\u8a71",
    toastTempChat: "\u5df2\u958b\u555f\u81e8\u6642\u5c0d\u8a71",
    toastClearCurrent: "\u76ee\u524d\u5c0d\u8a71\u5df2\u6e05\u7a7a",
    toastDeleteConversation: "\u5df2\u522a\u9664\u5c0d\u8a71\u7d00\u9304",
    toastCopyOk: "\u5df2\u8907\u88fd Streamlit \u555f\u52d5\u6307\u4ee4",
    toastCopyFail: "\u8907\u88fd\u5931\u6557\uff0c\u8acb\u624b\u52d5\u67e5\u770b\u6307\u4ee4",
    toastCopyMessageOk: "\u5df2\u8907\u88fd\u8a0a\u606f\u5167\u5bb9",
    toastCopyMessageFail: "\u8907\u88fd\u8a0a\u606f\u5931\u6557",
    toastEditMessage: "\u5df2\u5c07\u5167\u5bb9\u5e36\u5165\u8f38\u5165\u6846\u53ef\u7e7c\u7e8c\u7de8\u8f2f",
    toastEditMode: "\u7de8\u8f2f\u6a21\u5f0f\uff1a\u9001\u51fa\u5f8c\u5c07\u91cd\u5efa\u8a72\u53e5\u4e4b\u5f8c\u5c0d\u8a71",
    toastRedoStart: "\u6b63\u5728\u4f9d\u64da\u8a72\u5c0d\u8a71\u91cd\u65b0\u751f\u6210\u56de\u61c9",
    toastRedoMissingUser: "\u627e\u4e0d\u5230\u5c0d\u61c9\u7684\u7528\u6236\u8a0a\u606f\uff0c\u7121\u6cd5\u91cd\u4f5c",
    toastSettingsSaved: "\u8a2d\u5b9a\u5df2\u5132\u5b58",
    toastSettingsReset: "\u5df2\u9084\u539f\u6210\u9810\u8a2d\u8a2d\u5b9a",
    toastProfileAvatarUpdated: "\u982d\u50cf\u5df2\u66f4\u65b0\uff0c\u8a18\u5f97\u5132\u5b58\u8a2d\u5b9a",
    toastProfileAvatarCleared: "\u982d\u50cf\u5df2\u6e05\u9664\uff0c\u8a18\u5f97\u5132\u5b58\u8a2d\u5b9a",
    toastProfileAvatarTypeError: "\u8acb\u9078\u64c7\u5716\u7247\u6a94\u6848",
    toastProfileAvatarSizeError: "\u5716\u7247\u8acb\u5c0f\u65bc 2MB",
    toastProfileAvatarReadFail: "\u982d\u50cf\u8b80\u53d6\u5931\u6557",
    chatErrorPrefix: "\u751f\u6210\u5931\u6557: ",
    statusError: "\u56de\u61c9\u89e3\u6790\u5931\u6557",
    emptyChatHint: "\u4e3b\u4eba\uff0c\u60a8\u4f86\u4e86\u2026\u2026 \u8acb\u554f\u4eca\u5929\u6709\u4ec0\u9ebc\u9577\u9580\u6afb\u53ef\u4ee5\u70ba\u60a8\u6548\u52de\u7684\u5462\uff1f\ud83c\udf38",
    deleteConversationAria: "\u522a\u9664\u5c0d\u8a71\u7d00\u9304",
};

const state = {
    conversations: [],
    activeConversationId: null,
    isSending: false,
    sidebarCollapsed: false,
    editingConversationId: null,
    editingMessageIndex: null,
    settings: null,
    healthCheckTimerId: null,
    timePreviewTimerId: null,
    media: {
        availableDevices: {
            audioInput: [],
            audioOutput: [],
            videoInput: [],
        },
        micMonitor: {
            stream: null,
            audioContext: null,
            sourceNode: null,
            analyserNode: null,
            frameId: null,
        },
        cameraStream: null,
        testRecorder: {
            mediaRecorder: null,
            stream: null,
            chunks: [],
            timerId: null,
            objectUrl: "",
            isRecording: false,
        },
    },
};

const STATUS_META = {
    online: { label: "線上", className: "online", content: "" },
    idle: { label: "閒置", className: "idle", content: "" },
    error: { label: "錯誤", className: "error", content: "" },
    offline: { label: "離線", className: "offline", content: "" },
    generating: { label: "生成中", className: "generating", content: "<span></span><span></span><span></span>" },
};

let statusIdleTimer = null;
let currentStatusKey = "offline";
let toastHideTimer = null;

function clearStatusIdleTimer() {
    if (statusIdleTimer) {
        window.clearTimeout(statusIdleTimer);
        statusIdleTimer = null;
    }
}

function getStatusIdleDelayMs() {
    const uiState = state.settings ? state.settings.uiState : getDefaultSettings().uiState;
    const idleSeconds = clampNumber(uiState.statusIdleSeconds, 60, { min: 10, max: 3600, integer: true });
    return idleSeconds * 1000;
}

function scheduleIdleStatusTransition(delayMs = getStatusIdleDelayMs()) {
    clearStatusIdleTimer();
    const safeDelay = clampNumber(delayMs, getStatusIdleDelayMs(), { min: 0, max: 3600000, integer: true });
    statusIdleTimer = window.setTimeout(() => {
        statusIdleTimer = null;
        if (!state.isSending && currentStatusKey === "online") {
            setStatus("idle");
        }
    }, safeDelay);
}

const refs = {
    appShell: document.querySelector(".app-shell"),
    leftSidebar: document.getElementById("leftSidebar"),
    rightSidebar: document.getElementById("rightSidebar"),
    settingsBackdrop: document.getElementById("settingsBackdrop"),
    closeSidebarBtn: document.getElementById("closeSidebarBtn"),
    roleInfoBtn: document.getElementById("roleInfoBtn"),
    roleInfoAvatar: document.querySelector(".role-info-avatar"),
    closeRightSidebarBtn: document.getElementById("closeRightSidebarBtn"),
    viewFullProfileBtn: document.getElementById("viewFullProfileBtn"),
    saveModelSettingsBtn: document.getElementById("saveModelSettingsBtn"),
    resetModelSettingsBtn: document.getElementById("resetModelSettingsBtn"),
    newChatBtn: document.getElementById("newChatBtn"),
    tempChatBtn: document.getElementById("tempChatBtn"),
    openSettingsBtn: document.getElementById("openSettingsBtn"),
    settingsModal: document.getElementById("settingsModal"),
    closeSettingsBtn: document.getElementById("closeSettingsBtn"),
    settingsNav: document.getElementById("settingsNav"),
    saveSettingsBtn: document.getElementById("saveSettingsBtn"),
    resetSettingsBtn: document.getElementById("resetSettingsBtn"),
    searchConversationInput: document.getElementById("searchConversationInput"),
    conversationList: document.getElementById("conversationList"),
    conversationCount: document.getElementById("conversationCount"),
    activeConversationLabel: document.getElementById("activeConversationLabel"),
    brandStatusDot: document.getElementById("brandStatusDot"),
    chatList: document.getElementById("chatList"),
    chatJumpWrap: document.getElementById("chatJumpWrap"),
    jumpToLatestBtn: document.getElementById("jumpToLatestBtn"),
    composerWrap: document.querySelector(".composer-wrap"),
    promptInput: document.getElementById("promptInput"),
    voiceVisualizer: document.getElementById("voiceVisualizer"),
    micBtn: document.getElementById("micBtn"),
    toast: document.getElementById("toast"),
    modelPath: document.getElementById("modelPath"),
    tokenizerPath: document.getElementById("tokenizerPath"),
    historyRounds: document.getElementById("historyRounds"),
    maxLength: document.getElementById("maxLength"),
    maxNewTokens: document.getElementById("maxNewTokens"),
    topK: document.getElementById("topK"),
    temperature: document.getElementById("temperature"),
    topP: document.getElementById("topP"),
    repetitionPenalty: document.getElementById("repetitionPenalty"),
    temperatureLabel: document.getElementById("temperatureLabel"),
    topPLabel: document.getElementById("topPLabel"),
    repetitionPenaltyLabel: document.getElementById("repetitionPenaltyLabel"),
    doSample: document.getElementById("doSample"),
    settingsNavItems: Array.from(document.querySelectorAll(".settings-nav-item")),
    settingsPanels: Array.from(document.querySelectorAll(".settings-tab-panel")),
    avMicDevice: document.getElementById("avMicDevice"),
    avSpeakerDevice: document.getElementById("avSpeakerDevice"),
    avCameraDevice: document.getElementById("avCameraDevice"),
    speakerSupportHint: document.getElementById("speakerSupportHint"),
    refreshDevicesBtn: document.getElementById("refreshDevicesBtn"),
    avEchoCancellation: document.getElementById("avEchoCancellation"),
    avNoiseSuppression: document.getElementById("avNoiseSuppression"),
    avAutoGainControl: document.getElementById("avAutoGainControl"),
    avMicGain: document.getElementById("avMicGain"),
    avMicGainLabel: document.getElementById("avMicGainLabel"),
    startMicMonitorBtn: document.getElementById("startMicMonitorBtn"),
    stopMicMonitorBtn: document.getElementById("stopMicMonitorBtn"),
    recordMicTestBtn: document.getElementById("recordMicTestBtn"),
    micLevelBar: document.getElementById("micLevelBar"),
    micTestPlayback: document.getElementById("micTestPlayback"),
    startCameraTestBtn: document.getElementById("startCameraTestBtn"),
    stopCameraTestBtn: document.getElementById("stopCameraTestBtn"),
    cameraPreview: document.getElementById("cameraPreview"),
    avStatusText: document.getElementById("avStatusText"),
    themePreset: document.getElementById("themePreset"),
    displayFontScale: document.getElementById("displayFontScale"),
    displayFontScaleLabel: document.getElementById("displayFontScaleLabel"),
    displayChatUiMode: document.getElementById("displayChatUiMode"),
    displayDensity: document.getElementById("displayDensity"),
    displayAnimations: document.getElementById("displayAnimations"),
    displayMetaTokensPerSec: document.getElementById("displayMetaTokensPerSec"),
    displayMetaTokens: document.getElementById("displayMetaTokens"),
    displayMetaElapsed: document.getElementById("displayMetaElapsed"),
    displayMetaStopReason: document.getElementById("displayMetaStopReason"),
    profileUserName: document.getElementById("profileUserName"),
    profileAvatarPreview: document.getElementById("profileAvatarPreview"),
    profileAvatarUploadBtn: document.getElementById("profileAvatarUploadBtn"),
    profileAvatarClearBtn: document.getElementById("profileAvatarClearBtn"),
    profileAvatarFile: document.getElementById("profileAvatarFile"),
    shortcutEnterToSend: document.getElementById("shortcutEnterToSend"),
    shortcutEscClosePanels: document.getElementById("shortcutEscClosePanels"),
    shortcutSlashFocusInput: document.getElementById("shortcutSlashFocusInput"),
    languageSelect: document.getElementById("languageSelect"),
    timeFormatSelect: document.getElementById("timeFormatSelect"),
    timezoneSelect: document.getElementById("timezoneSelect"),
    statusIdleSeconds: document.getElementById("statusIdleSeconds"),
    timePreviewText: document.getElementById("timePreviewText"),
    toolVoiceInput: document.getElementById("toolVoiceInput"),
    toolConversationHistory: document.getElementById("toolConversationHistory"),
    toolMessageActions: document.getElementById("toolMessageActions"),
    toolHealthMonitor: document.getElementById("toolHealthMonitor"),
};

function nowTs() {
    return Date.now();
}

function isObjectLike(value) {
    return value !== null && typeof value === "object";
}

function clampNumber(value, fallback, options = {}) {
    const min = Number.isFinite(options.min) ? options.min : null;
    const max = Number.isFinite(options.max) ? options.max : null;
    const integer = options.integer === true;
    const digits = Number.isInteger(options.digits) ? options.digits : null;

    let num = Number(value);
    if (!Number.isFinite(num)) {
        num = Number(fallback);
    }

    if (Number.isFinite(min)) {
        num = Math.max(min, num);
    }
    if (Number.isFinite(max)) {
        num = Math.min(max, num);
    }
    if (integer) {
        num = Math.round(num);
    }
    if (Number.isInteger(digits) && digits >= 0) {
        num = Number(num.toFixed(digits));
    }

    return num;
}

function boolValue(value, fallback) {
    if (typeof value === "boolean") {
        return value;
    }
    return Boolean(fallback);
}

function pickString(value, fallback, allowedValues = null) {
    const text = String(value ?? "").trim();
    if (!text) {
        return fallback;
    }
    if (Array.isArray(allowedValues) && !allowedValues.includes(text)) {
        return fallback;
    }
    return text;
}

function sanitizeUserName(value, fallback = DEFAULT_USER_NAME) {
    const normalized = String(value ?? "").replace(/\s+/g, " ").trim();
    const clipped = normalized.slice(0, 24);
    return clipped || fallback;
}

function normalizeAvatarDataUrl(value, fallback = "") {
    const text = String(value ?? "").trim();
    if (!text) {
        return fallback;
    }
    if (!text.startsWith("data:image/")) {
        return fallback;
    }
    if (text.length > PROFILE_AVATAR_MAX_DATA_URL_LENGTH) {
        return fallback;
    }
    return text;
}

function getDisplayInitial(text, fallback = "主") {
    const chars = Array.from(String(text ?? "").trim());
    return chars[0] || fallback;
}

function getUserProfile() {
    const defaults = getDefaultSettings().profile;
    const profile = state.settings && isObjectLike(state.settings.profile)
        ? state.settings.profile
        : defaults;
    return {
        userName: sanitizeUserName(profile.userName, defaults.userName),
        userAvatarDataUrl: normalizeAvatarDataUrl(profile.userAvatarDataUrl, defaults.userAvatarDataUrl),
    };
}

function renderProfileAvatarPreview(userName, avatarDataUrl) {
    if (!refs.profileAvatarPreview) {
        return;
    }

    refs.profileAvatarPreview.innerHTML = "";
    if (avatarDataUrl) {
        const img = document.createElement("img");
        img.src = avatarDataUrl;
        img.alt = "";
        refs.profileAvatarPreview.appendChild(img);
        return;
    }

    refs.profileAvatarPreview.textContent = getDisplayInitial(userName, "主");
}

function cloneJson(value) {
    return JSON.parse(JSON.stringify(value));
}

function getDefaultSettings() {
    const defaults = bootstrap.defaults || {};
    const uiDefaults = bootstrap.uiDefaults || {};
    const deviceDefaults = bootstrap.deviceDefaults || {};
    const featureDefaults = bootstrap.featureDefaults || {};
    const profileDefaults = bootstrap.profileDefaults || {};
    const capabilities = bootstrap.capabilities || {};
    const canUseVoiceTranscription = capabilities.voiceTranscription !== false;

    return {
        schemaVersion: SETTINGS_SCHEMA_VERSION,
        params: {
            historyRounds: clampNumber(defaults.historyRounds, 3, { min: 0, max: 40, integer: true }),
            maxLength: clampNumber(defaults.maxLength, 4096, { min: 128, max: 8192, integer: true }),
            maxNewTokens: clampNumber(defaults.maxNewTokens, 512, { min: 16, max: 4096, integer: true }),
            topK: clampNumber(defaults.topK, 50, { min: 1, max: 200, integer: true }),
            temperature: clampNumber(defaults.temperature, 0.7, { min: 0.1, max: 1.8, digits: 2 }),
            topP: clampNumber(defaults.topP, 0.9, { min: 0.1, max: 1.0, digits: 2 }),
            repetitionPenalty: clampNumber(defaults.repetitionPenalty, 1.0, { min: 1.0, max: 2.0, digits: 2 }),
            doSample: boolValue(defaults.doSample, true),
        },
        uiState: {
            activeSection: pickString(uiDefaults.activeSection, DEFAULT_ACTIVE_SETTINGS_SECTION, SETTINGS_SECTIONS),
            themePreset: pickString(uiDefaults.themePreset, "cyber", ["cyber", "sunset"]),
            displayFontScale: clampNumber(uiDefaults.displayFontScale, 1, { min: 0.9, max: 1.3, digits: 2 }),
            chatUiMode: pickString(uiDefaults.chatUiMode, "bubbleOnly", CHAT_UI_MODES),
            displayDensity: pickString(uiDefaults.displayDensity, "normal", ["compact", "normal", "relaxed"]),
            displayAnimations: boolValue(uiDefaults.displayAnimations, true),
            metaShowTokensPerSec: boolValue(uiDefaults.metaShowTokensPerSec, true),
            metaShowTokens: boolValue(uiDefaults.metaShowTokens, true),
            metaShowElapsed: boolValue(uiDefaults.metaShowElapsed, true),
            metaShowStopReason: boolValue(uiDefaults.metaShowStopReason, true),
            statusIdleSeconds: clampNumber(uiDefaults.statusIdleSeconds, 60, { min: 10, max: 3600, integer: true }),
            shortcutEnterToSend: boolValue(uiDefaults.shortcutEnterToSend, true),
            shortcutEscClosePanels: boolValue(uiDefaults.shortcutEscClosePanels, true),
            shortcutSlashFocusInput: boolValue(uiDefaults.shortcutSlashFocusInput, true),
            language: pickString(uiDefaults.language, "zh-Hant", ["zh-Hant"]),
            timeFormat: pickString(uiDefaults.timeFormat, "24h", ["12h", "24h"]),
            timezone: pickString(uiDefaults.timezone, "local", ["local", "Asia/Taipei", "UTC"]),
        },
        profile: {
            userName: sanitizeUserName(profileDefaults.userName, DEFAULT_USER_NAME),
            userAvatarDataUrl: normalizeAvatarDataUrl(profileDefaults.userAvatarDataUrl, ""),
        },
        devicePrefs: {
            micDeviceId: "",
            speakerDeviceId: "",
            cameraDeviceId: "",
            echoCancellation: boolValue(deviceDefaults.echoCancellation, true),
            noiseSuppression: boolValue(deviceDefaults.noiseSuppression, true),
            autoGainControl: boolValue(deviceDefaults.autoGainControl, true),
            micGain: clampNumber(deviceDefaults.micGain, 1, { min: 0.5, max: 2.0, digits: 2 }),
        },
        featureToggles: {
            voiceInput: boolValue(featureDefaults.voiceInput, true) && canUseVoiceTranscription,
            conversationHistory: boolValue(featureDefaults.conversationHistory, true),
            messageActions: boolValue(featureDefaults.messageActions, true),
            healthMonitor: boolValue(featureDefaults.healthMonitor, true),
        },
    };
}

function sanitizeSettings(rawSettings) {
    const defaults = getDefaultSettings();
    if (!isObjectLike(rawSettings)) {
        return defaults;
    }

    const normalized = cloneJson(defaults);
    const params = isObjectLike(rawSettings.params) ? rawSettings.params : {};
    const uiState = isObjectLike(rawSettings.uiState) ? rawSettings.uiState : {};
    const profile = isObjectLike(rawSettings.profile) ? rawSettings.profile : {};
    const devicePrefs = isObjectLike(rawSettings.devicePrefs) ? rawSettings.devicePrefs : {};
    const featureToggles = isObjectLike(rawSettings.featureToggles) ? rawSettings.featureToggles : {};

    normalized.params.historyRounds = clampNumber(params.historyRounds ?? params.history_rounds, defaults.params.historyRounds, { min: 0, max: 40, integer: true });
    normalized.params.maxLength = clampNumber(params.maxLength ?? params.max_length, defaults.params.maxLength, { min: 128, max: 8192, integer: true });
    normalized.params.maxNewTokens = clampNumber(params.maxNewTokens ?? params.max_new_tokens, defaults.params.maxNewTokens, { min: 16, max: 4096, integer: true });
    normalized.params.topK = clampNumber(params.topK ?? params.top_k, defaults.params.topK, { min: 1, max: 200, integer: true });
    normalized.params.temperature = clampNumber(params.temperature, defaults.params.temperature, { min: 0.1, max: 1.8, digits: 2 });
    normalized.params.topP = clampNumber(params.topP ?? params.top_p, defaults.params.topP, { min: 0.1, max: 1.0, digits: 2 });
    normalized.params.repetitionPenalty = clampNumber(params.repetitionPenalty ?? params.repetition_penalty, defaults.params.repetitionPenalty, { min: 1.0, max: 2.0, digits: 2 });
    normalized.params.doSample = boolValue(params.doSample ?? params.do_sample, defaults.params.doSample);

    normalized.uiState.activeSection = pickString(uiState.activeSection, defaults.uiState.activeSection, SETTINGS_SECTIONS);
    normalized.uiState.themePreset = pickString(uiState.themePreset, defaults.uiState.themePreset, ["cyber", "sunset"]);
    normalized.uiState.displayFontScale = clampNumber(uiState.displayFontScale, defaults.uiState.displayFontScale, { min: 0.9, max: 1.3, digits: 2 });
    normalized.uiState.chatUiMode = pickString(uiState.chatUiMode, defaults.uiState.chatUiMode, CHAT_UI_MODES);
    normalized.uiState.displayDensity = pickString(uiState.displayDensity, defaults.uiState.displayDensity, ["compact", "normal", "relaxed"]);
    normalized.uiState.displayAnimations = boolValue(uiState.displayAnimations, defaults.uiState.displayAnimations);
    normalized.uiState.metaShowTokensPerSec = boolValue(uiState.metaShowTokensPerSec, defaults.uiState.metaShowTokensPerSec);
    normalized.uiState.metaShowTokens = boolValue(uiState.metaShowTokens, defaults.uiState.metaShowTokens);
    normalized.uiState.metaShowElapsed = boolValue(uiState.metaShowElapsed, defaults.uiState.metaShowElapsed);
    normalized.uiState.metaShowStopReason = boolValue(uiState.metaShowStopReason, defaults.uiState.metaShowStopReason);
    normalized.uiState.statusIdleSeconds = clampNumber(uiState.statusIdleSeconds, defaults.uiState.statusIdleSeconds, { min: 10, max: 3600, integer: true });
    normalized.uiState.shortcutEnterToSend = boolValue(uiState.shortcutEnterToSend, defaults.uiState.shortcutEnterToSend);
    normalized.uiState.shortcutEscClosePanels = boolValue(uiState.shortcutEscClosePanels, defaults.uiState.shortcutEscClosePanels);
    normalized.uiState.shortcutSlashFocusInput = boolValue(uiState.shortcutSlashFocusInput, defaults.uiState.shortcutSlashFocusInput);
    normalized.uiState.language = pickString(uiState.language, defaults.uiState.language, ["zh-Hant"]);
    normalized.uiState.timeFormat = pickString(uiState.timeFormat, defaults.uiState.timeFormat, ["12h", "24h"]);
    normalized.uiState.timezone = pickString(uiState.timezone, defaults.uiState.timezone, ["local", "Asia/Taipei", "UTC"]);

    normalized.profile.userName = sanitizeUserName(profile.userName, defaults.profile.userName);
    normalized.profile.userAvatarDataUrl = normalizeAvatarDataUrl(
        profile.userAvatarDataUrl ?? profile.userAvatar ?? profile.avatarDataUrl,
        defaults.profile.userAvatarDataUrl
    );

    normalized.devicePrefs.micDeviceId = pickString(devicePrefs.micDeviceId, defaults.devicePrefs.micDeviceId);
    normalized.devicePrefs.speakerDeviceId = pickString(devicePrefs.speakerDeviceId, defaults.devicePrefs.speakerDeviceId);
    normalized.devicePrefs.cameraDeviceId = pickString(devicePrefs.cameraDeviceId, defaults.devicePrefs.cameraDeviceId);
    normalized.devicePrefs.echoCancellation = boolValue(devicePrefs.echoCancellation, defaults.devicePrefs.echoCancellation);
    normalized.devicePrefs.noiseSuppression = boolValue(devicePrefs.noiseSuppression, defaults.devicePrefs.noiseSuppression);
    normalized.devicePrefs.autoGainControl = boolValue(devicePrefs.autoGainControl, defaults.devicePrefs.autoGainControl);
    normalized.devicePrefs.micGain = clampNumber(devicePrefs.micGain, defaults.devicePrefs.micGain, { min: 0.5, max: 2.0, digits: 2 });

    normalized.featureToggles.voiceInput = boolValue(featureToggles.voiceInput, defaults.featureToggles.voiceInput);
    normalized.featureToggles.conversationHistory = boolValue(featureToggles.conversationHistory, defaults.featureToggles.conversationHistory);
    normalized.featureToggles.messageActions = boolValue(featureToggles.messageActions, defaults.featureToggles.messageActions);
    normalized.featureToggles.healthMonitor = boolValue(featureToggles.healthMonitor, defaults.featureToggles.healthMonitor);
    normalized.schemaVersion = SETTINGS_SCHEMA_VERSION;

    return normalized;
}

function loadSettingsFromStorage() {
    try {
        const rawText = localStorage.getItem(SETTINGS_STORAGE_KEY);
        if (!rawText) {
            return getDefaultSettings();
        }
        const parsed = JSON.parse(rawText);
        return sanitizeSettings(parsed);
    } catch {
        return getDefaultSettings();
    }
}

function persistSettingsToStorage() {
    if (!state.settings) {
        return;
    }
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(state.settings));
}

function setAudioVideoStatus(message) {
    if (refs.avStatusText) {
        refs.avStatusText.textContent = String(message || "");
    }
}

function updateMicGainLabel() {
    if (refs.avMicGain && refs.avMicGainLabel) {
        refs.avMicGainLabel.textContent = Number(refs.avMicGain.value).toFixed(2);
    }
}

function updateDisplayScaleLabel() {
    if (refs.displayFontScale && refs.displayFontScaleLabel) {
        refs.displayFontScaleLabel.textContent = `${Number(refs.displayFontScale.value).toFixed(2)}x`;
    }
}

function setActiveSettingsSection(sectionKey, persist = true) {
    const target = pickString(sectionKey, DEFAULT_ACTIVE_SETTINGS_SECTION, SETTINGS_SECTIONS);

    for (const item of refs.settingsNavItems) {
        const isActive = item.dataset.settingsSection === target;
        item.classList.toggle("active", isActive);
        if (isActive) {
            item.setAttribute("aria-current", "page");
        } else {
            item.removeAttribute("aria-current");
        }
    }

    for (const panel of refs.settingsPanels) {
        panel.classList.toggle("active", panel.dataset.settingsPanel === target);
    }

    if (persist && state.settings && state.settings.uiState) {
        state.settings.uiState.activeSection = target;
    }
}

function applyThemePreset(themePreset) {
    const resolved = pickString(themePreset, "cyber", ["cyber", "sunset"]);
    document.body.setAttribute("data-theme", resolved);
}

function applyDisplayState(uiState) {
    const scale = clampNumber(uiState.displayFontScale, 1, { min: 0.9, max: 1.3, digits: 2 });
    document.documentElement.style.setProperty("--ui-font-scale", String(scale));
    document.body.setAttribute("data-chat-ui", pickString(uiState.chatUiMode, "bubbleOnly", CHAT_UI_MODES));
    document.body.setAttribute("data-density", uiState.displayDensity || "normal");
    document.body.classList.toggle("reduced-motion", !uiState.displayAnimations);
}

function applyProfileSettingsToControls() {
    const profile = getUserProfile();
    if (refs.profileUserName) {
        refs.profileUserName.value = profile.userName;
    }
    renderProfileAvatarPreview(profile.userName, profile.userAvatarDataUrl);
}

function updateTimePreview() {
    if (!state.settings || !refs.timePreviewText) {
        return;
    }

    const tz = state.settings.uiState.timezone || "local";
    const format = state.settings.uiState.timeFormat || "24h";
    const opts = {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: format === "12h",
    };

    if (tz !== "local") {
        opts.timeZone = tz;
    }

    try {
        const display = new Intl.DateTimeFormat("zh-Hant-TW", opts).format(new Date());
        refs.timePreviewText.textContent = `時間預覽：${display}${tz === "local" ? "（系統）" : `（${tz}）`}`;
    } catch {
        refs.timePreviewText.textContent = "時間預覽：格式錯誤";
    }
}

function syncSettingsParamsFromControls() {
    if (!state.settings) {
        return;
    }
    state.settings.params.historyRounds = clampNumber(refs.historyRounds.value, state.settings.params.historyRounds, { min: 0, max: 40, integer: true });
    state.settings.params.maxLength = clampNumber(refs.maxLength.value, state.settings.params.maxLength, { min: 128, max: 8192, integer: true });
    state.settings.params.maxNewTokens = clampNumber(refs.maxNewTokens.value, state.settings.params.maxNewTokens, { min: 16, max: 4096, integer: true });
    state.settings.params.topK = clampNumber(refs.topK.value, state.settings.params.topK, { min: 1, max: 200, integer: true });
    state.settings.params.temperature = clampNumber(refs.temperature.value, state.settings.params.temperature, { min: 0.1, max: 1.8, digits: 2 });
    state.settings.params.topP = clampNumber(refs.topP.value, state.settings.params.topP, { min: 0.1, max: 1.0, digits: 2 });
    state.settings.params.repetitionPenalty = clampNumber(refs.repetitionPenalty.value, state.settings.params.repetitionPenalty, { min: 1.0, max: 2.0, digits: 2 });
    state.settings.params.doSample = Boolean(refs.doSample.checked);
}

function applyParamsToControls() {
    if (!state.settings) {
        return;
    }
    const params = state.settings.params;
    refs.historyRounds.value = String(params.historyRounds);
    refs.maxLength.value = String(params.maxLength);
    refs.maxNewTokens.value = String(params.maxNewTokens);
    refs.topK.value = String(params.topK);
    refs.temperature.value = String(params.temperature);
    refs.topP.value = String(params.topP);
    refs.repetitionPenalty.value = String(params.repetitionPenalty);
    refs.doSample.checked = Boolean(params.doSample);
}

function applyFeatureToggles(forceRender = false) {
    if (!state.settings) {
        return;
    }
    const feature = state.settings.featureToggles;
    const capabilities = bootstrap.capabilities || {};
    const canUseVoiceTranscription = capabilities.voiceTranscription !== false;
    const voiceInputEnabled = feature.voiceInput && canUseVoiceTranscription;

    if (!canUseVoiceTranscription) {
        state.settings.featureToggles.voiceInput = false;
        if (refs.toolVoiceInput) {
            refs.toolVoiceInput.checked = false;
            refs.toolVoiceInput.disabled = true;
        }
    } else if (refs.toolVoiceInput) {
        refs.toolVoiceInput.disabled = false;
    }

    if (refs.micBtn) {
        refs.micBtn.disabled = !voiceInputEnabled;
        refs.micBtn.setAttribute("title", voiceInputEnabled ? "語音輸入" : "語音輸入已停用");
    }

    document.body.classList.toggle("feature-history-disabled", !feature.conversationHistory);
    document.body.classList.toggle("feature-message-actions-disabled", !feature.messageActions);

    if (refs.searchConversationInput) {
        refs.searchConversationInput.disabled = !feature.conversationHistory;
        if (!feature.conversationHistory) {
            refs.searchConversationInput.value = "";
        }
    }

    if (feature.healthMonitor) {
        if (!state.healthCheckTimerId) {
            state.healthCheckTimerId = window.setInterval(() => {
                void refreshBackendHealth();
            }, 45000);
        }
        void refreshBackendHealth();
    } else {
        if (state.healthCheckTimerId) {
            window.clearInterval(state.healthCheckTimerId);
            state.healthCheckTimerId = null;
        }
        setStatus("offline");
    }

    if (forceRender && state.conversations.length > 0) {
        renderConversationList();
        if (!state.isSending) {
            renderActiveConversation();
        }
    }
}

function applySettingsToUI() {
    if (!state.settings) {
        return;
    }

    refs.modelPath.textContent = bootstrap.modelPath || "";
    refs.tokenizerPath.textContent = bootstrap.tokenizerPath || "";

    applyParamsToControls();

    refs.themePreset.value = state.settings.uiState.themePreset;
    refs.displayFontScale.value = String(state.settings.uiState.displayFontScale);
    if (refs.displayChatUiMode) {
        refs.displayChatUiMode.value = state.settings.uiState.chatUiMode;
    }
    refs.displayDensity.value = state.settings.uiState.displayDensity;
    refs.displayAnimations.checked = Boolean(state.settings.uiState.displayAnimations);
    if (refs.displayMetaTokensPerSec) {
        refs.displayMetaTokensPerSec.checked = Boolean(state.settings.uiState.metaShowTokensPerSec);
    }
    if (refs.displayMetaTokens) {
        refs.displayMetaTokens.checked = Boolean(state.settings.uiState.metaShowTokens);
    }
    if (refs.displayMetaElapsed) {
        refs.displayMetaElapsed.checked = Boolean(state.settings.uiState.metaShowElapsed);
    }
    if (refs.displayMetaStopReason) {
        refs.displayMetaStopReason.checked = Boolean(state.settings.uiState.metaShowStopReason);
    }
    refs.shortcutEnterToSend.checked = Boolean(state.settings.uiState.shortcutEnterToSend);
    refs.shortcutEscClosePanels.checked = Boolean(state.settings.uiState.shortcutEscClosePanels);
    refs.shortcutSlashFocusInput.checked = Boolean(state.settings.uiState.shortcutSlashFocusInput);
    refs.languageSelect.value = state.settings.uiState.language;
    refs.timeFormatSelect.value = state.settings.uiState.timeFormat;
    refs.timezoneSelect.value = state.settings.uiState.timezone;
    if (refs.statusIdleSeconds) {
        refs.statusIdleSeconds.value = String(clampNumber(state.settings.uiState.statusIdleSeconds, 60, { min: 10, max: 3600, integer: true }));
    }

    refs.avEchoCancellation.checked = Boolean(state.settings.devicePrefs.echoCancellation);
    refs.avNoiseSuppression.checked = Boolean(state.settings.devicePrefs.noiseSuppression);
    refs.avAutoGainControl.checked = Boolean(state.settings.devicePrefs.autoGainControl);
    refs.avMicGain.value = String(state.settings.devicePrefs.micGain);

    refs.toolVoiceInput.checked = Boolean(state.settings.featureToggles.voiceInput);
    refs.toolConversationHistory.checked = Boolean(state.settings.featureToggles.conversationHistory);
    refs.toolMessageActions.checked = Boolean(state.settings.featureToggles.messageActions);
    refs.toolHealthMonitor.checked = Boolean(state.settings.featureToggles.healthMonitor);
    applyProfileSettingsToControls();

    updateRangeLabels();
    updateMicGainLabel();
    updateDisplayScaleLabel();
    setActiveSettingsSection(state.settings.uiState.activeSection, false);
    applyThemePreset(state.settings.uiState.themePreset);
    applyDisplayState(state.settings.uiState);
    applyFeatureToggles();
    updateTimePreview();

    if (state.timePreviewTimerId) {
        window.clearInterval(state.timePreviewTimerId);
    }
    state.timePreviewTimerId = window.setInterval(updateTimePreview, TIME_PREVIEW_UPDATE_MS);
}

function syncSettingsFromUi() {
    if (!state.settings) {
        state.settings = getDefaultSettings();
    }

    syncSettingsParamsFromControls();

    state.settings.uiState.themePreset = refs.themePreset.value;
    state.settings.uiState.displayFontScale = clampNumber(refs.displayFontScale.value, 1, { min: 0.9, max: 1.3, digits: 2 });
    state.settings.uiState.chatUiMode = pickString(
        refs.displayChatUiMode ? refs.displayChatUiMode.value : state.settings.uiState.chatUiMode,
        "bubbleOnly",
        CHAT_UI_MODES
    );
    state.settings.uiState.displayDensity = refs.displayDensity.value;
    state.settings.uiState.displayAnimations = Boolean(refs.displayAnimations.checked);
    state.settings.uiState.metaShowTokensPerSec = Boolean(refs.displayMetaTokensPerSec ? refs.displayMetaTokensPerSec.checked : state.settings.uiState.metaShowTokensPerSec);
    state.settings.uiState.metaShowTokens = Boolean(refs.displayMetaTokens ? refs.displayMetaTokens.checked : state.settings.uiState.metaShowTokens);
    state.settings.uiState.metaShowElapsed = Boolean(refs.displayMetaElapsed ? refs.displayMetaElapsed.checked : state.settings.uiState.metaShowElapsed);
    state.settings.uiState.metaShowStopReason = Boolean(refs.displayMetaStopReason ? refs.displayMetaStopReason.checked : state.settings.uiState.metaShowStopReason);
    state.settings.uiState.shortcutEnterToSend = Boolean(refs.shortcutEnterToSend.checked);
    state.settings.uiState.shortcutEscClosePanels = Boolean(refs.shortcutEscClosePanels.checked);
    state.settings.uiState.shortcutSlashFocusInput = Boolean(refs.shortcutSlashFocusInput.checked);
    state.settings.uiState.language = refs.languageSelect.value;
    state.settings.uiState.timeFormat = refs.timeFormatSelect.value;
    state.settings.uiState.timezone = refs.timezoneSelect.value;
    state.settings.uiState.statusIdleSeconds = clampNumber(
        refs.statusIdleSeconds ? refs.statusIdleSeconds.value : state.settings.uiState.statusIdleSeconds,
        60,
        { min: 10, max: 3600, integer: true }
    );

    if (!isObjectLike(state.settings.profile)) {
        state.settings.profile = cloneJson(getDefaultSettings().profile);
    }
    const prevProfile = getUserProfile();
    state.settings.profile.userName = sanitizeUserName(
        refs.profileUserName ? refs.profileUserName.value : prevProfile.userName,
        DEFAULT_USER_NAME
    );
    state.settings.profile.userAvatarDataUrl = normalizeAvatarDataUrl(
        state.settings.profile.userAvatarDataUrl,
        ""
    );
    renderProfileAvatarPreview(state.settings.profile.userName, state.settings.profile.userAvatarDataUrl);
    const profileChanged =
        prevProfile.userName !== state.settings.profile.userName ||
        prevProfile.userAvatarDataUrl !== state.settings.profile.userAvatarDataUrl;

    state.settings.devicePrefs.micDeviceId = refs.avMicDevice.value;
    state.settings.devicePrefs.speakerDeviceId = refs.avSpeakerDevice.value;
    state.settings.devicePrefs.cameraDeviceId = refs.avCameraDevice.value;
    state.settings.devicePrefs.echoCancellation = Boolean(refs.avEchoCancellation.checked);
    state.settings.devicePrefs.noiseSuppression = Boolean(refs.avNoiseSuppression.checked);
    state.settings.devicePrefs.autoGainControl = Boolean(refs.avAutoGainControl.checked);
    state.settings.devicePrefs.micGain = clampNumber(refs.avMicGain.value, 1, { min: 0.5, max: 2.0, digits: 2 });

    const prevFeatureHistory = state.settings.featureToggles.conversationHistory;

    state.settings.featureToggles.voiceInput = Boolean(refs.toolVoiceInput.checked);
    state.settings.featureToggles.conversationHistory = Boolean(refs.toolConversationHistory.checked);
    state.settings.featureToggles.messageActions = Boolean(refs.toolMessageActions.checked);
    state.settings.featureToggles.healthMonitor = Boolean(refs.toolHealthMonitor.checked);

    const historyChanged = prevFeatureHistory !== state.settings.featureToggles.conversationHistory;

    state.settings.schemaVersion = SETTINGS_SCHEMA_VERSION;
    applyThemePreset(state.settings.uiState.themePreset);
    applyDisplayState(state.settings.uiState);
    applyFeatureToggles(historyChanged);
    updateTimePreview();
    if (profileChanged && state.conversations.length > 0 && !state.isSending) {
        renderActiveConversation();
    }
    if (currentStatusKey === "online" && !state.isSending) {
        scheduleIdleStatusTransition();
    }
}

function saveAllSettings() {
    syncSettingsFromUi();
    persistSettingsToStorage();
    showToast(TXT.toastSettingsSaved);
}

function resetAllSettings() {
    state.settings = getDefaultSettings();
    applySettingsToUI();
    persistSettingsToStorage();
    showToast(TXT.toastSettingsReset);
}

function mediaApiAvailable() {
    return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && navigator.mediaDevices.enumerateDevices);
}

function speakerDeviceSelectionSupported() {
    return Boolean(refs.micTestPlayback && typeof refs.micTestPlayback.setSinkId === "function");
}

function fillDeviceSelect(selectNode, deviceList, fallbackName, preferredDeviceId = "") {
    if (!selectNode) {
        return;
    }

    selectNode.innerHTML = "";
    if (!Array.isArray(deviceList) || !deviceList.length) {
        const emptyOption = document.createElement("option");
        emptyOption.value = "";
        emptyOption.textContent = `沒有可用的${fallbackName}`;
        selectNode.appendChild(emptyOption);
        selectNode.disabled = true;
        return;
    }

    selectNode.disabled = false;
    for (let i = 0; i < deviceList.length; i += 1) {
        const device = deviceList[i];
        const option = document.createElement("option");
        option.value = device.deviceId || "";
        option.textContent = device.label || `${fallbackName} ${i + 1}`;
        selectNode.appendChild(option);
    }

    const preferredExists = preferredDeviceId && deviceList.some((d) => d.deviceId === preferredDeviceId);
    if (preferredExists) {
        selectNode.value = preferredDeviceId;
    } else {
        selectNode.selectedIndex = 0;
    }
}

async function applySpeakerDeviceToPlayback() {
    if (!refs.micTestPlayback || !refs.avSpeakerDevice) {
        return;
    }

    if (!speakerDeviceSelectionSupported()) {
        return;
    }

    const targetDeviceId = refs.avSpeakerDevice.value;
    if (!targetDeviceId) {
        return;
    }

    try {
        await refs.micTestPlayback.setSinkId(targetDeviceId);
    } catch {
        setAudioVideoStatus("喇叭切換失敗，已維持系統預設輸出");
    }
}

function updateSpeakerCapabilityHint() {
    if (!refs.avSpeakerDevice || !refs.speakerSupportHint) {
        return;
    }

    const supported = speakerDeviceSelectionSupported();
    if (supported) {
        refs.avSpeakerDevice.disabled = false;
        refs.speakerSupportHint.textContent = "可切換播放裝置（受 HTTPS/瀏覽器政策影響）。";
        return;
    }

    refs.avSpeakerDevice.disabled = true;
    refs.speakerSupportHint.textContent = "此瀏覽器不支援輸出裝置切換，將使用系統預設喇叭。";
}

function buildAudioConstraints() {
    const prefs = state.settings ? state.settings.devicePrefs : getDefaultSettings().devicePrefs;
    const constraints = {
        echoCancellation: Boolean(prefs.echoCancellation),
        noiseSuppression: Boolean(prefs.noiseSuppression),
        autoGainControl: Boolean(prefs.autoGainControl),
    };
    if (prefs.micDeviceId) {
        constraints.deviceId = { exact: prefs.micDeviceId };
    }
    return { audio: constraints };
}

async function refreshDeviceList(options = {}) {
    const requestPermission = options.requestPermission === true;

    if (!mediaApiAvailable()) {
        fillDeviceSelect(refs.avMicDevice, [], "麥克風");
        fillDeviceSelect(refs.avSpeakerDevice, [], "喇叭");
        fillDeviceSelect(refs.avCameraDevice, [], "攝影機");
        setAudioVideoStatus("瀏覽器不支援媒體裝置 API");
        updateSpeakerCapabilityHint();
        return;
    }

    if (requestPermission) {
        try {
            const tmpStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
            tmpStream.getTracks().forEach((track) => track.stop());
        } catch {
            try {
                const tmpAudio = await navigator.mediaDevices.getUserMedia({ audio: true });
                tmpAudio.getTracks().forEach((track) => track.stop());
            } catch {
                // keep going: enumerateDevices may still work with partial labels
            }
        }
    }

    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter((item) => item.kind === "audioinput");
        const audioOutputs = devices.filter((item) => item.kind === "audiooutput");
        const videoInputs = devices.filter((item) => item.kind === "videoinput");

        state.media.availableDevices.audioInput = audioInputs;
        state.media.availableDevices.audioOutput = audioOutputs;
        state.media.availableDevices.videoInput = videoInputs;

        const prefs = state.settings ? state.settings.devicePrefs : getDefaultSettings().devicePrefs;
        fillDeviceSelect(refs.avMicDevice, audioInputs, "麥克風", prefs.micDeviceId);
        fillDeviceSelect(refs.avSpeakerDevice, audioOutputs, "喇叭", prefs.speakerDeviceId);
        fillDeviceSelect(refs.avCameraDevice, videoInputs, "攝影機", prefs.cameraDeviceId);

        updateSpeakerCapabilityHint();
        syncSettingsFromUi();

        const deviceSummary = `已偵測 ${audioInputs.length} 麥克風 / ${audioOutputs.length} 喇叭 / ${videoInputs.length} 攝影機`;
        setAudioVideoStatus(deviceSummary);
    } catch {
        setAudioVideoStatus("讀取裝置清單失敗，請檢查瀏覽器權限");
    }
}

function resetMicMeter() {
    if (refs.micLevelBar) {
        refs.micLevelBar.style.width = "0%";
    }
}

function stopMicMonitor() {
    const monitor = state.media.micMonitor;
    if (monitor.frameId) {
        window.cancelAnimationFrame(monitor.frameId);
        monitor.frameId = null;
    }

    if (monitor.sourceNode) {
        monitor.sourceNode.disconnect();
        monitor.sourceNode = null;
    }
    monitor.analyserNode = null;

    if (monitor.audioContext) {
        monitor.audioContext.close().catch(() => {});
        monitor.audioContext = null;
    }

    if (monitor.stream) {
        monitor.stream.getTracks().forEach((track) => track.stop());
        monitor.stream = null;
    }

    resetMicMeter();
}

async function startMicMonitor() {
    if (!mediaApiAvailable()) {
        setAudioVideoStatus("瀏覽器不支援音訊監測");
        return;
    }

    stopMicMonitor();
    syncSettingsFromUi();

    try {
        const stream = await navigator.mediaDevices.getUserMedia(buildAudioConstraints());
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) {
            stream.getTracks().forEach((track) => track.stop());
            setAudioVideoStatus("此瀏覽器不支援 AudioContext");
            return;
        }

        const audioContext = new AudioCtx();
        const sourceNode = audioContext.createMediaStreamSource(stream);
        const analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 2048;
        sourceNode.connect(analyserNode);

        state.media.micMonitor.stream = stream;
        state.media.micMonitor.audioContext = audioContext;
        state.media.micMonitor.sourceNode = sourceNode;
        state.media.micMonitor.analyserNode = analyserNode;

        const samples = new Uint8Array(analyserNode.fftSize);
        const render = () => {
            if (!state.media.micMonitor.analyserNode) {
                return;
            }

            state.media.micMonitor.analyserNode.getByteTimeDomainData(samples);
            let sumSquares = 0;
            for (let i = 0; i < samples.length; i += 1) {
                const normalized = (samples[i] - 128) / 128;
                sumSquares += normalized * normalized;
            }
            const rms = Math.sqrt(sumSquares / samples.length);
            const gain = state.settings ? state.settings.devicePrefs.micGain : 1;
            const level = Math.min(1, rms * 5 * gain);

            if (refs.micLevelBar) {
                refs.micLevelBar.style.width = `${Math.round(level * 100)}%`;
            }

            state.media.micMonitor.frameId = window.requestAnimationFrame(render);
        };

        render();
        setAudioVideoStatus("麥克風音量監測中");
    } catch {
        setAudioVideoStatus("無法啟動麥克風監測，請確認權限與裝置");
    }
}

function stopMicRecordingTest() {
    const recorderState = state.media.testRecorder;
    if (recorderState.timerId) {
        window.clearTimeout(recorderState.timerId);
        recorderState.timerId = null;
    }
    if (recorderState.mediaRecorder && recorderState.mediaRecorder.state !== "inactive") {
        recorderState.mediaRecorder.stop();
    }
}

async function recordMicSample() {
    if (!mediaApiAvailable()) {
        setAudioVideoStatus("瀏覽器不支援錄音測試");
        return;
    }
    if (!window.MediaRecorder) {
        setAudioVideoStatus("此瀏覽器不支援 MediaRecorder");
        return;
    }

    const recorderState = state.media.testRecorder;
    if (recorderState.isRecording) {
        stopMicRecordingTest();
        return;
    }

    syncSettingsFromUi();

    try {
        const stream = await navigator.mediaDevices.getUserMedia(buildAudioConstraints());
        const mediaRecorder = new MediaRecorder(stream);

        recorderState.stream = stream;
        recorderState.mediaRecorder = mediaRecorder;
        recorderState.chunks = [];
        recorderState.isRecording = true;

        if (refs.recordMicTestBtn) {
            refs.recordMicTestBtn.textContent = "停止錄製";
        }

        mediaRecorder.addEventListener("dataavailable", (event) => {
            if (event.data && event.data.size > 0) {
                recorderState.chunks.push(event.data);
            }
        });

        mediaRecorder.addEventListener("stop", () => {
            recorderState.isRecording = false;
            if (refs.recordMicTestBtn) {
                refs.recordMicTestBtn.textContent = "錄製 5 秒測試";
            }

            if (recorderState.timerId) {
                window.clearTimeout(recorderState.timerId);
                recorderState.timerId = null;
            }

            if (recorderState.stream) {
                recorderState.stream.getTracks().forEach((track) => track.stop());
                recorderState.stream = null;
            }

            const blob = new Blob(recorderState.chunks, { type: "audio/webm" });
            recorderState.chunks = [];

            if (recorderState.objectUrl) {
                URL.revokeObjectURL(recorderState.objectUrl);
            }
            recorderState.objectUrl = URL.createObjectURL(blob);

            if (refs.micTestPlayback) {
                refs.micTestPlayback.src = recorderState.objectUrl;
                refs.micTestPlayback.load();
            }
            void applySpeakerDeviceToPlayback();
            setAudioVideoStatus("麥克風測試錄音完成，可直接播放");
        });

        mediaRecorder.start();
        recorderState.timerId = window.setTimeout(() => {
            if (mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
            }
        }, 5000);

        setAudioVideoStatus("麥克風錄音測試中（5 秒）");
    } catch {
        setAudioVideoStatus("無法開始錄音測試，請確認麥克風權限");
    }
}

function stopCameraPreview() {
    if (state.media.cameraStream) {
        state.media.cameraStream.getTracks().forEach((track) => track.stop());
        state.media.cameraStream = null;
    }
    if (refs.cameraPreview) {
        refs.cameraPreview.srcObject = null;
    }
}

async function startCameraPreview() {
    if (!mediaApiAvailable()) {
        setAudioVideoStatus("瀏覽器不支援攝影機預覽");
        return;
    }

    stopCameraPreview();
    syncSettingsFromUi();

    const videoConstraints = {};
    if (state.settings && state.settings.devicePrefs.cameraDeviceId) {
        videoConstraints.deviceId = { exact: state.settings.devicePrefs.cameraDeviceId };
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints });
        state.media.cameraStream = stream;

        if (refs.cameraPreview) {
            refs.cameraPreview.srcObject = stream;
            await refs.cameraPreview.play().catch(() => {});
        }

        setAudioVideoStatus("攝影機預覽啟用中");
    } catch {
        setAudioVideoStatus("無法開啟攝影機，請確認權限與裝置");
    }
}

function stopAllMediaTests() {
    stopMicMonitor();
    stopMicRecordingTest();
    stopCameraPreview();
}

function clearEditState() {
    state.editingConversationId = null;
    state.editingMessageIndex = null;
}

function setEditState(conversationId, messageIndex) {
    if (!conversationId || !Number.isInteger(messageIndex) || messageIndex < 0) {
        clearEditState();
        return;
    }
    state.editingConversationId = String(conversationId);
    state.editingMessageIndex = messageIndex;
}

function makeId() {
    return `conv_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function setStatus(statusKey) {
    const dot = refs.brandStatusDot;
    if (!dot) {
        return;
    }
    const resolvedKey = STATUS_META[statusKey] ? statusKey : "idle";
    if (resolvedKey !== "online") {
        clearStatusIdleTimer();
    }
    currentStatusKey = resolvedKey;
    const meta = STATUS_META[resolvedKey] || STATUS_META.idle;
    dot.className = `brand-status ${meta.className}`;
    dot.innerHTML = meta.content;
    dot.setAttribute("aria-label", `狀態: ${meta.label}`);
    dot.setAttribute("title", meta.label);
}

function markOnlineThenIdle(delayMs) {
    const effectiveDelay = delayMs !== null && delayMs !== undefined && Number.isFinite(Number(delayMs))
        ? Math.max(0, Number(delayMs))
        : getStatusIdleDelayMs();
    setStatus("online");
    scheduleIdleStatusTransition(effectiveDelay);
}

async function refreshBackendHealth() {
    if (state.settings && state.settings.featureToggles.healthMonitor === false) {
        return false;
    }
    if (state.isSending) {
        return true;
    }
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => {
        controller.abort();
    }, 5000);

    try {
        const response = await fetch("/api/health", {
            method: "GET",
            cache: "no-store",
            signal: controller.signal,
        });
        if (!response.ok) {
            setStatus("offline");
            return false;
        }
        const payload = await response.json();
        if (payload && payload.ok === true) {
            if (currentStatusKey === "offline" || currentStatusKey === "error") {
                markOnlineThenIdle();
            } else if (currentStatusKey === "online" && !statusIdleTimer) {
                scheduleIdleStatusTransition();
            }
            return true;
        }
        setStatus("offline");
        return false;
    } catch {
        setStatus("offline");
        return false;
    } finally {
        window.clearTimeout(timeoutId);
    }
}

function isBackendOfflineError(error) {
    if (error instanceof Error && error.name === "AbortError") {
        return true;
    }
    const message = error instanceof Error ? error.message : String(error || "");
    if (/failed to fetch|networkerror|network request failed|err_connection|err_internet_disconnected/i.test(message)) {
        return true;
    }
    const statusMatch = message.match(/HTTP\s+(\d+)/i);
    if (statusMatch) {
        const statusCode = Number(statusMatch[1]);
        return Number.isFinite(statusCode) && statusCode >= 500;
    }
    return false;
}

function showToast(text) {
    refs.toast.textContent = text;
    refs.toast.classList.add("show");

    if (toastHideTimer) {
        window.clearTimeout(toastHideTimer);
    }
    toastHideTimer = window.setTimeout(() => {
        refs.toast.classList.remove("show");
        toastHideTimer = null;
    }, 1800);
}

async function copyTextToClipboard(text) {
    const normalized = String(text || "");
    if (!normalized) {
        return false;
    }

    try {
        await navigator.clipboard.writeText(normalized);
        return true;
    } catch {
        try {
            const fallback = document.createElement("textarea");
            fallback.value = normalized;
            fallback.setAttribute("readonly", "readonly");
            fallback.style.position = "fixed";
            fallback.style.opacity = "0";
            fallback.style.pointerEvents = "none";
            document.body.appendChild(fallback);
            fallback.focus();
            fallback.select();
            const ok = document.execCommand("copy");
            document.body.removeChild(fallback);
            return Boolean(ok);
        } catch {
            return false;
        }
    }
}

function readFileAsDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("read-failed"));
        reader.readAsDataURL(file);
    });
}

async function handleProfileAvatarFileChange() {
    if (!refs.profileAvatarFile || !refs.profileAvatarFile.files || refs.profileAvatarFile.files.length === 0) {
        return;
    }

    const file = refs.profileAvatarFile.files[0];
    refs.profileAvatarFile.value = "";

    if (!file || !String(file.type || "").startsWith("image/")) {
        showToast(TXT.toastProfileAvatarTypeError);
        return;
    }

    if (file.size > PROFILE_AVATAR_MAX_FILE_SIZE_BYTES) {
        showToast(TXT.toastProfileAvatarSizeError);
        return;
    }

    try {
        const dataUrl = await readFileAsDataUrl(file);
        const normalized = normalizeAvatarDataUrl(dataUrl, "");
        if (!normalized) {
            showToast(TXT.toastProfileAvatarReadFail);
            return;
        }

        if (!state.settings) {
            state.settings = getDefaultSettings();
        }
        if (!isObjectLike(state.settings.profile)) {
            state.settings.profile = cloneJson(getDefaultSettings().profile);
        }

        state.settings.profile.userAvatarDataUrl = normalized;
        const profile = getUserProfile();
        renderProfileAvatarPreview(profile.userName, profile.userAvatarDataUrl);
        if (state.conversations.length > 0 && !state.isSending) {
            renderActiveConversation();
        }
        showToast(TXT.toastProfileAvatarUpdated);
    } catch {
        showToast(TXT.toastProfileAvatarReadFail);
    }
}

function focusPromptWithText(text) {
    refs.promptInput.value = String(text || "");
    autoResizeTextarea();
    refs.promptInput.focus();
    const end = refs.promptInput.value.length;
    refs.promptInput.setSelectionRange(end, end);
}

function formatTime(ts) {
    const d = new Date(ts);
    const now = new Date();
    const sameDay =
        d.getFullYear() === now.getFullYear() &&
        d.getMonth() === now.getMonth() &&
        d.getDate() === now.getDate();

    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    if (sameDay) {
        return `${hh}:${mm}`;
    }
    return `${d.getMonth() + 1}/${d.getDate()} ${hh}:${mm}`;
}

function getChatUiMode() {
    if (!state.settings || !state.settings.uiState) {
        return "bubbleOnly";
    }
    return pickString(state.settings.uiState.chatUiMode, "bubbleOnly", CHAT_UI_MODES);
}

function getMessageDisplayName(role) {
    if (role === "assistant") {
        return "長門櫻";
    }
    return getUserProfile().userName;
}

function formatMessageTimestamp(ts) {
    const rawTs = Number(ts);
    const date = Number.isFinite(rawTs) && rawTs > 0 ? new Date(rawTs) : new Date();
    const uiState = state.settings ? state.settings.uiState : getDefaultSettings().uiState;
    const options = {
        hour: "2-digit",
        minute: "2-digit",
        hour12: uiState.timeFormat === "12h",
    };

    if (uiState.timezone && uiState.timezone !== "local") {
        options.timeZone = uiState.timezone;
    }

    try {
        return new Intl.DateTimeFormat("zh-Hant-TW", options).format(date);
    } catch {
        return formatTime(date.getTime());
    }
}

function createMessageAvatar(role) {
    const avatarNode = document.createElement("div");
    avatarNode.className = `message-avatar ${role}`;
    avatarNode.setAttribute("aria-hidden", "true");

    if (role === "assistant") {
        const src = refs.roleInfoAvatar ? refs.roleInfoAvatar.getAttribute("src") : "";
        if (src) {
            const img = document.createElement("img");
            img.src = src;
            img.alt = "";
            avatarNode.appendChild(img);
            return avatarNode;
        }
        avatarNode.textContent = "櫻";
        return avatarNode;
    }

    const profile = getUserProfile();
    if (profile.userAvatarDataUrl) {
        const img = document.createElement("img");
        img.src = profile.userAvatarDataUrl;
        img.alt = "";
        avatarNode.appendChild(img);
        return avatarNode;
    }

    avatarNode.textContent = getDisplayInitial(profile.userName, "主");
    return avatarNode;
}

function makeMetaText(meta) {
    if (!meta) {
        return "";
    }

    const uiState = state.settings ? state.settings.uiState : getDefaultSettings().uiState;
    const showTokensPerSec = boolValue(uiState.metaShowTokensPerSec, true);
    const showTokens = boolValue(uiState.metaShowTokens, true);
    const showElapsed = boolValue(uiState.metaShowElapsed, true);
    const showStopReason = boolValue(uiState.metaShowStopReason, true);

    const elapsedValue = Number(meta.elapsed);
    const tokensValue = Number(meta.tokens);
    const stopReason = String(meta.stopReason || meta.stop_reason || "").trim();
    const parts = [];

    if (showTokensPerSec) {
        const explicitTps = Number(meta.tokensPerSecond ?? meta.tokens_per_second);
        const computedTps = Number.isFinite(explicitTps)
            ? explicitTps
            : (Number.isFinite(tokensValue) && Number.isFinite(elapsedValue) && elapsedValue > 0)
                ? tokensValue / elapsedValue
                : NaN;
        if (Number.isFinite(computedTps) && computedTps > 0) {
            parts.push(`${computedTps.toFixed(2)} token/s`);
        }
    }

    if (showTokens && Number.isFinite(tokensValue) && tokensValue >= 0) {
        parts.push(`${Math.round(tokensValue)} tokens`);
    }

    if (showElapsed && Number.isFinite(elapsedValue) && elapsedValue >= 0) {
        parts.push(`${elapsedValue.toFixed(2)}s`);
    }

    if (showStopReason && stopReason) {
        parts.push(`${stopReason}`);
    }

    return parts.join(" · ");
}

function makeConversationTitle(text) {
    const normalized = String(text || "").replace(/\s+/g, " ").trim();
    if (!normalized) {
        return TXT.defaultChatTitle;
    }
    return normalized.length > 20 ? `${normalized.slice(0, 20)}...` : normalized;
}

function normalizeConversation(raw) {
    if (!raw || typeof raw !== "object") {
        return null;
    }

    const id = String(raw.id || "").trim();
    if (!id) {
        return null;
    }

    const messages = Array.isArray(raw.messages)
        ? raw.messages
            .filter((m) => m && typeof m === "object")
            .map((m) => ({
                role: m.role === "assistant" ? "assistant" : "user",
                content: String(m.content || ""),
                meta: m.meta || null,
                ts: clampNumber(m.ts ?? m.timestamp, nowTs(), { min: 0, integer: true }),
            }))
            .filter((m) => m.content.trim().length > 0)
        : [];

    return {
        id,
        title: String(raw.title || TXT.defaultChatTitle),
        isTemporary: Boolean(raw.isTemporary),
        createdAt: Number(raw.createdAt || nowTs()),
        updatedAt: Number(raw.updatedAt || nowTs()),
        messages,
    };
}

function hasUserMessage(conv) {
    return conv.messages.some((m) => m.role === "user");
}

function hasAssistantMessage(conv) {
    return conv.messages.some((m) => m.role === "assistant");
}

function isConversationReadyForHistory(conv) {
    return !conv.isTemporary && hasUserMessage(conv) && hasAssistantMessage(conv);
}

function persistConversations() {
    const savable = state.conversations
        .filter((c) => isConversationReadyForHistory(c))
        .map((c) => ({
            id: c.id,
            title: c.title,
            isTemporary: false,
            createdAt: c.createdAt,
            updatedAt: c.updatedAt,
            messages: c.messages,
        }))
        .slice(0, 120);

    localStorage.setItem(STORAGE_KEY, JSON.stringify(savable));
}

function createConversation(isTemporary = false, silent = false) {
    clearEditState();
    state.conversations = state.conversations.filter((conv) => isConversationReadyForHistory(conv));

    const ts = nowTs();
    const conv = {
        id: makeId(),
        title: isTemporary ? TXT.tempChatTitle : TXT.defaultChatTitle,
        isTemporary,
        createdAt: ts,
        updatedAt: ts,
        messages: [],
    };

    state.conversations.unshift(conv);
    state.activeConversationId = conv.id;

    if (!silent) {
        showToast(isTemporary ? TXT.toastTempChat : TXT.toastNewChat);
    }

    return conv;
}

function loadConversations() {
    let restored = [];
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        if (Array.isArray(parsed)) {
            restored = parsed
                .map(normalizeConversation)
                .filter(Boolean)
                .filter((conv) => hasUserMessage(conv))
                .map((conv) => ({ ...conv, isTemporary: false }));
        }
    } catch {
        restored = [];
    }

    state.conversations = restored.sort((a, b) => b.updatedAt - a.updatedAt);
    const conv = createConversation(false, true);
    state.activeConversationId = conv.id;
}

function getActiveConversation() {
    let conv = state.conversations.find((c) => c.id === state.activeConversationId);
    if (!conv) {
        conv = createConversation(false, true);
        state.activeConversationId = conv.id;
    }
    return conv;
}

function touchConversation(conv) {
    conv.updatedAt = nowTs();
}

function syncSidebarA11y() {
    const expanded = !state.sidebarCollapsed;
    refs.leftSidebar.setAttribute("aria-hidden", "false");
    refs.closeSidebarBtn.setAttribute("aria-expanded", expanded ? "true" : "false");
    refs.closeSidebarBtn.setAttribute("aria-label", expanded ? "摺疊側欄" : "展開側欄");
    refs.closeSidebarBtn.setAttribute("title", expanded ? "摺疊側欄" : "展開側欄");
    refs.closeSidebarBtn.textContent = expanded ? "◀" : "▶";
}

function applySidebarState(persist = true) {
    refs.appShell.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
    refs.leftSidebar.classList.toggle("collapsed", state.sidebarCollapsed);
    syncSidebarA11y();
    if (persist) {
        localStorage.setItem(SIDEBAR_STATE_KEY, state.sidebarCollapsed ? "1" : "0");
    }
}

function restoreSidebarState() {
    const savedState = localStorage.getItem(SIDEBAR_STATE_KEY);
    state.sidebarCollapsed = savedState === "1";
    applySidebarState(false);
}

function closeSidebar() {
    state.sidebarCollapsed = true;
    applySidebarState();
}

function openSidebar() {
    state.sidebarCollapsed = false;
    applySidebarState();
}

function toggleSidebar() {
    if (state.sidebarCollapsed) {
        openSidebar();
        return;
    }
    closeSidebar();
}

function openSettings() {
    refs.settingsModal.classList.add("open");
    refs.settingsModal.setAttribute("aria-hidden", "false");
    refs.settingsBackdrop.classList.add("show");
    if (state.settings) {
        setActiveSettingsSection(state.settings.uiState.activeSection || DEFAULT_ACTIVE_SETTINGS_SECTION, false);
    }
    updateTimePreview();
    void refreshDeviceList({ requestPermission: false });
}

function closeSettings() {
    refs.settingsModal.classList.remove("open");
    refs.settingsModal.setAttribute("aria-hidden", "true");
    refs.settingsBackdrop.classList.remove("show");
    stopAllMediaTests();
}

function getConversationPreview(conv) {
    const last = [...conv.messages].reverse().find((m) => m.content && m.content.trim());
    if (!last) {
        return TXT.noMessageYet;
    }
    return String(last.content).replace(/\s+/g, " ").slice(0, 36);
}

function normalizeMessageMarkdownText(rawText) {
    const text = String(rawText || "");
    if (!text) {
        return "";
    }

    // Some model outputs include escaped newlines in a single line fenced block.
    if (text.includes("```") && text.includes("\\n") && !text.includes("\n")) {
        return text
            .replace(/\\r\\n/g, "\n")
            .replace(/\\n/g, "\n")
            .replace(/\\t/g, "\t");
    }

    return text;
}

function appendPlainContentBlock(contentNode, text) {
    if (!text) {
        return;
    }
    const textNode = document.createElement("div");
    textNode.className = "content-text";
    textNode.textContent = text;
    contentNode.appendChild(textNode);
}

function appendCodeContentBlock(contentNode, codeText, language = "") {
    const codeWrap = document.createElement("div");
    codeWrap.className = "content-code-block";

    const normalizedLanguage = String(language || "").trim();
    const normalizedCodeText = String(codeText || "");

    const headNode = document.createElement("div");
    headNode.className = "content-code-head";

    const langNode = document.createElement("div");
    langNode.className = "content-code-lang";
    langNode.textContent = normalizedLanguage || "code";
    headNode.appendChild(langNode);

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "content-code-copy-btn";
    copyBtn.innerHTML = '<svg class="content-code-copy-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><rect x="9" y="9" width="13" height="13" rx="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>';
    copyBtn.setAttribute("aria-label", "複製程式碼區塊");
    copyBtn.setAttribute("title", "複製程式碼");
    copyBtn.addEventListener("click", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        const ok = await copyTextToClipboard(normalizedCodeText);
        showToast(ok ? "已複製程式碼" : TXT.toastCopyMessageFail);
    });
    headNode.appendChild(copyBtn);
    codeWrap.appendChild(headNode);

    const preNode = document.createElement("pre");
    const codeNode = document.createElement("code");
    if (normalizedLanguage) {
        codeNode.className = `language-${normalizedLanguage.toLowerCase()}`;
    }
    codeNode.textContent = normalizedCodeText;
    preNode.appendChild(codeNode);
    codeWrap.appendChild(preNode);
    contentNode.appendChild(codeWrap);
}

function renderMessageContent(contentNode, rawText) {
    const text = normalizeMessageMarkdownText(rawText);
    contentNode.innerHTML = "";

    if (!text) {
        return;
    }

    const fenceRegex = /```([a-zA-Z0-9_+-]*)[ \t]*\n?([\s\S]*?)```/g;
    let cursor = 0;
    let match;

    while ((match = fenceRegex.exec(text)) !== null) {
        const prefixText = text.slice(cursor, match.index);
        appendPlainContentBlock(contentNode, prefixText);

        const language = match[1] || "";
        const codeBody = match[2] || "";
        appendCodeContentBlock(contentNode, codeBody, language);

        cursor = fenceRegex.lastIndex;
    }

    const suffixText = text.slice(cursor);
    appendPlainContentBlock(contentNode, suffixText);

    // If no blocks were appended (edge case), keep plain text rendering.
    if (!contentNode.childNodes.length) {
        appendPlainContentBlock(contentNode, text);
    }
}

function createMessage(role, content, meta = null, options = {}) {
    const featureAllowsActions = !state.settings || state.settings.featureToggles.messageActions !== false;
    const includeActions = options.includeActions !== false && featureAllowsActions;
    const deferActionsUntilComplete = role === "assistant" && options.deferActionsUntilComplete === true;
    const messageIndex = Number.isInteger(options.messageIndex) ? options.messageIndex : null;
    const messageTimestamp = clampNumber(options.timestamp, nowTs(), { min: 0, integer: true });
    const chatUiMode = getChatUiMode();
    const showAvatar = chatUiMode !== "bubbleOnly";
    const showHeader = chatUiMode === "discord";

    const rowNode = document.createElement("div");
    rowNode.className = `message-row ${role}`;
    if (messageIndex !== null) {
        rowNode.dataset.messageIndex = String(messageIndex);
    }
    rowNode.dataset.messageTs = String(messageTimestamp);

    const bodyNode = document.createElement("div");
    bodyNode.className = "message-body";

    let actionsNode = null;

    function createActionsNode(initialMeta = null) {
        const node = document.createElement("div");
        node.className = "message-actions";

        const copyBtn = document.createElement("button");
        copyBtn.type = "button";
        copyBtn.className = "message-action-btn";
        copyBtn.dataset.action = "copy";
        copyBtn.textContent = "⧉";
        copyBtn.setAttribute("aria-label", "複製訊息");
        copyBtn.setAttribute("title", "複製");
        node.appendChild(copyBtn);

        const secondBtn = document.createElement("button");
        secondBtn.type = "button";
        secondBtn.className = "message-action-btn";
        if (role === "assistant") {
            secondBtn.dataset.action = "redo";
            secondBtn.textContent = "↻";
            secondBtn.setAttribute("aria-label", "重作回應");
            secondBtn.setAttribute("title", "重作");
        } else {
            secondBtn.dataset.action = "edit";
            secondBtn.textContent = "✎";
            secondBtn.setAttribute("aria-label", "編輯訊息");
            secondBtn.setAttribute("title", "編輯");
        }
        node.appendChild(secondBtn);

        const metaNode = document.createElement("div");
        metaNode.className = "message-meta-info";
        if (initialMeta) {
            metaNode.textContent = makeMetaText(initialMeta);
        }
        node.appendChild(metaNode);

        return node;
    }

    function ensureActionsNode(initialMeta = null) {
        if (!actionsNode) {
            actionsNode = createActionsNode(initialMeta);
            rowNode.appendChild(actionsNode);
        }
        return actionsNode;
    }

    if (showAvatar) {
        bodyNode.appendChild(createMessageAvatar(role));
    }

    const node = document.createElement("article");
    node.className = `message ${role}`;

    if (showHeader) {
        const headerNode = document.createElement("div");
        headerNode.className = "message-header";

        const authorNode = document.createElement("span");
        authorNode.className = "message-author";
        authorNode.textContent = getMessageDisplayName(role);

        const timeNode = document.createElement("time");
        timeNode.className = "message-time";
        timeNode.dateTime = new Date(messageTimestamp).toISOString();
        timeNode.textContent = formatMessageTimestamp(messageTimestamp);

        headerNode.appendChild(authorNode);
        headerNode.appendChild(timeNode);
        node.appendChild(headerNode);
    }

    const contentNode = document.createElement("div");
    contentNode.className = "content";
    renderMessageContent(contentNode, content || "");
    node.appendChild(contentNode);

    bodyNode.appendChild(node);
    rowNode.appendChild(bodyNode);

    if (includeActions && (role === "assistant" || role === "user")) {
        if (!(deferActionsUntilComplete && role === "assistant")) {
            ensureActionsNode(meta);
        }
    }

    refs.chatList.appendChild(rowNode);
    scrollChatToBottom();

    return {
        node,
        rowNode,
        contentNode,
        setText(value) {
            renderMessageContent(contentNode, value);
            scrollChatToBottom();
        },
        setActionsVisible(isVisible) {
            if (isVisible) {
                ensureActionsNode();
            } else if (actionsNode) {
                actionsNode.remove();
                actionsNode = null;
            }
        },
        setMeta(newMeta) {
            if (!actionsNode) {
                ensureActionsNode(newMeta);
                return;
            }
            let metaNode = rowNode.querySelector(".message-meta-info");
            if (metaNode) {
                metaNode.textContent = makeMetaText(newMeta);
            }
        },
        setMessageIndex(index) {
            if (Number.isInteger(index) && index >= 0) {
                rowNode.dataset.messageIndex = String(index);
            }
        },
    };
}

async function redoAssistantAtIndex(messageIndex) {
    if (state.isSending) {
        return;
    }

    clearEditState();
    const conv = getActiveConversation();
    const target = conv.messages[messageIndex];
    if (!target || target.role !== "assistant") {
        return;
    }

    let userIndex = messageIndex - 1;
    while (userIndex >= 0 && conv.messages[userIndex].role !== "user") {
        userIndex -= 1;
    }

    if (userIndex < 0) {
        showToast(TXT.toastRedoMissingUser);
        return;
    }

    const userMessage = conv.messages[userIndex];
    conv.messages = conv.messages.slice(0, userIndex + 1);
    touchConversation(conv);
    if (!conv.isTemporary) {
        persistConversations();
    }

    renderConversationList();
    renderActiveConversation();
    showToast(TXT.toastRedoStart);

    const history = conv.isTemporary
        ? [{ role: "user", content: userMessage.content }]
        : conv.messages.slice();

    await sendPrompt({
        text: userMessage.content,
        appendUserMessage: false,
        history,
        clearPrompt: false,
        useEditingState: false,
    });
}

async function handleMessageAction(action, messageIndex) {
    if (state.settings && state.settings.featureToggles.messageActions === false) {
        return;
    }

    const conv = getActiveConversation();
    const msg = conv.messages[messageIndex];
    if (!msg) {
        return;
    }

    if (action === "copy") {
        const ok = await copyTextToClipboard(msg.content);
        showToast(ok ? TXT.toastCopyMessageOk : TXT.toastCopyMessageFail);
        return;
    }

    if (action === "edit") {
        if (msg.role !== "user") {
            return;
        }
        setEditState(conv.id, messageIndex);
        focusPromptWithText(msg.content);
        showToast(TXT.toastEditMode);
        return;
    }

    if (action === "redo") {
        if (msg.role !== "assistant") {
            return;
        }
        await redoAssistantAtIndex(messageIndex);
    }
}

function clearEmptyChatHint() {
    const hintNode = refs.chatList.querySelector(".empty-chat");
    if (hintNode) {
        hintNode.remove();
    }
}

function isChatNearBottom(threshold = CHAT_BOTTOM_THRESHOLD_PX) {
    if (!refs.chatList) {
        return true;
    }
    const remaining = refs.chatList.scrollHeight - refs.chatList.scrollTop - refs.chatList.clientHeight;
    return remaining <= threshold;
}

function updateJumpToLatestVisibility() {
    if (!refs.chatJumpWrap || !refs.jumpToLatestBtn || !refs.chatList) {
        return;
    }
    const scrollable = refs.chatList.scrollHeight > refs.chatList.clientHeight + 4;
    const shouldShow = scrollable && !isChatNearBottom();
    refs.chatJumpWrap.classList.toggle("show", shouldShow);
    refs.chatJumpWrap.setAttribute("aria-hidden", shouldShow ? "false" : "true");
    refs.jumpToLatestBtn.tabIndex = shouldShow ? 0 : -1;
}

function scrollChatToBottom(options = {}) {
    if (!refs.chatList) {
        return;
    }
    const behavior = options.smooth ? "smooth" : "auto";
    refs.chatList.scrollTo({ top: refs.chatList.scrollHeight, behavior });
    updateJumpToLatestVisibility();
}

function renderConversationList() {
    refs.conversationList.innerHTML = "";

    if (state.settings && state.settings.featureToggles.conversationHistory === false) {
        refs.conversationCount.textContent = "0";
        const disabledNode = document.createElement("div");
        disabledNode.className = "conversation-empty";
        disabledNode.textContent = "對話歷史已在註冊工具中停用";
        refs.conversationList.appendChild(disabledNode);
        return;
    }

    const query = refs.searchConversationInput.value.trim().toLowerCase();
    const sorted = [...state.conversations]
        .filter((conv) => isConversationReadyForHistory(conv))
        .sort((a, b) => b.updatedAt - a.updatedAt);
    const filtered = sorted.filter((conv) => {
        if (!query) {
            return true;
        }
        const title = String(conv.title || "").toLowerCase();
        const messageText = conv.messages.map((m) => String(m.content || "")).join(" ").toLowerCase();
        return title.includes(query) || messageText.includes(query);
    });

    refs.conversationCount.textContent = String(filtered.length);

    if (!filtered.length) {
        const emptyNode = document.createElement("div");
        emptyNode.className = "conversation-empty";
        emptyNode.textContent = query ? TXT.noSearchResults : TXT.noHistoryYet;
        refs.conversationList.appendChild(emptyNode);
        return;
    }

    for (const conv of filtered) {
        const item = document.createElement("div");
        item.setAttribute("role", "button");
        item.tabIndex = 0;
        item.className = `conversation-item${conv.id === state.activeConversationId ? " active" : ""}`;

        const titleRow = document.createElement("div");
        titleRow.className = "conv-title-row";

        const title = document.createElement("span");
        title.className = "conv-title";
        title.textContent = conv.title || TXT.defaultChatTitle;
        titleRow.appendChild(title);

        const actions = document.createElement("span");
        actions.className = "conv-actions";

        const deleteBtn = document.createElement("button");
        deleteBtn.type = "button";
        deleteBtn.className = "conversation-delete-btn";
        deleteBtn.setAttribute("aria-label", TXT.deleteConversationAria);
        deleteBtn.title = TXT.deleteConversationAria;
        deleteBtn.textContent = "🗑";
        deleteBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            event.preventDefault();
            deleteConversation(conv.id);
        });
        actions.appendChild(deleteBtn);
        titleRow.appendChild(actions);

        const preview = document.createElement("div");
        preview.className = "conv-preview";
        preview.textContent = getConversationPreview(conv);

        const meta = document.createElement("div");
        meta.className = "conv-meta";
        meta.textContent = formatTime(conv.updatedAt);

        item.appendChild(titleRow);
        item.appendChild(preview);
        item.appendChild(meta);

        const selectConversation = () => {
            if (state.isSending) {
                return;
            }
            clearEditState();
            state.activeConversationId = conv.id;
            renderConversationList();
            renderActiveConversation();
        };

        item.addEventListener("click", selectConversation);
        item.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                selectConversation();
            }
        });

        refs.conversationList.appendChild(item);
    }
}

function deleteConversation(conversationId) {
    if (state.isSending) {
        return;
    }

    const index = state.conversations.findIndex((conv) => conv.id === conversationId);
    if (index === -1) {
        return;
    }

    if (state.editingConversationId === conversationId) {
        clearEditState();
    }

    const isDeletingActive = state.activeConversationId === conversationId;
    state.conversations.splice(index, 1);

    if (isDeletingActive) {
        const fallback = [...state.conversations].sort((a, b) => b.updatedAt - a.updatedAt)[0];
        if (fallback) {
            state.activeConversationId = fallback.id;
        } else {
            const conv = createConversation(false, true);
            state.activeConversationId = conv.id;
        }
    }

    persistConversations();
    renderConversationList();
    renderActiveConversation();
    showToast(TXT.toastDeleteConversation);
}

function renderActiveConversation() {
    refs.chatList.innerHTML = "";
    const conv = getActiveConversation();

    if (!conv.messages.length) {
        const emptyNode = document.createElement("div");
        emptyNode.className = "empty-chat";
        emptyNode.textContent = TXT.emptyChatHint;
        refs.chatList.appendChild(emptyNode);
    }

    for (let i = 0; i < conv.messages.length; i += 1) {
        const msg = conv.messages[i];
        createMessage(msg.role, msg.content, msg.meta || null, { messageIndex: i, timestamp: msg.ts });
    }

    const prefix = conv.isTemporary ? TXT.tempChatPrefix : TXT.normalChatPrefix;
    refs.activeConversationLabel.textContent = `${prefix} · ${conv.title}`;
    scrollChatToBottom();
}

function resetActiveConversation() {
    const conv = getActiveConversation();
    clearEditState();
    conv.messages = [];
    conv.title = conv.isTemporary ? TXT.tempChatTitle : TXT.defaultChatTitle;
    touchConversation(conv);
    persistConversations();
    renderConversationList();
    renderActiveConversation();
}

function updateRangeLabels() {
    refs.temperatureLabel.textContent = Number(refs.temperature.value).toFixed(2);
    refs.topPLabel.textContent = Number(refs.topP.value).toFixed(2);
    refs.repetitionPenaltyLabel.textContent = Number(refs.repetitionPenalty.value).toFixed(2);
    updateMicGainLabel();
    updateDisplayScaleLabel();
}

function collectParams() {
    const fallback = state.settings ? state.settings.params : getDefaultSettings().params;
    return {
        history_rounds: clampNumber(refs.historyRounds.value, fallback.historyRounds, { min: 0, max: 40, integer: true }),
        max_length: clampNumber(refs.maxLength.value, fallback.maxLength, { min: 128, max: 8192, integer: true }),
        max_new_tokens: clampNumber(refs.maxNewTokens.value, fallback.maxNewTokens, { min: 16, max: 4096, integer: true }),
        temperature: clampNumber(refs.temperature.value, fallback.temperature, { min: 0.1, max: 1.8, digits: 2 }),
        top_k: clampNumber(refs.topK.value, fallback.topK, { min: 1, max: 200, integer: true }),
        top_p: clampNumber(refs.topP.value, fallback.topP, { min: 0.1, max: 1.0, digits: 2 }),
        repetition_penalty: clampNumber(refs.repetitionPenalty.value, fallback.repetitionPenalty, { min: 1.0, max: 2.0, digits: 2 }),
        do_sample: Boolean(refs.doSample.checked),
    };
}

let currentAbortController = null;

async function sendPrompt(options = {}) {
    const inputText = typeof options.text === "string" ? options.text : refs.promptInput.value;
    const text = inputText.trim();
    if (!text || state.isSending) {
        return;
    }

    const conv = getActiveConversation();
    const appendUserMessage = options.appendUserMessage !== false;
    const customHistory = Array.isArray(options.history) ? options.history : null;
    const clearPrompt = options.clearPrompt !== false;
    const useEditingState = options.useEditingState !== false;
    const hasEditableTarget =
        useEditingState &&
        state.editingConversationId === conv.id &&
        Number.isInteger(state.editingMessageIndex) &&
        state.editingMessageIndex >= 0 &&
        conv.messages[state.editingMessageIndex] &&
        conv.messages[state.editingMessageIndex].role === "user";
    const editMessageIndex = hasEditableTarget ? state.editingMessageIndex : null;
    if (useEditingState && state.editingConversationId === conv.id && editMessageIndex === null && state.editingMessageIndex !== null) {
        clearEditState();
    }

    state.isSending = true;
    currentAbortController = new AbortController();

    if (refs.micBtn) {
        refs.micBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12" rx="2" ry="2"/></svg>`;
        refs.micBtn.style.color = "var(--c-pink)";
        refs.micBtn.setAttribute("aria-label", "停止生成");
        refs.micBtn.setAttribute("title", "停止生成");
    }

    setStatus("generating");

    if (clearPrompt) {
        refs.promptInput.value = "";
        autoResizeTextarea();
    }

    const userMessageTs = nowTs();

    if (editMessageIndex !== null) {
        const prefixMessages = conv.messages.slice(0, editMessageIndex);
        conv.messages = [...prefixMessages, { role: "user", content: text, ts: userMessageTs }];
        clearEditState();
    } else if (appendUserMessage) {
        conv.messages.push({ role: "user", content: text, ts: userMessageTs });
        if (conv.title === TXT.defaultChatTitle || conv.title === TXT.tempChatTitle) {
            conv.title = makeConversationTitle(text);
        }
    }

    touchConversation(conv);
    if (!conv.isTemporary) {
        persistConversations();
    }

    renderConversationList();
    if (editMessageIndex !== null) {
        renderActiveConversation();
    } else {
        clearEmptyChatHint();
    }
    
    let userMessageIndex = conv.messages.length - 1;
    if (appendUserMessage && editMessageIndex === null) {
        createMessage("user", text, null, { includeActions: true, messageIndex: userMessageIndex, timestamp: userMessageTs });
    }

    const assistantIndex = conv.messages.length; // 預計的 assistant 索引
    const assistantMessageTs = nowTs();
    const assistantView = createMessage("assistant", "", null, {
        includeActions: true,
        deferActionsUntilComplete: true,
        messageIndex: assistantIndex,
        timestamp: assistantMessageTs,
    });
    let assistantText = "";
    let finalMeta = null;
    syncSettingsParamsFromControls();
    const params = collectParams();
    const requestHistory = customHistory || (conv.isTemporary
        ? [{ role: "user", content: text }]
        : conv.messages);

    if (conv.isTemporary) {
        params.stateless_chat = true;
    }

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            signal: currentAbortController.signal,
            body: JSON.stringify({
                message: text,
                history: requestHistory,
                params,
            }),
        });

        if (!response.ok || !response.body) {
            throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) {
                    continue;
                }

                const payload = JSON.parse(trimmed);
                if (payload.type === "delta") {
                    assistantText += payload.delta || "";
                    assistantView.setText(assistantText);
                } else if (payload.type === "done") {
                    assistantText = payload.text || assistantText;
                    assistantView.setText(assistantText || TXT.emptyResponse);
                    finalMeta = {
                        elapsed: Number(payload.elapsed || 0),
                        tokens: Number(payload.tokens || 0),
                        stopReason: payload.stop_reason || payload.stopReason || "completed",
                    };
                } else if (payload.type === "error") {
                    throw new Error(payload.message || TXT.statusError);
                }
            }
        }

        if (!assistantText) {
            assistantText = TXT.emptyResponse;
            assistantView.setText(assistantText);
        }

        if (finalMeta) {
            assistantView.setMeta(finalMeta);
        }
        assistantView.setActionsVisible(true);

        conv.messages.push({ role: "assistant", content: assistantText, meta: finalMeta, ts: assistantMessageTs });
        touchConversation(conv);
        if (!conv.isTemporary) {
            persistConversations();
        }
        renderConversationList();
        // 移除了重新渲染 activeConversation 避免整個 DOM 重建造成的閃爍
        markOnlineThenIdle();
    } catch (err) {
        if (err.name === "AbortError") {
            finalMeta = {
                elapsed: 0,
                tokens: 0,
                stopReason: "aborted",
            };
            conv.messages.push({ role: "assistant", content: assistantText || "（生成已中斷）", meta: finalMeta, ts: assistantMessageTs });
            assistantView.setText(assistantText || "（生成已中斷）");
            assistantView.setActionsVisible(true);
            assistantView.setMeta(finalMeta);
            touchConversation(conv);
            if (!conv.isTemporary) {
                persistConversations();
            }
            renderConversationList();
            markOnlineThenIdle();
        } else {
            const errorMessage = `${TXT.chatErrorPrefix}${err instanceof Error ? err.message : String(err)}`;
            assistantView.setText(errorMessage);
            assistantView.setActionsVisible(true);
            conv.messages.push({ role: "assistant", content: errorMessage, ts: assistantMessageTs });
            touchConversation(conv);
            if (!conv.isTemporary) {
                persistConversations();
            }
            renderConversationList();
            setStatus(isBackendOfflineError(err) ? "offline" : "error");
        }
    } finally {
        state.isSending = false;
        currentAbortController = null;
        if (refs.micBtn) {
            refs.micBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>`;
            refs.micBtn.style.color = "";
            refs.micBtn.setAttribute("aria-label", "語音輸入");
            refs.micBtn.setAttribute("title", "語音輸入");
        }
        refs.promptInput.focus();
    }
}

function bootstrapSettings() {
    state.settings = loadSettingsFromStorage();
    applySettingsToUI();
}

function autoResizeTextarea() {
    const input = refs.promptInput;
    const minHeight = 46;
    const viewportHalf = Math.floor(window.innerHeight * 0.5);
    const maxHeight = Math.max(minHeight, viewportHalf);

    input.rows = 1;
    input.style.height = "auto";
    input.style.maxHeight = `${maxHeight}px`;

    const nextHeight = Math.max(minHeight, Math.min(maxHeight, input.scrollHeight));
    input.style.height = `${nextHeight}px`;
    input.style.overflowY = input.scrollHeight > maxHeight ? "auto" : "hidden";
    updateJumpToLatestAnchor();
}

function updateJumpToLatestAnchor() {
    if (!refs.chatJumpWrap || !refs.composerWrap) {
        return;
    }
    const composerHeight = Math.max(0, refs.composerWrap.offsetHeight);
    refs.chatJumpWrap.style.bottom = `${composerHeight + 10}px`;
}

function bindEvents() {
    refs.closeSidebarBtn.addEventListener("click", () => {
        toggleSidebar();
    });

    if (refs.roleInfoBtn) {
        refs.roleInfoBtn.addEventListener("click", () => {
            refs.appShell.classList.toggle("right-sidebar-expanded");
        });
    }

    if (refs.closeRightSidebarBtn) {
        refs.closeRightSidebarBtn.addEventListener("click", () => {
            refs.appShell.classList.remove("right-sidebar-expanded");
        });
    }

    if (refs.viewFullProfileBtn) {
        refs.viewFullProfileBtn.addEventListener("click", () => {
            openSettings();
            setActiveSettingsSection("profile", true);
            persistSettingsToStorage();
        });
    }

    if (refs.saveModelSettingsBtn) {
        refs.saveModelSettingsBtn.addEventListener("click", () => {
            syncSettingsParamsFromControls();
            persistSettingsToStorage();
            showToast(TXT.toastSettingsSaved);
        });
    }

    if (refs.resetModelSettingsBtn) {
        refs.resetModelSettingsBtn.addEventListener("click", () => {
            if (!state.settings) {
                state.settings = getDefaultSettings();
            }
            const defaults = getDefaultSettings();
            state.settings.params = cloneJson(defaults.params);
            applyParamsToControls();
            updateRangeLabels();
            persistSettingsToStorage();
            showToast(TXT.toastSettingsReset);
        });
    }

    refs.newChatBtn.addEventListener("click", () => {
        if (state.isSending) {
            return;
        }
        createConversation(false);
        renderConversationList();
        renderActiveConversation();
    });

    refs.tempChatBtn.addEventListener("click", () => {
        if (state.isSending) {
            return;
        }
        createConversation(true);
        renderConversationList();
        renderActiveConversation();
    });

    refs.searchConversationInput.addEventListener("input", () => {
        renderConversationList();
    });

    refs.openSettingsBtn.addEventListener("click", () => {
        openSettings();
    });

    refs.closeSettingsBtn.addEventListener("click", () => {
        closeSettings();
    });

    refs.settingsBackdrop.addEventListener("click", () => {
        closeSettings();
    });

    if (refs.settingsNav) {
        refs.settingsNav.addEventListener("click", (event) => {
            if (!(event.target instanceof Element)) {
                return;
            }
            const navItem = event.target.closest(".settings-nav-item");
            if (!navItem) {
                return;
            }
            const section = navItem.dataset.settingsSection;
            setActiveSettingsSection(section, true);
            persistSettingsToStorage();
        });
    }

    if (refs.saveSettingsBtn) {
        refs.saveSettingsBtn.addEventListener("click", () => {
            saveAllSettings();
        });
    }

    if (refs.resetSettingsBtn) {
        refs.resetSettingsBtn.addEventListener("click", () => {
            resetAllSettings();
            void refreshDeviceList({ requestPermission: false });
        });
    }

    const syncSettingsOnly = () => {
        syncSettingsFromUi();
    };

    const paramControls = [
        refs.historyRounds,
        refs.maxLength,
        refs.maxNewTokens,
        refs.topK,
        refs.temperature,
        refs.topP,
        refs.repetitionPenalty,
        refs.doSample,
    ];

    for (const control of paramControls) {
        if (!control) {
            continue;
        }
        control.addEventListener("change", syncSettingsOnly);
    }

    // Avoid full UI rerender while dragging sliders or typing param inputs.
    // We only mirror params into state during input, then run full sync on change.
    const paramControlsInput = [
        refs.historyRounds,
        refs.maxLength,
        refs.maxNewTokens,
        refs.topK,
        refs.temperature,
        refs.topP,
        refs.repetitionPenalty,
    ];

    for (const control of paramControlsInput) {
        if (!control) {
            continue;
        }
        control.addEventListener("input", syncSettingsParamsFromControls);
    }

    const settingsControls = [
        refs.themePreset,
        refs.displayChatUiMode,
        refs.displayDensity,
        refs.displayAnimations,
        refs.shortcutEnterToSend,
        refs.shortcutEscClosePanels,
        refs.shortcutSlashFocusInput,
        refs.languageSelect,
        refs.timeFormatSelect,
        refs.timezoneSelect,
        refs.statusIdleSeconds,
        refs.toolVoiceInput,
        refs.toolConversationHistory,
        refs.toolMessageActions,
        refs.toolHealthMonitor,
        refs.avMicDevice,
        refs.avSpeakerDevice,
        refs.avCameraDevice,
        refs.avEchoCancellation,
        refs.avNoiseSuppression,
        refs.avAutoGainControl,
    ];

    for (const control of settingsControls) {
        if (!control) {
            continue;
        }
        control.addEventListener("change", syncSettingsOnly);
    }

    if (refs.profileUserName) {
        refs.profileUserName.addEventListener("input", syncSettingsOnly);
        refs.profileUserName.addEventListener("change", syncSettingsOnly);
    }

    if (refs.profileAvatarUploadBtn && refs.profileAvatarFile) {
        refs.profileAvatarUploadBtn.addEventListener("click", () => {
            refs.profileAvatarFile.click();
        });
    }

    if (refs.profileAvatarFile) {
        refs.profileAvatarFile.addEventListener("change", () => {
            void handleProfileAvatarFileChange();
        });
    }

    if (refs.profileAvatarClearBtn) {
        refs.profileAvatarClearBtn.addEventListener("click", () => {
            if (!state.settings) {
                state.settings = getDefaultSettings();
            }
            if (!isObjectLike(state.settings.profile)) {
                state.settings.profile = cloneJson(getDefaultSettings().profile);
            }

            state.settings.profile.userAvatarDataUrl = "";
            const profile = getUserProfile();
            renderProfileAvatarPreview(profile.userName, profile.userAvatarDataUrl);
            if (state.conversations.length > 0 && !state.isSending) {
                renderActiveConversation();
            }
            showToast(TXT.toastProfileAvatarCleared);
        });
    }

    if (refs.displayFontScale) {
        refs.displayFontScale.addEventListener("input", () => {
            updateDisplayScaleLabel();
            syncSettingsFromUi();
        });
    }

    if (refs.avMicGain) {
        refs.avMicGain.addEventListener("input", () => {
            updateMicGainLabel();
            syncSettingsFromUi();
        });
    }

    if (refs.themePreset) {
        refs.themePreset.addEventListener("change", () => {
            syncSettingsFromUi();
            applyThemePreset(refs.themePreset.value);
        });
    }

    if (refs.displayDensity || refs.displayAnimations || refs.displayFontScale || refs.displayChatUiMode) {
        const applyDisplay = () => {
            syncSettingsFromUi();
            applyDisplayState(state.settings.uiState);
            if (state.conversations.length > 0 && !state.isSending) {
                renderActiveConversation();
            }
        };
        if (refs.displayChatUiMode) {
            refs.displayChatUiMode.addEventListener("change", applyDisplay);
        }
        if (refs.displayDensity) {
            refs.displayDensity.addEventListener("change", applyDisplay);
        }
        if (refs.displayAnimations) {
            refs.displayAnimations.addEventListener("change", applyDisplay);
        }
        if (refs.displayFontScale) {
            refs.displayFontScale.addEventListener("change", applyDisplay);
        }
    }

    if (refs.timeFormatSelect) {
        refs.timeFormatSelect.addEventListener("change", () => {
            syncSettingsFromUi();
            updateTimePreview();
        });
    }

    if (refs.timezoneSelect) {
        refs.timezoneSelect.addEventListener("change", () => {
            syncSettingsFromUi();
            updateTimePreview();
        });
    }

    if (refs.avSpeakerDevice) {
        refs.avSpeakerDevice.addEventListener("change", () => {
            syncSettingsFromUi();
            void applySpeakerDeviceToPlayback();
        });
    }

    if (refs.refreshDevicesBtn) {
        refs.refreshDevicesBtn.addEventListener("click", () => {
            void refreshDeviceList({ requestPermission: true });
        });
    }

    if (refs.startMicMonitorBtn) {
        refs.startMicMonitorBtn.addEventListener("click", () => {
            void startMicMonitor();
        });
    }

    if (refs.stopMicMonitorBtn) {
        refs.stopMicMonitorBtn.addEventListener("click", () => {
            stopMicMonitor();
            setAudioVideoStatus("麥克風監測已停止");
        });
    }

    if (refs.recordMicTestBtn) {
        refs.recordMicTestBtn.addEventListener("click", () => {
            void recordMicSample();
        });
    }

    if (refs.startCameraTestBtn) {
        refs.startCameraTestBtn.addEventListener("click", () => {
            void startCameraPreview();
        });
    }

    if (refs.stopCameraTestBtn) {
        refs.stopCameraTestBtn.addEventListener("click", () => {
            stopCameraPreview();
            setAudioVideoStatus("攝影機預覽已停止");
        });
    }

    refs.promptInput.addEventListener("keydown", (event) => {
        const enterToSendEnabled = !state.settings || state.settings.uiState.shortcutEnterToSend !== false;
        if (!enterToSendEnabled) {
            return;
        }

        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void sendPrompt();
        }
    });

    refs.promptInput.addEventListener("input", autoResizeTextarea);

    if (refs.jumpToLatestBtn) {
        refs.jumpToLatestBtn.addEventListener("click", () => {
            scrollChatToBottom({ smooth: true });
            refs.promptInput.focus();
        });
    }

    const defaultPromptPlaceholder = refs.promptInput.placeholder;

    // Whisper Audio Recording
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    if (refs.micBtn) {
        refs.micBtn.addEventListener("click", async () => {
            if (state.isSending) {
                if (currentAbortController) {
                    currentAbortController.abort();
                }
                return;
            }

            const capabilities = bootstrap.capabilities || {};
            const voiceEnabled = (!state.settings || state.settings.featureToggles.voiceInput !== false) && capabilities.voiceTranscription !== false;
            if (!voiceEnabled) {
                showToast(capabilities.voiceTranscription === false ? "語音轉錄後端未啟用" : "語音輸入已在註冊工具中停用");
                return;
            }

            if (isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                refs.micBtn.style.color = "";
                showToast("語音輸入結束，正在轉換...");
                return;
            }

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                refs.promptInput.style.color = "transparent";
                refs.promptInput.placeholder = "";
                refs.promptInput.readOnly = true;
                
                refs.voiceVisualizer.style.display = "block";
                refs.voiceVisualizer.width = refs.promptInput.offsetWidth || 300;
                refs.voiceVisualizer.height = 40;
                
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const analyser = audioCtx.createAnalyser();
                analyser.fftSize = 256;
                const source = audioCtx.createMediaStreamSource(stream);
                source.connect(analyser);
                const bufferLength = analyser.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                const canvasCtx = refs.voiceVisualizer.getContext("2d");
                let visualizerReqId;

                const historyLength = 180;
                const amplitudeHistory = new Array(historyLength).fill(0.01);
                
                let lastDrawTime = 0;
                const drawInterval = 40; // ms between updates to slow down speed

                isRecording = true;
                refs.micBtn.style.color = "red";
                showToast("開始錄音，再次點擊結束");

                const cPink = getComputedStyle(document.documentElement).getPropertyValue('--c-pink').trim() || '#ff69b4';
                const cCyan = getComputedStyle(document.documentElement).getPropertyValue('--c-cyan').trim() || '#00d4ff';

                const drawVisualizer = (timestamp) => {
                    if (!isRecording) return;
                    visualizerReqId = requestAnimationFrame(drawVisualizer);
                    
                    if (timestamp - lastDrawTime < drawInterval) return;
                    lastDrawTime = timestamp;
                    
                    analyser.getByteTimeDomainData(dataArray);
                    
                    let maxAmp = 0;
                    for (let i = 0; i < bufferLength; i++) {
                        const v = Math.abs((dataArray[i] / 128.0) - 1.0);
                        if (v > maxAmp) maxAmp = v;
                    }
                    
                    const spike = maxAmp * (0.8 + Math.random() * 0.4);
                    amplitudeHistory.shift();
                    amplitudeHistory.push(spike);
                    
                    const width = refs.voiceVisualizer.width;
                    const height = refs.voiceVisualizer.height;
                    
                    canvasCtx.clearRect(0, 0, width, height);
                    
                    const gradient = canvasCtx.createLinearGradient(0, 0, width, 0);
                    gradient.addColorStop(0, "transparent");
                    gradient.addColorStop(0.15, cCyan);
                    gradient.addColorStop(0.85, cPink);
                    gradient.addColorStop(1, "transparent");
                    
                    canvasCtx.fillStyle = gradient;
                    canvasCtx.beginPath();
                    
                    const sliceWidth = width / (historyLength - 1);
                    
                    canvasCtx.moveTo(0, height / 2);
                    for (let i = 0; i < historyLength; i++) {
                        const x = i * sliceWidth;
                        let v = amplitudeHistory[i];
                        if (i > historyLength - 8) {
                            v *= (historyLength - i) / 8; // Taper off on the right
                        }
                        if (i < 8) {
                            v *= i / 8; // Taper off on the left
                        }
                        const y = (height / 2) - (v * height / 2 * 0.85) - 0.5;
                        canvasCtx.lineTo(x, y);
                    }
                    for (let i = historyLength - 1; i >= 0; i--) {
                        const x = i * sliceWidth;
                        let v = amplitudeHistory[i];
                        if (i > historyLength - 8) {
                            v *= (historyLength - i) / 8;
                        }
                        if (i < 8) {
                            v *= i / 8;
                        }
                        const y = (height / 2) + (v * height / 2 * 0.85) + 0.5;
                        canvasCtx.lineTo(x, y);
                    }
                    canvasCtx.closePath();
                    canvasCtx.fill();
                };
                visualizerReqId = requestAnimationFrame(drawVisualizer);

                mediaRecorder.addEventListener("dataavailable", (event) => {
                    audioChunks.push(event.data);
                });

                mediaRecorder.addEventListener("stop", async () => {
                    stream.getTracks().forEach((track) => track.stop());
                    if (visualizerReqId) cancelAnimationFrame(visualizerReqId);
                    if (audioCtx.state !== "closed") audioCtx.close();
                    
                    refs.voiceVisualizer.style.display = "none";
                    refs.promptInput.style.color = "";
                    refs.promptInput.readOnly = false;
                    
                    const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
                    const formData = new FormData();
                    formData.append("audio", audioBlob, "recording.webm");

                    refs.promptInput.placeholder = "正在轉換語音...";
                    refs.promptInput.disabled = true;

                    try {
                        const response = await fetch("/api/transcribe", {
                            method: "POST",
                            body: formData,
                        });

                        if (!response.ok) {
                            throw new Error("語音轉換失敗");
                        }

                        const data = await response.json();
                        if (data.text) {
                            refs.promptInput.value = (refs.promptInput.value + " " + data.text).trim();
                            autoResizeTextarea();
                            showToast("語音轉換成功");
                        }
                    } catch (error) {
                        console.error(error);
                        showToast(error.message || "語音轉換失敗");
                    } finally {
                        refs.promptInput.placeholder = defaultPromptPlaceholder;
                        refs.promptInput.disabled = false;
                        refs.promptInput.focus();
                    }
                });

                mediaRecorder.start();
            } catch (err) {
                console.error(err);
                showToast("無法存取麥克風設備");
            }
        });
    }

    refs.chatList.addEventListener("click", (event) => {
        if (!(event.target instanceof Element)) {
            return;
        }
        const actionBtn = event.target.closest(".message-action-btn");
        if (!actionBtn) {
            return;
        }

        const messageRow = actionBtn.closest(".message-row");
        if (!messageRow) {
            return;
        }

        const action = String(actionBtn.dataset.action || "").trim();
        const messageIndex = Number(messageRow.dataset.messageIndex);
        if (!action || !Number.isInteger(messageIndex) || messageIndex < 0) {
            return;
        }

        event.preventDefault();
        void handleMessageAction(action, messageIndex);
    });

    refs.chatList.addEventListener("scroll", () => {
        updateJumpToLatestVisibility();
    }, { passive: true });

    window.addEventListener("resize", () => {
        autoResizeTextarea();
        updateJumpToLatestVisibility();
    });

    refs.temperature.addEventListener("input", updateRangeLabels);
    refs.topP.addEventListener("input", updateRangeLabels);
    refs.repetitionPenalty.addEventListener("input", updateRangeLabels);

    document.addEventListener("keydown", (event) => {
        const escapeEnabled = !state.settings || state.settings.uiState.shortcutEscClosePanels !== false;
        if (event.key === "Escape" && escapeEnabled) {
            if (refs.settingsModal.classList.contains("open")) {
                closeSettings();
                return;
            }
            if (refs.appShell.classList.contains("right-sidebar-expanded")) {
                refs.appShell.classList.remove("right-sidebar-expanded");
                return;
            }
            if (!state.sidebarCollapsed) {
                closeSidebar();
            }
            return;
        }

        const slashFocusEnabled = !state.settings || state.settings.uiState.shortcutSlashFocusInput !== false;
        if (!slashFocusEnabled || event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey) {
            return;
        }

        const activeEl = document.activeElement;
        const tagName = activeEl ? activeEl.tagName : "";
        const isTypingField = tagName === "INPUT" || tagName === "TEXTAREA" || (activeEl && activeEl.isContentEditable);
        if (isTypingField) {
            return;
        }

        event.preventDefault();
        refs.promptInput.focus();
    });
}

function init() {
    setStatus("offline");
    closeSettings();
    bootstrapSettings();
    loadConversations();
    restoreSidebarState();
    refs.appShell.classList.add("right-sidebar-expanded");
    bindEvents();
    void refreshDeviceList({ requestPermission: false });
    renderConversationList();
    renderActiveConversation();
    autoResizeTextarea();
    if (state.settings && state.settings.featureToggles.healthMonitor) {
        void refreshBackendHealth();
    }
    refs.promptInput.focus();
}

init();
