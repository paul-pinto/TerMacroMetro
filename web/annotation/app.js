const state = {
  key: sessionStorage.getItem("annotationKey") || "",
  annotator: localStorage.getItem("annotationAnnotator")
    || "Jhonny Paul Pinto Phillips",
  current: null,
  selectedLabel: null,
};

const authView = document.querySelector("#authView");
const appView = document.querySelector("#appView");
const apiKeyInput = document.querySelector("#apiKey");
const annotatorInput = document.querySelector("#annotator");
const loginButton = document.querySelector("#loginButton");
const logoutButton = document.querySelector("#logoutButton");
const loginMessage = document.querySelector("#loginMessage");

const totalValue = document.querySelector("#totalValue");
const annotatedValue = document.querySelector("#annotatedValue");
const pendingValue = document.querySelector("#pendingValue");
const doubtfulValue = document.querySelector("#doubtfulValue");
const progressBar = document.querySelector("#progressBar");

const itemView = document.querySelector("#itemView");
const emptyView = document.querySelector("#emptyView");
const articleMeta = document.querySelector("#articleMeta");
const articleTitle = document.querySelector("#articleTitle");
const articleText = document.querySelector("#articleText");
const commentInput = document.querySelector("#comment");
const doubtInput = document.querySelector("#doubt");
const saveButton = document.querySelector("#saveButton");
const skipButton = document.querySelector("#skipButton");
const openSourceButton = document.querySelector("#openSourceButton");
const message = document.querySelector("#message");

const labelButtons = [
  ...document.querySelectorAll(".label-button"),
];

apiKeyInput.value = state.key;
annotatorInput.value = state.annotator;

function apiUrl(path) {
  const basePath = window.location.pathname
    .replace(/\/annotation\/?$/, "")
    .replace(/\/$/, "");

  return `${basePath}/api/annotation${path}`;
}

async function apiRequest(path, options = {}) {
  const headers = {
    "X-Annotation-Key": state.key,
    ...(options.headers || {}),
  };

  if (options.body) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(
    apiUrl(path),
    {
      ...options,
      headers,
    },
  );

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      payload.detail || `Error HTTP ${response.status}`,
    );
  }

  return payload;
}

function setMessage(text = "", error = false) {
  message.textContent = text;
  message.classList.toggle("error", error);
}

function setLoginMessage(text = "", error = false) {
  loginMessage.textContent = text;
  loginMessage.classList.toggle("error", error);
}

function selectLabel(label) {
  state.selectedLabel = label;

  for (const button of labelButtons) {
    button.classList.toggle(
      "selected",
      button.dataset.label === label,
    );
  }

  saveButton.disabled = !label;
}

function renderMeta(item) {
  const values = [
    item.fuente,
    item.fecha,
    item.tema,
    item.indicadores,
  ].filter(Boolean);

  articleMeta.innerHTML = "";

  for (const value of values) {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = String(value);
    articleMeta.appendChild(pill);
  }
}

function renderItem(item) {
  state.current = item;

  articleTitle.textContent = item.titulo || "Sin título";
  articleText.textContent = item.texto || "";
  renderMeta(item);

  commentInput.value = item.comentario || "";
  doubtInput.checked = Number(item.duda || 0) === 1;

  selectLabel(
    ["positivo", "neutral", "negativo"].includes(
      item.sentimiento,
    )
      ? item.sentimiento
      : null,
  );

  itemView.classList.remove("hidden");
  emptyView.classList.add("hidden");
  setMessage("");
  window.scrollTo({
    top: 0,
    behavior: "smooth",
  });
}

async function loadStatus() {
  const status = await apiRequest("/status");

  totalValue.textContent = status.total;
  annotatedValue.textContent = status.annotated;
  pendingValue.textContent = status.pending;
  doubtfulValue.textContent = status.doubtful;
  progressBar.style.width = `${status.progress}%`;

  return status;
}

async function loadNext() {
  setMessage("Cargando siguiente noticia…");

  const payload = await apiRequest(
    "/items?state=pending&limit=1",
  );

  if (!payload.items.length) {
    state.current = null;
    itemView.classList.add("hidden");
    emptyView.classList.remove("hidden");
    setMessage("");
    return;
  }

  renderItem(payload.items[0]);
}

async function enterApp() {
  await loadStatus();

  authView.classList.add("hidden");
  appView.classList.remove("hidden");

  await loadNext();
}

async function login() {
  state.key = apiKeyInput.value.trim();
  state.annotator = annotatorInput.value.trim();

  if (!state.key || !state.annotator) {
    setLoginMessage(
      "Ingresá la clave y el nombre del anotador.",
      true,
    );
    return;
  }

  loginButton.disabled = true;
  setLoginMessage("Validando…");

  try {
    sessionStorage.setItem(
      "annotationKey",
      state.key,
    );

    localStorage.setItem(
      "annotationAnnotator",
      state.annotator,
    );

    await enterApp();
    setLoginMessage("");
  } catch (error) {
    sessionStorage.removeItem("annotationKey");
    setLoginMessage(error.message, true);
  } finally {
    loginButton.disabled = false;
  }
}

async function saveCurrent() {
  if (!state.current || !state.selectedLabel) {
    return;
  }

  saveButton.disabled = true;
  setMessage("Guardando…");

  try {
    await apiRequest(
      `/items/${encodeURIComponent(
        state.current.annotation_id,
      )}`,
      {
        method: "PUT",
        body: JSON.stringify({
          sentimiento: state.selectedLabel,
          anotador: state.annotator,
          duda: doubtInput.checked ? 1 : 0,
          comentario: commentInput.value.trim(),
        }),
      },
    );

    await loadStatus();
    await loadNext();
  } catch (error) {
    setMessage(error.message, true);
    saveButton.disabled = false;
  }
}

async function skipCurrent() {
  if (!state.current) {
    return;
  }

  const currentId = state.current.annotation_id;

  const payload = await apiRequest(
    "/items?state=pending&limit=2",
  );

  const next = payload.items.find(
    (item) => item.annotation_id !== currentId,
  );

  if (next) {
    renderItem(next);
  } else {
    setMessage(
      "No existe otra noticia pendiente para saltar.",
      true,
    );
  }
}

function logout() {
  sessionStorage.removeItem("annotationKey");

  state.key = "";
  state.current = null;
  state.selectedLabel = null;

  appView.classList.add("hidden");
  authView.classList.remove("hidden");

  apiKeyInput.value = "";
}

loginButton.addEventListener("click", login);
logoutButton.addEventListener("click", logout);
saveButton.addEventListener("click", saveCurrent);
skipButton.addEventListener("click", skipCurrent);

openSourceButton.addEventListener("click", () => {
  const url = state.current?.url;

  if (url) {
    window.open(
      url,
      "_blank",
      "noopener,noreferrer",
    );
  }
});

for (const button of labelButtons) {
  button.addEventListener("click", () => {
    selectLabel(button.dataset.label);
  });
}

document.addEventListener("keydown", (event) => {
  if (appView.classList.contains("hidden")) {
    return;
  }

  if (
    event.target instanceof HTMLInputElement
    || event.target instanceof HTMLTextAreaElement
  ) {
    return;
  }

  if (event.key === "1") {
    selectLabel("positivo");
  }

  if (event.key === "2") {
    selectLabel("neutral");
  }

  if (event.key === "3") {
    selectLabel("negativo");
  }

  if (
    event.key === "Enter"
    && state.selectedLabel
  ) {
    saveCurrent();
  }
});

if (state.key) {
  enterApp().catch(() => {
    sessionStorage.removeItem("annotationKey");
    authView.classList.remove("hidden");
    appView.classList.add("hidden");
  });
}
