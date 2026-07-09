// Lógica de la interfaz web del clasificador de sentimiento.
// Se comunica con la API FastAPI mediante fetch (mismo origen).

const $ = (sel) => document.querySelector(sel);

const CONFIG_CLASES = {
  positivo: { emoji: "😊", clase: "pos", bg: "bg-pos" },
  negativo: { emoji: "😠", clase: "neg", bg: "bg-neg" },
  neutral:  { emoji: "😐", clase: "neu", bg: "bg-neu" },
};

const EJEMPLOS = [
  "El celular es excelente, lo recomiendo totalmente.",
  "Pésima calidad, la laptop se dañó al segundo día.",
  "El producto está bien, cumple su función sin más.",
  "No compren esto, es una pérdida de dinero.",
  "Me encantó, llegó rápido y funciona perfecto.",
];

// --- Predicción -------------------------------------------------------------

async function analizar() {
  const texto = $("#entrada").value.trim();
  if (!texto) {
    mostrarError("Escribe una reseña primero.");
    return;
  }

  ocultar("#resultado");
  ocultar("#error");
  mostrar("#cargando");

  try {
    const resp = await fetch("/predecir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `Error ${resp.status}`);
    }
    const data = await resp.json();
    mostrarResultado(data);
  } catch (e) {
    mostrarError(e.message);
  } finally {
    ocultar("#cargando");
  }
}

function mostrarResultado(data) {
  const conf = CONFIG_CLASES[data.sentimiento] || CONFIG_CLASES.neutral;

  $("#emoji-resultado").textContent = conf.emoji;
  const st = $("#sentimiento-texto");
  st.textContent = data.sentimiento;
  st.className = "sentimiento " + conf.clase;
  $("#confianza-texto").textContent =
    `Confianza: ${(data.confianza * 100).toFixed(1)}%`;

  // Barras de probabilidad ordenadas de mayor a menor.
  const barras = $("#barras");
  barras.innerHTML = "";
  const entradas = Object.entries(data.probabilidades)
    .sort((a, b) => b[1] - a[1]);

  for (const [clase, prob] of entradas) {
    const c = CONFIG_CLASES[clase] || CONFIG_CLASES.neutral;
    const fila = document.createElement("div");
    fila.className = "barra-fila";
    fila.innerHTML = `
      <div class="barra-etiqueta">${clase}</div>
      <div class="barra-pista">
        <div class="barra-relleno ${c.bg}" style="width:0%"></div>
      </div>
      <div class="barra-valor">${(prob * 100).toFixed(1)}%</div>`;
    barras.appendChild(fila);
    // Animación: fijamos el ancho tras insertar.
    requestAnimationFrame(() => {
      fila.querySelector(".barra-relleno").style.width = `${prob * 100}%`;
    });
  }

  mostrar("#resultado");
}

// --- Métricas del modelo ----------------------------------------------------

async function cargarMetricas() {
  try {
    const resp = await fetch("/metricas");
    if (!resp.ok) throw new Error();
    const m = await resp.json();
    $("#metricas").innerHTML = `
      <div class="metrica-fila"><span>Exactitud</span><span>${(m.exactitud * 100).toFixed(1)}%</span></div>
      <div class="metrica-fila"><span>F1 macro</span><span>${(m.f1_macro * 100).toFixed(1)}%</span></div>
      <div class="metrica-fila"><span>Ejemplos entrenamiento</span><span>${m.n_entrenamiento}</span></div>
      <div class="metrica-fila"><span>Ejemplos prueba</span><span>${m.n_prueba}</span></div>`;
  } catch {
    $("#metricas").textContent = "No disponibles (entrena el modelo primero).";
  }
}

// --- Utilidades de UI -------------------------------------------------------

function mostrar(sel) { $(sel).classList.remove("oculto"); }
function ocultar(sel) { $(sel).classList.add("oculto"); }
function mostrarError(msg) {
  const e = $("#error");
  e.textContent = "⚠️ " + msg;
  mostrar("#error");
}

function crearEjemplos() {
  const cont = $("#ejemplos");
  EJEMPLOS.forEach((txt) => {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = txt.length > 42 ? txt.slice(0, 42) + "…" : txt;
    chip.title = txt;
    chip.addEventListener("click", () => {
      $("#entrada").value = txt;
      analizar();
    });
    cont.appendChild(chip);
  });
}

// --- Inicialización ---------------------------------------------------------

$("#btn-analizar").addEventListener("click", analizar);
$("#btn-limpiar").addEventListener("click", () => {
  $("#entrada").value = "";
  ocultar("#resultado");
  ocultar("#error");
});
$("#entrada").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analizar();
});

crearEjemplos();
cargarMetricas();
