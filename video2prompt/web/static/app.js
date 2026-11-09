"use strict";

const $ = (selector) => document.querySelector(selector);

// Model output and file names are always inserted as text, never as HTML.
function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "on") Object.entries(value).forEach(([e, f]) => node.addEventListener(e, f));
    else node.setAttribute(key, value);
  }
  node.append(...children.flat(Infinity).filter((c) => c !== null && c !== undefined));
  return node;
}

const STAGES = {
  download: "Téléchargement de la vidéo",
  cuts: "Détection des plans",
  transcribe: "Transcription de la parole",
  analyse: "Analyse des images",
  style: "Synthèse du style",
  write: "Écriture des prompts",
};

const STYLE_FIELDS = [
  ["visual_style", "Style"],
  ["color_grade", "Étalonnage"],
  ["lighting", "Lumière"],
  ["camera_language", "Caméra"],
  ["pacing", "Rythme"],
  ["mood", "Ambiance"],
];

const state = { mode: "file", file: null, config: null, jobId: null, timer: null, objectUrl: null, target: null };

function show(view) {
  for (const name of ["setup", "progress", "result"]) $(`#view-${name}`).hidden = name !== view;
  window.scrollTo(0, 0);
}

function showError(selector, message) {
  const node = $(selector);
  node.textContent = message || "";
  node.hidden = !message;
}

/* ---------- Setup ---------- */

async function loadConfig() {
  const engine = $("#engine");
  try {
    const response = await fetch("/api/config");
    state.config = await response.json();
  } catch {
    engine.textContent = "Serveur injoignable";
    engine.className = "engine bad";
    return;
  }
  const { targets, ollama, claude_available: claude } = state.config;

  $("#targets").replaceChildren(
    ...targets.map((target) =>
      h("label", { class: "chip" },
        (() => {
          const box = h("input", { type: "checkbox", value: target.key, class: "sr-only" });
          box.checked = target.key !== "generic";
          return box;
        })(),
        target.label)
    )
  );

  const select = $("#model");
  select.replaceChildren(...ollama.models.map((name) => h("option", { value: name }, name)));
  if (ollama.models.includes(ollama.default)) select.value = ollama.default;

  if (ollama.ok && ollama.models.length) {
    engine.textContent = "Ollama connecté";
    engine.className = "engine ok";
  } else {
    engine.textContent = ollama.ok ? "Aucun modèle de vision" : "Ollama arrêté";
    engine.className = "engine bad";
  }

  const claudeRadio = $("#be-claude");
  claudeRadio.disabled = !claude;
  $("#claude-hint").hidden = claude;
  document.querySelectorAll('input[name="backend"]').forEach((radio) =>
    radio.addEventListener("change", () => {
      $("#model-field").hidden = claudeRadio.checked;
    })
  );
}

function setMode(mode) {
  state.mode = mode;
  $("#tab-file").setAttribute("aria-selected", String(mode === "file"));
  $("#tab-url").setAttribute("aria-selected", String(mode === "url"));
  $("#drop").hidden = mode !== "file";
  $("#url-pane").hidden = mode !== "url";
  if (mode === "url") $("#url").focus();
}

function formatSize(bytes) {
  return bytes > 1e9 ? `${(bytes / 1e9).toFixed(1)} Go` : `${Math.max(1, Math.round(bytes / 1e6))} Mo`;
}

function setFile(file) {
  state.file = file;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = file ? URL.createObjectURL(file) : null;
  const video = $("#video-preview");
  if (file) video.src = state.objectUrl;
  else video.removeAttribute("src");
  $("#drop-empty").hidden = Boolean(file);
  $("#drop-filled").hidden = !file;
  $("#file-name").textContent = file ? `${file.name} · ${formatSize(file.size)}` : "";
  if (!file) $("#file").value = "";
  showError("#form-error", "");
}

function errorText(payload) {
  if (Array.isArray(payload?.detail)) return "Un réglage est hors limites.";
  return payload?.detail || "L'envoi a échoué.";
}

async function submit(event) {
  event.preventDefault();
  const targets = [...document.querySelectorAll("#targets input:checked")].map((box) => box.value);
  const url = $("#url").value.trim();

  if (state.mode === "file" && !state.file) return showError("#form-error", "Choisis d'abord une vidéo.");
  if (state.mode === "url" && !url) return showError("#form-error", "Colle d'abord le lien de la vidéo.");
  if (!targets.length) return showError("#form-error", "Choisis au moins un générateur.");
  showError("#form-error", "");

  const backend = document.querySelector('input[name="backend"]:checked').value;
  const data = new FormData();
  if (state.mode === "file") data.append("file", state.file);
  else data.append("url", url);
  data.append("targets", targets.join(","));
  data.append("backend", backend);
  if (backend === "ollama") data.append("model", $("#model").value);
  data.append("variations", $("#variations").value);
  data.append("max_shots", $("#max-shots").value);
  data.append("frames_per_shot", $("#frames").value);
  data.append("scene_threshold", $("#threshold").value);
  data.append("transcribe", String($("#transcribe").checked));
  data.append("whisper_model", $("#whisper").value);

  const button = $("#submit");
  button.disabled = true;
  button.textContent = state.mode === "file" ? "Envoi de la vidéo…" : "Démarrage…";
  try {
    const response = await fetch("/api/jobs", { method: "POST", body: data });
    const payload = await response.json().catch(() => null);
    if (!response.ok) return showError("#form-error", errorText(payload));
    location.hash = `job=${payload.id}`;
  } catch {
    showError("#form-error", "Le serveur ne répond pas.");
  } finally {
    button.disabled = false;
    button.textContent = "Générer les prompts";
  }
}

/* ---------- Progress ---------- */

function frameUrl(jobId, name) {
  return `/api/jobs/${jobId}/frames/${encodeURIComponent(name.split("/").pop())}`;
}

function stagesFor(job) {
  return Object.keys(STAGES).filter((key) =>
    (key !== "download" || job.from_url) && (key !== "transcribe" || job.transcribes)
  );
}

function renderProgress(job) {
  $("#progress-name").textContent = job.name;
  const stages = stagesFor(job);
  const current = stages.indexOf(job.stage);

  $("#stages").replaceChildren(
    ...stages.map((key, index) =>
      h("li", { class: index < current ? "done" : index === current ? "active" : "" }, STAGES[key])
    )
  );

  const percent = job.total ? Math.round((job.done / job.total) * 100) : 0;
  $("#bar-fill").style.width = `${percent}%`;
  $("#bar").setAttribute("aria-valuenow", String(percent));
  $("#progress-count").textContent = job.total > 1 ? `${job.done} / ${job.total}` : "";

  const failed = job.status === "error" || job.status === "cancelled";
  const titles = {
    queued: "En file d'attente…",
    running: STAGES[job.stage] || "Préparation…",
    error: "L'analyse a échoué",
    cancelled: "Analyse annulée",
  };
  $("#progress-title").textContent = titles[job.status] || "";
  showError("#progress-error", job.error);
  $("#cancel").hidden = failed;
  $("#retry").hidden = !failed;

  if (job.preview) {
    const image = $("#preview");
    const source = frameUrl(job.id, job.preview);
    if (image.getAttribute("src") !== source) image.src = source;
    image.hidden = false;
  }

  const log = $("#log");
  const text = job.logs.join("\n");
  if (log.textContent !== text) {
    log.textContent = text;
    log.scrollTop = log.scrollHeight;
  }
}

async function poll() {
  const id = state.jobId;
  try {
    const response = await fetch(`/api/jobs/${id}`);
    if (response.status === 404) return goHome();
    const job = await response.json();
    if (id !== state.jobId) return;
    if (job.status === "done") return renderResult(job);
    renderProgress(job);
    if (job.status === "queued" || job.status === "running") state.timer = setTimeout(poll, 700);
  } catch {
    state.timer = setTimeout(poll, 2000);
  }
}

function openJob(id) {
  clearTimeout(state.timer);
  state.jobId = id;
  $("#preview").hidden = true;
  $("#preview").removeAttribute("src");
  $("#bar-fill").style.width = "0";
  show("progress");
  poll();
}

function goHome() {
  clearTimeout(state.timer);
  state.jobId = null;
  if (location.hash) history.pushState("", document.title, location.pathname);
  show("setup");
}

/* ---------- Result ---------- */

function copyButton(text) {
  const button = h("button", { class: "copy", type: "button" }, "Copier");
  button.addEventListener("click", async () => {
    let label = "Copié";
    try {
      await navigator.clipboard.writeText(text);
      button.classList.add("copied");
    } catch {
      label = "Échec";
    }
    button.textContent = label;
    setTimeout(() => {
      button.textContent = "Copier";
      button.classList.remove("copied");
    }, 1400);
  });
  return button;
}

function promptNode(text, small = false) {
  const parts = text.split(/(\[[A-Z][A-Z_ 0-9]*\])/g).map((part, index) =>
    index % 2 ? h("mark", {}, part) : part
  );
  return h("p", { class: small ? "prompt small" : "prompt" }, parts);
}

function promptCard(text, { hero = false, title = null, small = false, before = null } = {}) {
  const words = text.trim().split(/\s+/).length;
  return h("div", { class: hero ? "card hero" : "card" },
    before,
    title ? h("h4", {}, title) : null,
    promptNode(text, small),
    h("div", { class: "card-foot" }, h("span", { class: "words" }, `${words} mots`), copyButton(text))
  );
}

function block(title, ...children) {
  return h("div", { class: "block" }, h("h3", {}, title), children);
}

function shotCard(job, shot, key) {
  const strip = h("div", { class: "strip" },
    shot.frames.map((frame, index) => {
      const url = frameUrl(job.id, frame);
      return h("a", { href: url, download: frame.split("/").pop(), title: index ? "Télécharger l'image" : "Télécharger l'image de départ" },
        h("img", { src: url, alt: `Plan ${shot.index}, image ${index + 1}` }),
        index === 0 ? h("span", { class: "start" }, "Départ") : null
      );
    })
  );
  const head = h("div", { class: "shot-head" },
    h("strong", {}, `Plan ${shot.index}`),
    h("span", { class: "shot-time" }, `${shot.start.toFixed(1)} → ${shot.end.toFixed(1)} s`)
  );
  return promptCard(shot.prompts[key], { before: [head, strip] });
}

function renderTarget(job, key) {
  state.target = key;
  document.querySelectorAll("#result-tabs button").forEach((tab) =>
    tab.setAttribute("aria-selected", String(tab.dataset.key === key))
  );
  const { masters, negative_prompts: negatives, shots } = job.result;
  const master = masters[key];
  $("#result-main").replaceChildren(
    block("Prompt complet", promptCard(master.master_prompt, { hero: true })),
    block("Modèle réutilisable", promptCard(master.template_prompt)),
    master.variations.length
      ? block("Variations", master.variations.map((v) => promptCard(v.prompt, { title: v.concept })))
      : "",
    block("Prompt négatif", promptCard(negatives[key], { small: true })),
    block("Plan par plan", shots.map((shot) => shotCard(job, shot, key)))
  );
}

function renderResult(job) {
  const { result } = job;
  const { video, shots, style } = result;
  $("#result-name").textContent = job.name;
  $("#result-facts").textContent =
    `${video.width}×${video.height} · ${video.duration.toFixed(1)} s · ${shots.length} plan(s) · ${result.model}`;
  $("#dl-md").href = `/api/jobs/${job.id}/download/prompts.md`;
  $("#dl-json").href = `/api/jobs/${job.id}/download/result.json`;

  const notices = [];
  if (result.shots_skipped) notices.push(`${result.shots_skipped} plan(s) courts n'ont pas été analysés.`);
  if (result.transcription_error) notices.push("La transcription a échoué : les dialogues ne figurent pas dans ces prompts.");
  $("#result-notice").textContent = notices.join(" ");
  $("#result-notice").hidden = !notices.length;

  const labels = Object.fromEntries(state.config.targets.map((t) => [t.key, t.label]));
  const keys = Object.keys(result.masters);
  $("#result-tabs").replaceChildren(
    ...keys.map((key) => {
      const tab = h("button", { type: "button", role: "tab" }, labels[key] || key);
      tab.dataset.key = key;
      tab.addEventListener("click", () => renderTarget(job, key));
      return tab;
    })
  );

  $("#style-panel").replaceChildren(
    h("h3", {}, "Style de la vidéo"),
    h("p", {}, style.summary),
    h("dl", {}, STYLE_FIELDS.map(([field, label]) => [h("div", {}, h("dt", {}, label), h("dd", {}, style[field]))]))
  );

  renderTarget(job, keys[0]);
  show("result");
}

/* ---------- Wiring ---------- */

function route() {
  const match = location.hash.match(/job=([a-f0-9]+)/);
  if (match) openJob(match[1]);
  else goHome();
}

function init() {
  $("#tab-file").addEventListener("click", () => setMode("file"));
  $("#tab-url").addEventListener("click", () => setMode("url"));
  $("#file").addEventListener("change", (e) => setFile(e.target.files[0] || null));
  $("#file-clear").addEventListener("click", (e) => {
    e.preventDefault();
    setFile(null);
  });
  $("#video-preview").addEventListener("click", (e) => e.preventDefault());

  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((name) =>
    drop.addEventListener(name, (e) => {
      e.preventDefault();
      drop.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    drop.addEventListener(name, (e) => {
      e.preventDefault();
      drop.classList.remove("over");
    })
  );
  drop.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });
  // A file dropped beside the zone must not navigate away from the page.
  ["dragover", "drop"].forEach((name) => window.addEventListener(name, (e) => e.preventDefault()));

  $("#threshold").addEventListener("input", (e) => {
    $("#threshold-value").textContent = Number(e.target.value).toFixed(2);
  });
  $("#form").addEventListener("submit", submit);
  $("#home").addEventListener("click", goHome);
  $("#again").addEventListener("click", goHome);
  $("#retry").addEventListener("click", goHome);
  $("#cancel").addEventListener("click", () => {
    if (!state.jobId) return;
    $("#progress-title").textContent = "Annulation…";
    fetch(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
  });
  window.addEventListener("hashchange", route);

  loadConfig().then(route);
}

init();
