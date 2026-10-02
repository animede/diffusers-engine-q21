const host = window.location.hostname || "127.0.0.1";
const API_BASE = `${window.location.protocol === "file:" ? "http:" : window.location.protocol}//${host}:8000/api`;
const API_ORIGIN = API_BASE.replace(/\/api$/, "");

const $ = (id) => document.getElementById(id);
const form = $("generateForm");
const fileInput = $("imageInput");
const previews = $("previews");
const dropzone = $("dropzone");
let selectedFiles = [];
let activeJob = null;
let turboMode = false;
let lastJob = null; // 言語切替時に完了/エラー表示を再描画するために保持

function toast(message) {
  const element = $("toast");
  element.textContent = message;
  element.classList.remove("hidden");
  clearTimeout(element.timer);
  element.timer = setTimeout(() => element.classList.add("hidden"), 5000);
}

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch (_) { /* noop */ }
    throw new Error(message);
  }
  return response.json();
}

async function refreshHealth() {
  try {
    const data = await api("/health");
    const model = data.model;
    turboMode = model.acceleration === "viggle-r128";
    $("steps").value = turboMode ? "6" : $("steps").value;
    $("steps").disabled = turboMode;
    $("steps").title = turboMode ? t("steps.turboFixed") : "";
    $("guidance").value = turboMode ? "1" : $("guidance").value;
    $("guidance").disabled = turboMode;
    $("referenceHelp").textContent = t(turboMode ? "reference.help.turbo" : "reference.help");
    $("serverStatus").className = "server-status online";
    const labels = {
      ready: t("status.model.ready"),
      loading: t("status.model.loading"),
      error: t("status.model.error"),
      not_loaded: model.downloaded ? t("status.connected") : t("status.model.missing"),
    };
    $("statusLabel").textContent = labels[model.status] || t("status.connected");
    $("loadModelButton").classList.toggle("hidden", !model.downloaded || model.status === "ready" || model.status === "loading");
    if (!activeJob && !lastJob) {
      if (!model.downloaded) $("jobInfo").textContent = t("job.needDownload");
      else if (model.status === "ready") {
        const precision = {
          nvfp4: "NVFP4 W4A4",
          fp8: "FP8 W8A8",
          bf16: "BF16",
        }[model.quantization] || model.quantization.toUpperCase();
        const acceleration = turboMode ? " · Viggle r128 / 6-step" : "";
        $("jobInfo").textContent = `${model.device} · ${precision}${acceleration} · ${model.start_profile} · ${t("job.resident")}`;
      }
      else if (model.status === "loading") $("jobInfo").textContent = t("job.loading");
      else $("jobInfo").textContent = t("job.lazy");
    }
  } catch (error) {
    $("serverStatus").className = "server-status error";
    $("statusLabel").textContent = t("status.disconnected");
    if (!activeJob && !lastJob) $("jobInfo").textContent = t("job.startApi", { api: API_BASE });
  }
}

function renderPreviews() {
  previews.replaceChildren();
  selectedFiles.forEach((file, index) => {
    const wrapper = document.createElement("div");
    wrapper.className = "preview";
    const image = document.createElement("img");
    image.src = URL.createObjectURL(file);
    image.onload = () => URL.revokeObjectURL(image.src);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.onclick = () => { selectedFiles.splice(index, 1); renderPreviews(); };
    wrapper.append(image, remove);
    previews.append(wrapper);
  });
  $("modePill").textContent = selectedFiles.length ? "IMAGE EDIT" : "TEXT TO IMAGE";
}

function addFiles(files) {
  const images = [...files].filter((file) => file.type.startsWith("image/"));
  const limit = turboMode ? 3 : 10;
  const exceedsLimit = selectedFiles.length + images.length > limit;
  selectedFiles = [...selectedFiles, ...images].slice(0, limit);
  if (images.length !== files.length) toast(t("toast.nonImage"));
  if (exceedsLimit) toast(t("toast.maxImages", { limit }));
  renderPreviews();
}

fileInput.addEventListener("change", () => { addFiles(fileInput.files); fileInput.value = ""; });
["dragenter", "dragover"].forEach((event) => dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.add("dragging"); }));
["dragleave", "drop"].forEach((event) => dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.remove("dragging"); }));
dropzone.addEventListener("drop", (event) => addFiles(event.dataTransfer.files));

$("presets").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-w]");
  if (!button) return;
  $("width").value = button.dataset.w;
  $("height").value = button.dataset.h;
  document.querySelectorAll("#presets button").forEach((item) => item.classList.toggle("active", item === button));
});

function setGenerating(active) {
  $("generateButton").disabled = active;
  $("progressOverlay").classList.toggle("hidden", !active);
}

function updateProgress(job) {
  const percent = Math.round((job.progress || 0) * 100);
  const message = translateServerMessage(job.message);
  $("progressMessage").textContent = message;
  $("progressBar").style.width = `${percent}%`;
  $("progressPercent").textContent = `${percent}%`;
  $("jobInfo").textContent = t("job.label", { id: job.id.slice(0, 8), message });
}

function renderFinishedJob() {
  if (!lastJob) return;
  if (lastJob.status === "completed") {
    $("jobInfo").textContent = t("job.done", { id: lastJob.id.slice(0, 8) });
  } else if (lastJob.status === "failed") {
    $("jobInfo").textContent = t("job.error", { message: lastJob.error || t("error.generationFailed") });
  }
}

async function pollJob(jobId) {
  while (true) {
    const job = await api(`/jobs/${jobId}`);
    updateProgress(job);
    if (job.status === "completed") {
      lastJob = job;
      const url = `${API_ORIGIN}${job.output_url}`;
      $("resultImage").src = `${url}?t=${Date.now()}`;
      $("resultImage").classList.remove("hidden");
      $("emptyState").classList.add("hidden");
      $("downloadButton").href = url;
      $("downloadButton").classList.remove("hidden");
      renderFinishedJob();
      return;
    }
    if (job.status === "failed") {
      lastJob = job;
      throw new Error(job.error || t("error.generationFailed"));
    }
    await new Promise((resolve) => setTimeout(resolve, 900));
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (activeJob) return;
  const width = Number($("width").value);
  const height = Number($("height").value);
  if (width % 16 || height % 16) return toast(t("toast.multiple16"));
  $("canvas").style.aspectRatio = `${width} / ${height}`;

  const body = new FormData();
  body.append("prompt", $("prompt").value.trim());
  body.append("negative_prompt", $("negativePrompt").value.trim());
  body.append("width", width);
  body.append("height", height);
  body.append("steps", $("steps").value);
  body.append("guidance_scale", $("guidance").value);
  body.append("seed", $("seed").value);
  selectedFiles.forEach((file) => body.append("images", file, file.name));

  setGenerating(true);
  lastJob = null;
  $("progressBar").style.width = "0%";
  $("progressMessage").textContent = t("progress.submitting");
  try {
    const job = await api("/jobs", { method: "POST", body });
    activeJob = job.id;
    await pollJob(job.id);
  } catch (error) {
    toast(error.message);
    $("jobInfo").textContent = t("job.error", { message: error.message });
  } finally {
    activeJob = null;
    setGenerating(false);
    refreshHealth();
  }
});

$("loadModelButton").addEventListener("click", async () => {
  try {
    await api("/model/load", { method: "POST" });
    $("loadModelButton").classList.add("hidden");
    $("jobInfo").textContent = t("job.loading");
  } catch (error) { toast(error.message); }
});

document.addEventListener("langchange", () => {
  renderFinishedJob();
  refreshHealth();
});

refreshHealth();
setInterval(refreshHealth, 4000);
