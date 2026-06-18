const form = document.querySelector("#uploadForm");
const input = document.querySelector("#scanPackage");
const dropZone = document.querySelector("#dropZone");
const fileName = document.querySelector("#fileName");
const submitButton = document.querySelector("#submitButton");
const resetButton = document.querySelector("#resetButton");
const capabilityBadge = document.querySelector("#capabilityBadge");
const jobIdLabel = document.querySelector("#jobId");
const stateText = document.querySelector("#stateText");
const progressBar = document.querySelector("#progressBar");
const details = document.querySelector("#details");
const downloads = document.querySelector("#downloads");
const resultLink = document.querySelector("#resultLink");
const previewLink = document.querySelector("#previewLink");
const printLink = document.querySelector("#printLink");
const lidarLink = document.querySelector("#lidarLink");

let pollTimer = null;
let currentJobId = "";

const progressByState = {
  idle: 0,
  queued: 25,
  processing: 68,
  complete: 100,
  failed: 100,
};

async function loadCapabilities() {
  try {
    const response = await fetch("/capabilities");
    const payload = await response.json();
    capabilityBadge.textContent = payload.state || "Unknown";
    capabilityBadge.className = `badge ${payload.state || ""}`;
  } catch (error) {
    capabilityBadge.textContent = "Offline";
    capabilityBadge.className = "badge unavailable";
  }
}

function setState(state, payload = {}) {
  const label = state.charAt(0).toUpperCase() + state.slice(1);
  stateText.textContent = label;
  progressBar.style.width = `${progressByState[state] ?? 0}%`;
  details.classList.toggle("error", state === "failed");
  details.textContent = JSON.stringify(payload, null, 2);
}

function setDownloads(jobId) {
  resultLink.href = `/jobs/${jobId}/result`;
  previewLink.href = `/jobs/${jobId}/preview`;
  printLink.href = `/jobs/${jobId}/print`;
  lidarLink.href = `/jobs/${jobId}/lidar`;
  downloads.hidden = false;
}

function clearPoll() {
  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

async function pollJob(jobId) {
  try {
    const response = await fetch(`/jobs/${jobId}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || `HTTP ${response.status}`);
    }

    setState(payload.status || "queued", payload);

    if (payload.status === "complete") {
      setDownloads(jobId);
      submitButton.disabled = false;
      return;
    }

    if (payload.status === "failed") {
      submitButton.disabled = false;
      return;
    }

    pollTimer = window.setTimeout(() => pollJob(jobId), 1800);
  } catch (error) {
    setState("failed", { error: error.message });
    submitButton.disabled = false;
  }
}

async function startJob(file) {
  const body = new FormData();
  body.append("scan_package", file);

  submitButton.disabled = true;
  downloads.hidden = true;
  setState("queued", { file: file.name });

  const response = await fetch("/jobs", {
    method: "POST",
    body,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }

  currentJobId = payload.job_id;
  jobIdLabel.textContent = currentJobId;
  setState(payload.status || "queued", payload);
  await pollJob(currentJobId);
}

function setInputFiles(files) {
  if (!files || files.length === 0) {
    fileName.textContent = "Choose or drop a zip file";
    return;
  }
  fileName.textContent = files[0].name;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearPoll();
  const file = input.files?.[0];
  if (!file) {
    setState("failed", { error: "No file selected." });
    return;
  }

  try {
    await startJob(file);
  } catch (error) {
    setState("failed", { error: error.message });
    submitButton.disabled = false;
  }
});

input.addEventListener("change", () => setInputFiles(input.files));

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  if (!event.dataTransfer?.files?.length) {
    return;
  }
  input.files = event.dataTransfer.files;
  setInputFiles(input.files);
});

resetButton.addEventListener("click", () => {
  clearPoll();
  form.reset();
  currentJobId = "";
  fileName.textContent = "Choose or drop a zip file";
  jobIdLabel.textContent = "No job";
  downloads.hidden = true;
  submitButton.disabled = false;
  setState("idle", {});
});

setState("idle", {});
loadCapabilities();
