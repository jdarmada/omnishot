import "./style.css";
import {
  fetchRecent,
  fetchStatus,
  pickAndSetLibrary,
  revealClip,
  searchImage,
  searchSimilar,
  searchText,
  type Hit,
} from "./api";

const $ = <T extends HTMLElement>(id: string) => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} missing`);
  return el as T;
};

const qInput = $<HTMLInputElement>("q");
const goBtn = $<HTMLButtonElement>("go");
const uploadImageBtn = $<HTMLButtonElement>("upload-image");
const imageFileInput = $<HTMLInputElement>("image-file");
const changeFolderBtn = $<HTMLButtonElement>("change-folder");
const libPath = $("lib-path");
const grid = $("grid");
const gridHeader = $("grid-header");
const empty = $("empty");
const eventsEl = $("events");
const logToggle = $<HTMLButtonElement>("log-toggle");
const progressEl = $("progress");
const pLabel = $("p-label");
const pCount = $("p-count");
const pFill = $("p-fill");
const queryChip = $("querychip");
const qImg = $<HTMLImageElement>("qimg");
const searchRow = $("searchrow");
const latEl = $("s-lat");

let imageB64: string | null = null;
let mode: "recent" | "search" = "recent";
let lastChunks = -1;
let logOpen = false;

const EMPTY_DEFAULT = "Link a folder of footage to get started.";
const EMPTY_DEFAULT_HINT =
  "Use Change folder… or drop videos into the library path above, then describe the shot you need.";

function shortenPath(path: string): string {
  const parts = path.replace(/\\/g, "/").split("/").filter(Boolean);
  if (parts.length <= 3) return path;
  return `…/${parts.slice(-3).join("/")}`;
}

function tc(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

const MONTHS = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

// uploaded_at is "YYYY-MM-DD"; parse manually so timezones can't shift the day
function fmtDate(iso?: string | null): string {
  if (!iso) return "new";
  const [, m, d] = iso.split("-").map(Number);
  if (!m || !d) return "new";
  return `${MONTHS[m - 1]} ${d}`;
}

function pushUrl(url: string): void {
  const current = location.pathname + location.search;
  if (current !== url) history.pushState(null, "", url);
}

function setEmptyMessage(title: string, hint?: string): void {
  empty.style.display = "block";
  grid.innerHTML = "";
  gridHeader.hidden = true;
  const titleEl = empty.firstElementChild;
  if (titleEl) titleEl.textContent = title;
  const hintEl = empty.querySelector(".hint");
  if (hintEl && hint) hintEl.textContent = hint;
}

function render(hits: Hit[], emptyMessage = "No matches. Try different words."): void {
  if (!hits.length) {
    setEmptyMessage(emptyMessage, mode === "recent" ? EMPTY_DEFAULT_HINT : undefined);
    return;
  }
  empty.style.display = "none";
  gridHeader.hidden = mode !== "recent";
  grid.innerHTML = hits
    .map(
      (h, i) => `
    <div class="clip">
      <div class="rank ${i === 0 && mode === "search" ? "top" : ""}">${mode === "search" ? `#${i + 1}` : fmtDate(h.uploaded_at)}</div>
      <video src="/api/clip/${encodeURIComponent(h.chunk_id)}" muted loop playsinline preload="metadata"></video>
      <div class="meta">
        <span>
          <span class="id" title="${h.clip_id}">${h.clip_id}</span><br>
          <span class="timecode"><b>${tc(h.start_sec)}–${tc(h.end_sec)}</b> in source · ${h.duration.toFixed(1)}s</span>
        </span>
        <span>
          <button class="similar" type="button" data-similar="${h.chunk_id}">≈ More</button>
          <button class="reveal" type="button" data-reveal="${h.chunk_id}">Reveal ↗</button>
        </span>
      </div>
    </div>`
    )
    .join("");

  grid.querySelectorAll<HTMLVideoElement>("video").forEach((video) => {
    video.addEventListener("mouseenter", () => void video.play());
    video.addEventListener("mouseleave", () => video.pause());
  });

  grid.querySelectorAll<HTMLButtonElement>("[data-similar]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.similar;
      if (id) void similar(id);
    });
  });

  grid.querySelectorAll<HTMLButtonElement>("[data-reveal]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.reveal;
      if (id) void reveal(id, btn);
    });
  });
}

async function loadRecent(): Promise<void> {
  try {
    const data = await fetchRecent(9);
    mode = "recent";
    latEl.textContent = "";
    render(data.hits, EMPTY_DEFAULT);
  } catch {
    // backend not ready yet; poll will retry on the next chunk-count change
  }
}

function updateProgress(
  state: string,
  current: string | null,
  done: number,
  total: number
): void {
  const processing = state === "processing" && total > 0;
  progressEl.hidden = !processing;
  if (!processing) return;
  pLabel.textContent = `indexing ${current ?? "…"}`;
  pCount.textContent = `${Math.min(done + 1, total)} of ${total}`;
  const pct = Math.min(((done + 0.5) / total) * 100, 98);
  pFill.style.width = `${pct.toFixed(1)}%`;
}

async function poll(): Promise<void> {
  try {
    const s = await fetchStatus();
    libPath.textContent = shortenPath(s.watch_dir);
    libPath.title = s.watch_dir;
    $("s-clips").textContent = String(s.clips);
    $("s-chunks").textContent = String(s.chunks);
    $("s-state").innerHTML =
      s.state === "processing"
        ? `<span class="busy">⚙ indexing</span>`
        : s.state === "switching"
          ? `<span class="busy">switching library…</span>`
          : `<span>● ${s.state}</span>`;
    updateProgress(s.state, s.current, s.queue_done, s.queue_total);
    eventsEl.innerHTML = (s.events || [])
      .map((e) => `<div>${e.t} · ${e.msg}</div>`)
      .join("");

    // Refresh the landing grid when new footage finishes indexing.
    if (mode === "recent" && s.chunks !== lastChunks) {
      lastChunks = s.chunks;
      void loadRecent();
    }
  } catch {
    $("s-state").textContent = "backend offline";
  }
}

async function changeFolder(): Promise<void> {
  changeFolderBtn.disabled = true;
  changeFolderBtn.textContent = "Picking…";
  try {
    const res = await pickAndSetLibrary();
    libPath.textContent = shortenPath(res.watch_dir);
    libPath.title = res.watch_dir;
    mode = "recent";
    lastChunks = -1;
    setEmptyMessage(
      "Indexing your library…",
      "Videos in this folder will appear here as they finish embedding."
    );
    await poll();
  } catch (e) {
    const msg = (e as Error).message;
    if (!/no folder selected/i.test(msg)) {
      alert(`Could not change library: ${msg}`);
    }
  } finally {
    changeFolderBtn.disabled = false;
    changeFolderBtn.textContent = "Change folder…";
  }
}

async function goHome(push = true): Promise<void> {
  clearImage();
  qInput.value = "";
  if (push) pushUrl(location.pathname);
  await loadRecent();
}

async function runTextSearch(query: string, push = true): Promise<void> {
  goBtn.disabled = true;
  goBtn.textContent = "…";
  try {
    const data = await searchText(query);
    mode = "search";
    latEl.innerHTML = `<span class="lat">embed ${data.embed_ms.toFixed(0)}ms · search ${data.search_ms.toFixed(1)}ms</span>`;
    render(data.hits);
    if (push) pushUrl(`${location.pathname}?q=${encodeURIComponent(query)}`);
  } catch (e) {
    setEmptyMessage(`Search failed: ${(e as Error).message}`);
  } finally {
    goBtn.disabled = false;
    goBtn.textContent = "Find";
  }
}

async function find(): Promise<void> {
  if (imageB64) {
    await findByImage();
    return;
  }
  const query = qInput.value.trim();
  if (!query) {
    await goHome();
    return;
  }
  await runTextSearch(query);
}

async function findByImage(): Promise<void> {
  if (!imageB64) return;
  goBtn.disabled = true;
  goBtn.textContent = "…";
  try {
    const data = await searchImage(imageB64);
    mode = "search";
    latEl.innerHTML = `<span class="lat">image query · embed ${data.embed_ms.toFixed(0)}ms · search ${data.search_ms.toFixed(1)}ms</span>`;
    render(data.hits);
    // Image queries can't be encoded in the URL; just clear stale params.
    history.replaceState(null, "", location.pathname);
  } catch (e) {
    alert(`Image search failed: ${(e as Error).message}`);
  } finally {
    goBtn.disabled = false;
    goBtn.textContent = "Find";
  }
}

function clearImage(): void {
  imageB64 = null;
  imageFileInput.value = "";
  queryChip.classList.remove("visible");
}

function loadImageFile(file: File): void {
  if (!file.type.startsWith("image/")) {
    alert("Please choose an image file (JPEG, PNG, WebP, …).");
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = String(reader.result);
    imageB64 = dataUrl.split(",")[1] ?? null;
    qImg.src = dataUrl;
    queryChip.classList.add("visible");
    qInput.value = "";
    void findByImage();
  };
  reader.readAsDataURL(file);
}

async function similar(chunkId: string, push = true): Promise<void> {
  clearImage();
  qInput.value = "";
  try {
    const data = await searchSimilar(chunkId);
    mode = "search";
    latEl.innerHTML = `<span class="lat">similar via stored vector · search ${data.search_ms.toFixed(1)}ms · no embedding call</span>`;
    render(data.hits);
    if (push) pushUrl(`${location.pathname}?similar=${encodeURIComponent(chunkId)}`);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (e) {
    alert(`Similar search failed: ${(e as Error).message}`);
  }
}

// Restore the view described by the current URL (initial load + back/forward).
async function dispatchFromLocation(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const q = params.get("q");
  const sim = params.get("similar");
  if (q) {
    qInput.value = q;
    await runTextSearch(q, false);
  } else if (sim) {
    await similar(sim, false);
  } else {
    await goHome(false);
  }
}

async function reveal(chunkId: string, btn: HTMLButtonElement): Promise<void> {
  try {
    const ok = await revealClip(chunkId);
    if (ok) {
      btn.textContent = "Revealed ✓";
      btn.classList.add("done");
    } else {
      btn.textContent = "not found";
    }
  } catch {
    btn.textContent = "error";
  }
}

["dragenter", "dragover"].forEach((ev) => {
  searchRow.addEventListener(ev, (e) => {
    e.preventDefault();
    searchRow.classList.add("dropping");
  });
});
["dragleave", "drop"].forEach((ev) => {
  searchRow.addEventListener(ev, (e) => {
    e.preventDefault();
    searchRow.classList.remove("dropping");
  });
});

searchRow.addEventListener("drop", (e) => {
  const dt = (e as DragEvent).dataTransfer;
  if (!dt) return;
  const file = [...dt.files].find((f) => f.type.startsWith("image/"));
  if (file) loadImageFile(file);
});

uploadImageBtn.addEventListener("click", () => imageFileInput.click());
imageFileInput.addEventListener("change", () => {
  const file = imageFileInput.files?.[0];
  if (file) loadImageFile(file);
});

logToggle.addEventListener("click", () => {
  logOpen = !logOpen;
  eventsEl.hidden = !logOpen;
  logToggle.classList.toggle("open", logOpen);
  logToggle.textContent = logOpen ? "log ▴" : "log ▾";
});

$("clear-image").addEventListener("click", clearImage);
$("logo").addEventListener("click", () => void goHome());
changeFolderBtn.addEventListener("click", () => void changeFolder());
goBtn.addEventListener("click", () => void find());
qInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") void find();
});
window.addEventListener("popstate", () => void dispatchFromLocation());

void poll();
void dispatchFromLocation();
setInterval(() => void poll(), 3000);
