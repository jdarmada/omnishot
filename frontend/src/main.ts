import "./style.css";
import {
  fetchStatus,
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
const grid = $("grid");
const empty = $("empty");
const eventsEl = $("events");
const queryChip = $("querychip");
const qImg = $<HTMLImageElement>("qimg");
const searchRow = $("searchrow");
const latEl = $("s-lat");

let imageB64: string | null = null;

function tc(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function setEmptyMessage(title: string, hint?: string): void {
  empty.style.display = "block";
  grid.innerHTML = "";
  const titleEl = empty.firstElementChild;
  if (titleEl) titleEl.textContent = title;
  const hintEl = empty.querySelector(".hint");
  if (hintEl && hint) hintEl.textContent = hint;
}

function render(hits: Hit[]): void {
  if (!hits.length) {
    setEmptyMessage("No matches. Try different words.");
    return;
  }
  empty.style.display = "none";
  grid.innerHTML = hits
    .map(
      (h, i) => `
    <div class="clip">
      <div class="rank ${i === 0 ? "top" : ""}">#${i + 1}</div>
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

async function poll(): Promise<void> {
  try {
    const s = await fetchStatus();
    $("s-dir").textContent = s.watch_dir.split("/").slice(-2).join("/");
    $("s-clips").textContent = String(s.clips);
    $("s-chunks").textContent = String(s.chunks);
    $("s-state").innerHTML =
      s.state === "processing"
        ? `<span class="busy">⚙ indexing ${s.current}…</span>`
        : `<span>● ${s.state}</span>`;
    eventsEl.innerHTML = (s.events || [])
      .map((e) => `<div>${e.t} · ${e.msg}</div>`)
      .join("");
  } catch {
    $("s-state").textContent = "backend offline";
  }
}

async function find(): Promise<void> {
  if (imageB64) {
    await findByImage();
    return;
  }
  const query = qInput.value.trim();
  if (!query) return;
  goBtn.disabled = true;
  goBtn.textContent = "…";
  try {
    const data = await searchText(query);
    latEl.innerHTML = `<span class="lat">embed ${data.embed_ms.toFixed(0)}ms · search ${data.search_ms.toFixed(1)}ms</span>`;
    render(data.hits);
  } catch (e) {
    setEmptyMessage(`Search failed: ${(e as Error).message}`);
  } finally {
    goBtn.disabled = false;
    goBtn.textContent = "Find";
  }
}

async function findByImage(): Promise<void> {
  if (!imageB64) return;
  goBtn.disabled = true;
  goBtn.textContent = "…";
  try {
    const data = await searchImage(imageB64);
    latEl.innerHTML = `<span class="lat">image query · embed ${data.embed_ms.toFixed(0)}ms · search ${data.search_ms.toFixed(1)}ms</span>`;
    render(data.hits);
  } catch (e) {
    alert(`Image search failed: ${(e as Error).message}`);
  } finally {
    goBtn.disabled = false;
    goBtn.textContent = "Find";
  }
}

function clearImage(): void {
  imageB64 = null;
  queryChip.classList.remove("visible");
}

async function similar(chunkId: string): Promise<void> {
  clearImage();
  qInput.value = "";
  try {
    const data = await searchSimilar(chunkId);
    latEl.innerHTML = `<span class="lat">similar via stored vector · search ${data.search_ms.toFixed(1)}ms · no embedding call</span>`;
    render(data.hits);
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (e) {
    alert(`Similar search failed: ${(e as Error).message}`);
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
  if (!file) return;
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
});

$("clear-image").addEventListener("click", clearImage);
goBtn.addEventListener("click", () => void find());
qInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") void find();
});

void poll();
setInterval(() => void poll(), 3000);
