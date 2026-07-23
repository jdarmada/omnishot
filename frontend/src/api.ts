export interface Hit {
  chunk_id: string;
  clip_id: string;
  score: number;
  duration: number;
  start_sec: number;
  end_sec: number;
}

export interface SearchResponse {
  hits: Hit[];
  embed_ms: number;
  search_ms: number;
}

export interface StatusResponse {
  clips: number;
  chunks: number;
  state: string;
  current: string | null;
  watch_dir: string;
  events: { t: string; msg: string }[];
}

async function parseError(r: Response): Promise<string> {
  try {
    const body = await r.json();
    return body.detail || r.statusText;
  } catch {
    return r.statusText;
  }
}

export async function fetchStatus(): Promise<StatusResponse> {
  const r = await fetch("/api/status");
  if (!r.ok) throw new Error(await parseError(r));
  return r.json();
}

export async function searchText(query: string, k = 9): Promise<SearchResponse> {
  const r = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k }),
  });
  if (!r.ok) throw new Error(await parseError(r));
  return r.json();
}

export async function searchImage(image_b64: string, k = 9): Promise<SearchResponse> {
  const r = await fetch("/api/search_image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_b64, k }),
  });
  if (!r.ok) throw new Error(await parseError(r));
  return r.json();
}

export async function searchSimilar(chunkId: string): Promise<SearchResponse> {
  const r = await fetch(`/api/similar/${encodeURIComponent(chunkId)}`, {
    method: "POST",
  });
  if (!r.ok) throw new Error(await parseError(r));
  return r.json();
}

export async function revealClip(chunkId: string): Promise<boolean> {
  const r = await fetch(`/api/reveal/${encodeURIComponent(chunkId)}`, {
    method: "POST",
  });
  return r.ok;
}
