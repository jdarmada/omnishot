# Talk Notes — Slide ↔ Demo Mapping

This is how each piece of the repo ties to a slot in the conference talk. Use this as your run-of-show.

## Slot 1 — Opener (3 min)

**Slide:** Real footage of a video editor scrubbing through b-roll. Show the clock running. Frame the problem in one sentence.

**Talking point:** "This is a colleague of mine. She spends hours doing this every week. Multimodal embeddings should have made this a solved problem by now."

**Demo:** None yet. Just storytelling.

## Slot 2 — The naive solution (3 min)

**Slide:** "The model is the easy part."

**Demo:** Open `notebooks/01_baseline_retrieval.ipynb`. Run the three sample queries live. Audience sees results coming back, sees the scores, sees the chunk IDs. They're impressed.

**Talking point:** "Jina v5-omni, Elasticsearch, fifty lines of code. We're done, right?"

## Slot 3 — Where it falls over (4 min)

**Slides (3):**
1. *Storage math* — for a 10,000-clip library at 1024-d float32, that's 40 MB of vectors. At 100k clips, 400 MB. At 1M clips with three chunks each, 12 GB just for embeddings. (Show this as a back-of-the-napkin table.)
2. *The 32-frame problem* — v5-omni samples 32 evenly-spaced frames from any clip. For a 3-minute clip, that's one frame per 5.6 seconds. Show the actual frame samples for a long clip. Whole shots disappear.
3. *One query that fails* — show "find a clip with the same vibe as this reference clip" (image-to-image), which Elastic explicitly flags as a weakness.

**Demo:** Either pre-recorded screenshots, or open `notebooks/02_chunking_experiment.ipynb` and show the long-clip recall drop live.

## Slot 4 — What actually matters (10 min)

This is the meat of the talk. Three subsections, each producing a chart.

### 4a. Chunking (3 min)
Open `notebooks/02_chunking_experiment.ipynb`. Show the bar chart of recall@10 by clip-duration bucket, comparing `whole` vs. `scene` vs. `fixed30`.

**Money line:** "Scene chunking helps clips over 60 seconds by X%, and surprisingly *hurts* short clips by Y%. The crossover is at roughly 30 seconds."

### 4b. Compression (4 min)
Open `notebooks/03_compression_bench.ipynb`. Run the benchmark live if time permits, otherwise show pre-computed output. Show the Pareto chart from the final cell.

**Money line:** "BBQ-1024 cuts our storage 32×. We lose X% recall. With a reranking pass over the top-50, we recover most of it. This is the move."

### 4c. Hybrid (3 min)
Switch to the live UI. Type "outdoor shot". Toggle "uploaded this week" + "under 15s". Show how the result set changes. Then toggle hybrid mode and show how the BM25-fused version surfaces a different ordering.

**Money line:** "Pure vector DBs treat retrieval as a math problem. Search engines treat it as a query problem. Multimodal makes the difference matter more, not less."

## Slot 5 — What still doesn't work (2 min)

**Slide:** A short list of failure modes, each one specific:
- Image-to-image similarity (Elastic flags this)
- Subtle audio queries (the 30-second segmentation is coarse)
- Compositional queries needing scene-graph reasoning
- Long-tail concepts outside training

**Demo:** Show the q13 "same vibe as moody-rain-clip" query failing live in the UI. Don't hide it.

**Talking point:** "I'm not selling you snake oil. Here's where we tried and it didn't work, and here's what we'd try next."

## Slot 6 — Takeaways (3 min)

Three slides:

1. **The pattern** — one-liner architecture diagram. Jina v5-omni embeddings → Elasticsearch hybrid retrieval → optional rerank.
2. **The eval methodology** — point at this repo. "If you build one of these, build the eval set first, not the model code."
3. **The thesis** — "The bottleneck has moved. The model isn't the hard part anymore. Evaluate honestly, compress aggressively, and reach for hybrid retrieval."

## Pre-talk checklist

- [ ] Build the eval set with the actual editor (~1 hour interview)
- [ ] Ingest a corpus of 200-500 clips into all 7+ configs
- [ ] Pre-warm Elasticsearch (run each query once to populate caches)
- [ ] Pre-compute notebook outputs as backup
- [ ] Have a stable test query ready: one that *always works* (for the opener) and one that *always fails* (for the negatives section)
- [ ] Record a screencap of the UI as backup in case wifi is unreliable
- [ ] Print the money chart on a slide as a static fallback

## Stage logistics

- Open the UI in one browser tab, the notebook in another
- Have a terminal visible for one moment when you run `ingest.py` (so the audience sees how short it is)
- Keep the URL bar visible — `localhost:8000` is part of the demo's authenticity
- Don't switch between configs mid-demo; use the same index throughout the live portion
