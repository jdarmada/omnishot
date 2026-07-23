# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-07-23

### Added

- Folder-watch ingest: drop videos in a linked library folder → scene-chunked
  (PySceneDetect), embedded (Jina v5-omni-small), indexed (Elasticsearch HNSW)
- Search by text description, by uploaded/dropped reference image, and
  "more like this" via stored vectors
- Reveal-in-file-manager for source clips (macOS / Windows / Linux)
- Library folder picker (native OS dialog) with corpus rebuild on switch
- Lightweight download scripts (Pexels, YouTube) and batch ingest CLI
- Vite + TypeScript frontend; FastAPI backend
- One-command full-stack startup via Docker Compose
- `/api/health` endpoint; graceful wait for Elasticsearch on startup
- CI (lint, tests, frontend build, Docker build) and Dependabot
