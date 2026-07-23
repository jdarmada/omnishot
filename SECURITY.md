# Security Policy

## Threat model: this is a local tool

omnishot-ts is designed to run on **your own machine** for **your own footage**.
Several endpoints intentionally interact with the host OS and have no
authentication:

- `POST /api/reveal/{id}` — opens your file manager at the source clip
- `GET /api/pick-folder` / `POST /api/library/pick` — opens a native folder dialog on the server
- `POST /api/library` — points the indexer at any readable folder path

**Do not expose this app to the public internet or an untrusted network.**
The server binds to `127.0.0.1` by default (uvicorn's default, and the
provided docker-compose maps ports to localhost only). Keep it that way unless
you put an authenticating reverse proxy in front and understand the
consequences.

Defense-in-depth measures in place:

- `/api/clip` refuses to serve files outside the configured `CHUNKS_DIR`,
  even if the Elasticsearch index is tampered with.
- Compose binds Elasticsearch and the app to `127.0.0.1` only.
- Secrets live in `.env` (gitignored); never commit API keys.

## Supported versions

Only the latest release receives security fixes.

## Reporting a vulnerability

Please report vulnerabilities privately via
[GitHub Security Advisories](https://github.com/jdarmada/omnishot-ts/security/advisories/new)
rather than opening a public issue. Include reproduction steps and impact.
You should receive a response within a week.
