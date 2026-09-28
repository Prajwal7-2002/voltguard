# VoltGuard web dashboard

Next.js (App Router) + TypeScript + Tailwind + TanStack Query + Recharts.

The browser only talks to this app. Requests to `/api/*` are proxied to the
FastAPI service at `VOLTGUARD_API_URL` (default `http://localhost:8000`), read at
request time so one image works in any environment.

```bash
pnpm install
pnpm dev          # http://localhost:3000 (API must be running)
pnpm lint && pnpm typecheck && pnpm build
```

| Page | Backed by |
|---|---|
| Overview | `GET /models/latest`, `GET /data/summary` |
| Live monitor, Fleet | `POST /simulator/sessions`, `POST /simulator/sessions/{id}/tick` |
| Explainability | `POST /predict/explain`, `GET /data/summary` |
| Model | `GET /models/latest` |
