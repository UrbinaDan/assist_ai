# Assist

Live transcription + interview coach (PWA, Next.js, Codespaces-ready)

## Quick Start (Codespaces)

1. Open this repo in GitHub Codespaces.
2. In the terminal:
   ```sh
   cd apps/web
   pnpm install
   pnpm dev
   ```
3. Set port 3000 to **Public**.
4. Open the Codespaces URL on your phone.

- On Android Chrome: tap menu → **Install app** or **Add to Home Screen**.
- On iOS Safari: tap Share → **Add to Home Screen**.

You’ll get a full-screen PWA.

## Features

- Mic capture with live volume meter.
- PWA manifest + service worker.
- Minimal UI for transcript and interview tips.

## What’s next

- Wire up WebSocket to stream audio chunks to ASR service.
- Show live transcript and interview tips from backend.

# assist_ai
AI assistant for setting reminders, meetings, and taking notes of any discussion you have. 

´´´
bash

your-app/
├─ apps/
│  ├─ web/                         # Next.js PWA
│  │  ├─ app/                      # App Router pages / routes
│  │  ├─ components/               # UI components
│  │  ├─ hooks/                    # useAudioStream, useWebSocket, etc.
│  │  ├─ lib/                      # client utils
│  │  ├─ public/                   # icons, manifest.json, robots.txt
│  │  ├─ service-worker.js         # PWA caching, push notifications
│  │  └─ next.config.js
│  ├─ api/                         # Fastify orchestrator
│  │  ├─ src/
│  │  │  ├─ index.ts               # Fastify bootstrapping
│  │  │  ├─ routes/                # /auth, /ws, /calendar, /llm
│  │  │  ├─ sockets/               # ws handlers (transcript, tips)
│  │  │  ├─ pipeline/              # transcript windowing, triggers
│  │  │  ├─ services/              # redis, prisma, calendar, auth
│  │  │  ├─ queue/                 # enqueue LLM jobs (BullMQ)
│  │  │  └─ schemas/               # zod schemas (shared import)
│  │  └─ Dockerfile
│  └─ asr/                         # Python Whisper service
│     ├─ app.py                    # FastAPI, /ingest, /flush, /health
│     ├─ vad.py                    # voice activity detection
│     ├─ diarization.py            # (optional) pyannote hooks
│     ├─ segments.py               # rolling segmenter
│     └─ Dockerfile
│
├─ packages/
│  ├─ ui/                          # design system (buttons, toasts, panes)
│  ├─ prompts/                     # prompt templates
│  ├─ schemas/                     # zod types (ActionJSON, TipsJSON, etc.)
│  ├─ types/                       # shared TS types
│  └─ config/                      # eslint, tsconfig, tailwind, env parsing
│
├─ infra/
│  ├─ docker-compose.yml           # web, api, asr, redis, postgres
│  ├─ nginx.conf                   # (optional) reverse proxy / TLS
│  └─ terraform/                   # (later) cloud infra
│
├─ prisma/
│  ├─ schema.prisma
│  └─ migrations/
│
├─ .env.example                    # required env vars
├─ package.json
└─ pnpm-workspace.yaml


´´´
