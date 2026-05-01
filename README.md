# Mykare Voice AI Agent

A production-grade, real-time voice AI assistant for healthcare appointment booking. Built with [LiveKit Agents](https://livekit.io/), the agent hears, understands, and speaks naturally while displaying a synchronized AI avatar — all within a 3–5 second response latency.

**Demo:**

<video controls width="100%">
  <source src="assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag. Watch the demo on [YouTube](https://youtu.be/42f8CcbzagY).
</video>

## Features

- **Real-Time Voice Conversation** — Full-duplex, natural speech interactions with context awareness across 5+ turn exchanges
- **AI Avatar** — Lip-synced virtual avatar powered by [Tavus](https://www.tavus.io/) with smooth, lag-free rendering
- **Intelligent Tool Calling** — The agent autonomously extracts structured data (name, phone, date, time, intent) and calls backend tools for appointment operations
- **Complete Appointment Lifecycle** — Book, retrieve, modify, and cancel appointments with double-booking prevention and SQLite persistence
- **Call Summaries** — Auto-generated conversation summaries with appointment details, user preferences, and timestamps
- **Live Tool Status UI** — Real-time visual feedback on the frontend showing tool execution status (e.g., "Fetching slots…", "Booking confirmed")
- **Authentication** — Login-gated access with credential validation

## Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│   Frontend (Web)    │         │   Backend (Agent)    │
│                     │         │                      │
│  Next.js 15 + React │◄──RTC──►│  Python + LiveKit    │
│  Tailwind CSS 4     │         │  Agents SDK          │
│  LiveKit Components │         │                      │
│  Radix UI           │         │  OpenAI (STT/TTS/LLM)│
│  Agents UI          │         │  Tavus (Avatar)      │
│                     │         │  SQLite (Database)   │
└─────────────────────┘         └──────────────────────┘
         │                                │
         └────────── LiveKit Cloud ───────┘
                    (WebRTC)
```

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Next.js 15, React 19, TypeScript, Tailwind CSS 4, Radix UI, LiveKit Components |
| **Backend** | Python 3.10+, LiveKit Agents SDK, asyncio |
| **AI/ML** | OpenAI (GPT-4o Transcribe, GPT-5.3 Chat, GPT-4o-mini TTS), Tavus Avatars |
| **Audio** | Silero VAD, LiveKit Turn Detector (multilingual), AI Coustics noise cancellation |
| **Database** | SQLite (via aiosqlite) |
| **Infrastructure** | LiveKit Cloud (WebRTC), uv (Python package manager), pnpm |

## How It Works

1. **User Identification** — The agent asks for the patient's name and phone number, using the phone number as a unique identifier
2. **Intent Detection** — The LLM determines whether the user wants to book, check, modify, or cancel an appointment
3. **Tool Execution** — The agent calls the appropriate tool (`fetch_slots`, `book_appointment`, `retrieve_appointments`, `cancel_appointment`, `modify_appointment`)
4. **Real-Time Feedback** — Tool call status is sent via RPC to the frontend and displayed visually
5. **Conversation Summary** — At the end, the agent generates a concise summary of all actions taken

## Getting Started

### Prerequisites

- [LiveKit Cloud](https://cloud.livekit.io/) account
- [OpenAI](https://platform.openai.com/) API key
- [Tavus](https://www.tavus.io/) API key and configured replica/persona
- Python 3.10+ and [uv](https://github.com/astral-sh/uv)
- Node.js 20+ and [pnpm](https://pnpm.io/)

### Backend Setup

```bash
cd backend/agent-starter-python

# Install dependencies
uv sync

# Configure environment
cp .env.example .env.local
# Fill in LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, OPENAI_API_KEY, TAVUS_API_KEY

# Download required models (VAD, turn detector)
uv run python src/agent.py download-files

# Start the agent
uv run python src/agent.py dev
```

### Frontend Setup

```bash
cd frontend/agent-starter-react

# Install dependencies
pnpm install

# Configure environment
cp .env.example .env.local
# Fill in LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET

# Start the dev server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
├── backend/agent-starter-python/
│   ├── src/
│   │   ├── agent.py          # Main agent logic, tool definitions, LLM instructions
│   │   └── database.py        # SQLite operations for users, slots, appointments
│   ├── tests/                 # Test suite (pytest)
│   └── pyproject.toml         # Python dependencies
├── frontend/agent-starter-react/
│   ├── app/                   # Next.js app router pages and API routes
│   ├── components/            # React components (UI, agents-ui integration)
│   ├── hooks/                 # Custom hooks (tool call status, etc.)
│   └── package.json           # Node.js dependencies
└── docs/
    └── TODO                   # Feature checklist
```

## License

MIT
