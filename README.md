# LocalMind — Fully Local Conversational AI Platform

LocalMind is an application developed as part of research into scaling test-time compute. The reasoning modes feature in the application (`Medium`, `High`, and `Extra`) implements the findings of this research, allowing the system to allocate additional compute resources at inference time to improve answer accuracy.

While the system is designed to run entirely locally without transmitting data to external servers, an active internet connection is required during the initial startup to download the default model (`Qwen/Qwen2.5-3B-Instruct-GGUF`). Internet access is also required to browse, download, and add new models from Hugging Face.

## Video Demo

[Watch the demonstration video here](https://github.com/user-attachments/assets/f5fb53d5-b44e-4a4f-8154-eae625603a93)

---

## Core Features

* **Scaling Test-Time Compute**: Offers configurable reasoning modes (`Low`, `Medium`, `High`, `Extra`) that implement test-time compute allocation strategies using self-consistency and multi-sample verification.
* **On-Demand Model Management**: An integrated model browser allows users to search Hugging Face for GGUF models, view available quantizations, and download files directly to the local models registry.
* **Dynamic Model Routing**: Utilizes `llama-swap` to automatically load, unload, and hot-swap local GGUF models in VRAM/RAM based on the selected chat configuration.
* **Private Retrieval-Augmented Generation (RAG)**: Supports local document ingestion (e.g., PDF, Python, Markdown) via direct drag-and-drop into the chat window. Document text is chunked and stored in a local Chroma vector database to provide grounded contextual citations.
* **KaTeX LaTeX & Markdown Rendering**: Renders standard Markdown structures, syntax-highlighted code blocks, and mathematical notation using KaTeX auto-render (supporting both inline `$` and block `$$` equations).
* **Configurable System Persona & Settings**: Switch between pre-configured personalities (General, Coder, Researcher, Gen-Z) and tune generative settings like inference temperature directly from the input composer.

---

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Vanilla HTML5 / CSS3 / ES6 JavaScript (single-page app) |
| **Backend** | Python, FastAPI, async SQLAlchemy (`asyncpg`) |
| **AI Engine** | LangChain, LangGraph, ChatOpenAI (local endpoint) |
| **Vector Store** | ChromaDB, HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Model Serving** | llama-swap, llama.cpp (GGUF format) |
| **Database** | PostgreSQL 15 |
| **Deployment** | Docker |

---

## Getting Started

### Prerequisites

* Python 3.10+ (if running bare-metal)
* Docker & Docker Compose (recommended for containerized execution)
* NVIDIA GPU with CUDA support (highly recommended for local model inference speed)

### Installation & Launch

#### Option A: Running with Docker (Recommended)

LocalMind includes pre-configured container definitions and quick-launch wrapper scripts.

1. Clone the repository:
   ```bash
   git clone https://github.com/abcdef54/LocalMind.git
   cd LocalMind
   ```

2. Start the application stack using the quick-launch script:
   
   * **On Windows (PowerShell/CMD)**:
     ```powershell
     .\start.bat -c 32 -p 18000
     ```
   
   * **On Linux / macOS (Terminal)**:
     ```bash
     chmod +x start.sh
     ./start.sh -c 32 -p 18000
     ```

   *Optional Flags*:
   * `-c` or `--context-length`: Sets the default context length in thousands of tokens (e.g., `16`, `32`).
   * `-p` or `--port`: Sets the host port that `llama-swap` runs on (default: `18000`).

3. Once all containers show as running and healthy, open **`http://localhost:5500`** in a browser to access the frontend user interface.

---

#### Option B: Bare-Metal Installation (Manual Development Setup)

To run the backend and frontend services directly on your host machine without Docker:

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/abcdef54/LocalMind.git
   cd LocalMind
   ```

2. Set up a Python virtual environment and install the required dependencies:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. Start `llama-swap` inside a Docker container (handles GGUF model execution):
   ```powershell
   # Windows PowerShell example:
   docker run -it --rm `
     --gpus all `
     -p 18000:18000 `
     -v <path-to-project>\src\backend\ai\llms\cpp_models:/models `
     -v <path-to-project>\llama_swap_config.yaml:/app/config.yaml `
     ghcr.io/mostlygeek/llama-swap:cuda `
     --config /app/config.yaml --watch-config --listen 0.0.0.0:18000
   ```

4. Start the FastAPI backend server:
   ```bash
   python main.py -c 32 -p 18000
   ```

5. Serve or open the static frontend:
   Open `src/frontend/index.html` in your web browser, or serve the directory using a static web server on port `5500`.

---

## Development Journey

Database schema changes now use Alembic. Backend startup applies pending migrations;
see [database migration instructions](src/backend/database/migrations/README.md)
for existing-database backups, baseline adoption, CLI commands, and integration tests.

The [API v1 foundation](src/backend/api/README.md) documents shared settings,
request IDs, error responses, compatibility, and tests. Its OpenAPI schema is
available at `/api/v1/openapi.json`; feature endpoints are added incrementally.

The [durable run executor](src/backend/execution/README.md) documents PostgreSQL
claims, leases, recovery and shutdown. Its AI handler is connected in the later
LangGraph integration task; legacy chat behavior remains unchanged at this stage.

The [submission upload guide](src/backend/uploads/README.md) describes multipart sends,
filename duplicate checks, upload limits, storage cleanup and integration boundaries.

LocalMind has evolved through several major architectural generations.

#### LocalMind v1.0 — The Prototype

The first prototype focused on building a simple conversational interface with basic in-memory conversation history and a minimal frontend.

#### LocalMind v2.0 — Accounts & Retrieval

Introduced a client-server architecture featuring JWT authentication, Google OAuth, document uploads, Chroma-based Retrieval-Augmented Generation (RAG), and citation support.

#### LocalMind v3.x — User Experience

Focused on improving usability and presentation through rich Markdown rendering, LaTeX support, syntax highlighting, tables, improved file handling, conversation management, and iterative UI refinements.

#### LocalMind v4.0 — Research & Experimentation

This phase explored test-time reasoning techniques.

The original Deep Think implementation introduced self-reflection and multi-pass inference, serving as the first experimental reasoning pipeline and laying the groundwork for more advanced orchestration.

#### LocalMind v5.0 — Fully Local AI Platform (Current)

A complete architectural redesign replacing cloud-hosted inference with a fully local AI stack.

Major architectural changes include:

* Complete removal of authentication and cloud dependencies
* LangGraph orchestration engine
* Multi-model support through llama-swap
* Integrated Hugging Face model browser and downloader
* Dynamic GGUF model management
* Configurable reasoning modes (Low, Medium, High, Extra)
* Automatic conversation summarization and memory management
* Local RAG with Chroma vector search
* Configurable local inference parameters
* Designed as a platform for experimenting with test-time compute scaling and reasoning algorithms.

---

## Repository Structure

```text
LocalMind/
├── .env                                  # Environment variables (auto-created on first run)
├── .gitignore
├── docker-compose.yaml                   # Multi-container orchestration
├── LICENSE                               # MIT License
├── llama_swap_config.yaml                # Auto-generated llama-swap model routing config
├── main.py                               # Entry point — starts the FastAPI server
├── requirements.txt                      # Python dependencies
│
├── src/
│   ├── backend/                          # FastAPI async backend
│   │   ├── __init__.py
│   │   ├── app.py                        # Application factory, lifespan, middleware, router mounts
│   │   ├── backend.Dockerfile
│   │   ├── constants.py                  # Project-wide constants and paths
│   │   │
│   │   ├── ai/                           # AI orchestration layer
│   │   │   ├── __init__.py
│   │   │   │
│   │   │   ├── langchain/                # LangChain tools and vector store
│   │   │   │   ├── __init__.py
│   │   │   │   ├── tools.py              # Agent tools
│   │   │   │   └── vector_db.py          # Chroma vector DB wrapper
│   │   │   │
│   │   │   ├── langgraph/                # LangGraph agent pipeline
│   │   │   │   ├── __init__.py
│   │   │   │   ├── graph_builder.py      # Graph compilation, LocalMindAI class
│   │   │   │   ├── graph_configs.py      # Reasoning mode width configurations
│   │   │   │   ├── graph_nodes.py        # Node implementations
│   │   │   │   ├── graph_states.py       # AgentState and Candidate dataclasses
│   │   │   │   └── graph_utils.py        # LLM factory, seeding, token logging
│   │   │   │
│   │   │   ├── llms/                     # Local model management
│   │   │   │   ├── __init__.py
│   │   │   │   ├── huggingface.py        # HF API browsing, quant listing, GGUF downloading
│   │   │   │   ├── model_manager.py      # Local file scanning, llama-swap config generation
│   │   │   │   └── cpp_models/           # Downloaded GGUF model files
│   │   │   │
│   │   │   └── system_prompt/            # Personality prompt templates
│   │   │       ├── __init__.py
│   │   │       ├── instruction.py        # Prompt builder with personality injection
│   │   │       └── system_instructions.json  # Personality definitions
│   │   │
│   │   ├── database/                     # Data access layer
│   │   │   ├── __init__.py
│   │   │   ├── crud.py                   # Async CRUD operations for all entities
│   │   │   ├── exceptions.py             # Custom domain exceptions
│   │   │   ├── models.py                 # SQLAlchemy ORM models
│   │   │   ├── schemas.py                # Pydantic request/response schemas
│   │   │   └── session.py                # Async session factory and DB init
│   │   │
│   │   ├── routes/                       # API route handlers
│   │   │   ├── __init__.py
│   │   │   ├── chat.py                   # POST /chat — agent inference endpoint
│   │   │   ├── conversations.py          # CRUD routes for conversations and messages
│   │   │   ├── dependencies.py           # FastAPI dependency injection
│   │   │   ├── models.py                 # Model browsing, downloading, and management routes
│   │   │   └── uploads.py                # File upload, ingestion, and attachment routes
│   │   │
│   │   └── services/                     # Business logic layer
│   │       ├── attachment_services.py    # File upload orchestration and vector DB ingestion
│   │       ├── conversation_services.py  # Chat history, summarization, and message management
│   │       └── model_services.py         # Model install, sync, deletion, and llama-swap config
│   │
│   └── frontend/                         # Vanilla SPA frontend
│       ├── index.html                    # Entry point HTML
│       ├── app.js                        # Application logic, state management, rendering
│       ├── app.css                       # Liquid glass UI styles
│       └── frontend.Dockerfile
```

---

## References

1. Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
2. Universal Self-Consistency for Large Language Model Generation
3. Scalable Best-of-N Selection for Large Language Models via Self-Certainty

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
