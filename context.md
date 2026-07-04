# Engineering Context

## RAG Pipeline Decisions

- **Record-based chunking is intentional.** `build_rag()` now splits the loaded document into individual records before masking and chunking. This prevents multiple patient records from being merged into one retrieval unit, which previously caused cross-patient contamination in retrieval results.
- **Pre-masking happens per record before embedding.** In `pre` mode, each record is masked independently before chunking, embedding, and FAISS indexing. This preserves the core privacy guarantee that raw sensitive data never enters the vector store.
- **Queries are never masked.** This is a deliberate rule across all privacy modes. `raw` and `post` share the same raw index; `post` masks only retrieved context before generation.

## Privacy and Masking Fixes

- **Medical ID masking was expanded to cover compact formats.** The masking logic was updated to catch values like `MRN1002`, `MRN 1002`, `MRN:1002`, and `UHID-12345`. Earlier masking missed compact medical ID formats and leaked them into stored or retrieved text.
- **End-to-end masking coverage was validated for major PHI fields.** The current masking flow was verified against names, phone numbers, PAN, Aadhaar, email, DOB-style fields, and medical IDs before embedding in `pre` mode.

## Generation and CLI Behavior

- **Prompt-echo truncation is required in `rag_answer()`.** Model output is post-processed to stop at prompt-structure markers such as `\nContext:`, `\nQuestion:`, and `[/INST]`. This prevents prompt echo from being returned to the user when chat models repeat scaffold text.
- **Rich markup must be disabled for streamed model output.** `console.print(token, end="", markup=False)` is required when printing LLM responses in the CLI. Rich interprets bracketed tokens such as `[/ASSIST]` as formatting markup and raises `MarkupError` if markup parsing is left enabled.

## Dataset and Evaluation Decisions

- **The Compose/Docker sample document lives under `data/sample_patient_data.txt`.** Docker workflows assume this file exists and mount `./data` to `/data` inside the container.
- **The benchmark/test dataset was expanded to full patient-style records.** The project moved to a larger structured sample set with consistent patient fields so privacy and retrieval behavior can be exercised on more realistic records instead of underspecified samples.
- **The main research result is the privacy-utility tradeoff.** `pre` masking improves storage privacy but degrades identity-based retrieval because raw entity names in queries do not align with masked entities in indexed chunks.

## Packaging and Release Decisions

- **Current package version is `0.2.0a1`.** The project is positioned as an experimental research alpha and not a production-ready package.
- **TestPyPI is the intended package release target at this stage.** The project was assessed as appropriate for GitHub/TestPyPI distribution, but not for claiming production readiness on PyPI.

## Docker Runtime Decisions

- **The runtime image uses `Dockerfile.runtime` and `python:3.11-slim`.** The image installs the package from `pyproject.toml`, uses a multi-stage build, runs as a non-root user, and sets `secure-rag` as the entrypoint.
- **CPU-only PyTorch must be installed explicitly in Docker.** On Linux ARM64, default `torch` wheels pull large NVIDIA CUDA dependency wheels even for CPU-only usage. The runtime image installs `torch` from `https://download.pytorch.org/whl/cpu` first, then installs the project normally so `sentence-transformers` reuses the CPU-only install.
- **The spaCy model is baked into the Docker image.** `en_core_web_sm` is installed during image build so container users do not need a separate runtime setup step.
- **The container user must have a real home directory.** Creating `appuser` with `--no-create-home` caused `PermissionError: '/home/appuser'` during Hugging Face cache initialization. The validated fix is to create the user with a writable home directory so `~/.cache` can be created normally.

## Docker Compose Decisions

- **Compose passes the document path as a CLI argument.** The `secure-rag` service runs `secure-rag /data/${RAG_INPUT_FILE}` and mounts `./data` to `/data`. The selected input file is controlled through `RAG_INPUT_FILE` in `.env` / `.env.example`.
- **`docker compose run --rm secure-rag` is the preferred interactive workflow.** It attaches cleanly to the terminal for the chat-style CLI. `docker compose up` is still useful for running the full stack, especially when optional services like Ollama are involved.
- **Ollama support is optional and profile-gated in Compose.** The `ollama` service exists behind a Compose profile and is not started by default.

## CI and Registry Decisions

- **Python CI and Docker CI have intentionally separate responsibilities.**
  - Python CI validates package installation, tests, build artifacts, and metadata.
  - Docker CI validates container build health, CLI startup, and mounted sample data access.
- **Docker CI is intentionally lightweight.** It builds `Dockerfile.runtime`, runs `secure-rag --help`, and verifies `/data/sample_patient_data.txt` is mount-accessible. It does not run inference, benchmarks, or external LLM calls.
- **Docker layer caching was intentionally not added to Docker CI.** For this alpha-stage project, a clean build was considered preferable to extra workflow complexity and cache-management overhead.
- **GHCR publishing is release-driven.** The publish workflow triggers only on `release: published`, keeping distribution intentional and avoiding image publishes on ordinary pushes.
- **GHCR image names must be normalized to lowercase.** `github.repository` preserves GitHub casing, but GHCR rejects uppercase repository names. The workflow lowercases `GITHUB_REPOSITORY` in a shell step and uses that normalized value for `latest` and release-version tags.

## Known Future Work

- **Identity-based retrieval in `pre` mode remains an open limitation.** This is a design-level tradeoff rather than a bug fix completed in this session.
- **Generation quality still depends heavily on the chosen backend model.** Prompt-echo handling is improved, but model behavior such as outside-knowledge drift remains a broader system limitation.
