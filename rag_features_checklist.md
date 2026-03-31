# Privacy-Aware RAG Project Checklist

## Core Package
- [x] Python package structure (`secure_rag/`)
- [x] Public Python API export (`build_rag`, `rag_answer`)
- [x] CLI entry point via `pyproject.toml`
- [x] Local distribution artifacts (`dist/` wheel and sdist)
- [x] Editable package install in local virtual environment
- [x] Automated release workflow
- [x] CI for install, test, and build validation

---

## Core RAG System
- [x] Embedding model integration
- [x] Lazy in-memory embedding model reuse
- [x] FAISS vector store
- [x] Retrieval pipeline
- [x] LLM generation
- [x] Prompt grounding
- [x] Streaming token generation
- [x] Reusable vector store object per chat session
- [x] Persistent vector index on disk(prototype level)
- [ ] Metadata-aware retrieval
- [ ] Source citation retrieval


---

## Document Ingestion
- [x] `.txt` ingestion
- [x] `.pdf` ingestion
- [x] File path validation
- [x] Unsupported file type validation
- [x] Empty chunk validation
- [ ] Directory ingestion
- [ ] Automatic indexing pipeline
- [ ] Medical report-specific ingestion

---

## Chunking and Retrieval Quality
- [x] Basic text chunking
- [x] Chunk overlap
- [ ] Semantic chunking
- [ ] Hybrid search (keyword + vector)
- [ ] Retrieval reranking model
- [ ] Context filtering
- [ ] Query rewriting
- [ ] Context compression
- [ ] Conversation memory

---

## Privacy Layer
- [x] Regex-based email masking
- [x] Regex-based phone masking
- [x] Query masking before retrieval
- [x] Retrieved context masking before generation
- [ ] Named Entity Recognition (NER) masking
- [ ] Medical entity masking
- [ ] Reversible masking tokens
- [ ] Masking evaluation metrics
- [ ] PHI coverage beyond email and phone

---

## Model and Generation Layer
- [x] Lazy API client initialization
- [x] Environment-based API key loading
- [x] Hugging Face Router endpoint support
- [x] Configurable model via environment variable
- [x] Streaming generation response handling
- [x] Streaming tokens generations
- [ ] Local/offline generation option
- [ ] Answer citation display
- [ ] Confidence or groundedness scoring

---

## Interfaces
- [x] Packaged CLI chat interface
- [x] Streamlit chat UI scaffold
- [x] FastAPI app scaffold
- [ ] FastAPI endpoint correctly wired to packaged RAG pipeline
- [ ] Streamlit document upload
- [ ] Chat history
- [ ] Source citation display in UI

---

## Logging and Observability
- [x] Basic query/context logging hooks
- [ ] Production logging configuration
- [ ] Sensitive-data-safe logging policy
- [ ] Metrics collection for latency and quality

---

## Testing and Validation
- [x] Manual test/demo scripts
- [x] Automated unit tests
- [ ] Automated integration tests
- [x] Mocked tests for generation layer
- [ ] Regression test coverage
- [ ] Benchmark suite

---

## Packaging and Distribution
- [x] `pyproject.toml` package metadata
- [x] Console script registration
- [x] README with install and usage examples
- [x] TestPyPI publish ready
- [ ] TestPyPI publish flow
- [ ] PyPI publish flow
- [x] Twine/build verification in clean environment

---

## Research and Evaluation
- [ ] Retrieval accuracy metrics
- [ ] Hallucination detection
- [ ] Masking accuracy evaluation
- [ ] Response quality evaluation
- [ ] Latency benchmarking
- [ ] Ablation studies
- [ ] Dataset documentation
- [ ] Reproducible experiment scripts
- [ ] Paper manuscript source

---

## Medical Readiness
- [ ] Presidio PII detection
- [ ] HIPAA entity masking
- [ ] Differential privacy experiments
- [ ] Secure embedding storage
- [ ] Medical terminology-aware masking
- [ ] PHI leakage evaluation
- [ ] Human review workflow
- [ ] Non-diagnostic safety guardrails
- [ ] Audit logging
- [ ] API authentication
- [ ] Local-only inference option for sensitive deployments

---

## Deployment
- [ ] Docker container
- [ ] Cloud deployment
- [ ] Environment-specific configuration
- [ ] Secrets management hardening
- [ ] Deployment documentation
