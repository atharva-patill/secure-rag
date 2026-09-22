# 02 — Dataset

**Original reports:** `reports/dataset/validation_report.md`, `reports/dataset/distribution_statistics.md`, `data/dataset_architecture.md`, `data/generate_dataset.py`

## 1. Experiment Objective
Describe the synthetic hospital dataset that underpins all retrieval/generation experiments.

## 2. Dataset / Query Population
- **Records:** 120 patient records, deterministic seed 42, blank-line separated in `data/sample_patient_data.txt` (ignored via `.gitignore` as sensitive synthetic data; reproducible via `python3 data/generate_dataset.py`).
- **Ground truth queries:** 605 total in `benchmarks/retrieval/ground_truth_v2.json` (ignored but on disk; regenerable via `benchmarks/retrieval/ground_truth.py`).
  - 601 single-target (1 relevant record each; 240 record_retrieval + 360 entity_retrieval)
  - 4 true multi-record aggregates (>1 relevant)
  - 1 single-target aggregate (T2D+Hypertension, 1 relevant)
- **Multi-record aggregates (5):**
  - AGG_HYPERTENSION 16, AGG_AMLODIPINE_5MG 17, AGG_PARACETAMOL_650MG 20, AGG_METFORMIN_500MG 7, AGG_T2D_HYPERTENSION 1

## 3. Method
- **Architecture:** Shared domain model — Disease Library (33 diseases, long-tail weights), Treatment Library (107 treatments, 2–4 plans/disease), Medication Library (shared across diseases), Hospital Library (57 hospitals weighted), Doctor Roster (24 names) → Patient Generator (120 patients). See `data/dataset_architecture.md`.
- **Record format:** Per-patient text block with demographics, Diagnosis:, Treatment:, Notes, Contact, Aadhaar/PAN/MRN, etc., in masker-compatible formats.
- **Determinism:** `random.seed(42)` + `Faker.seed(42)`; exactly 120 primary-diagnosis slots allocated proportional to weights then shuffled.

## 4. Experimental Controls
- Seed fixed; generation is reproducible; reports generated 2026-08-06.

## 5. Metrics
- Disease frequency, treatment frequency, gender split, average diagnoses/patient, etc. (distribution report).

## 6. Main Numerical Results
- Total patients 120, unique diseases 33, unique treatments 107, avg diagnoses 1.77, avg treatments 3.46, gender 58M/62F.
- Top diseases: Hypertension 16, T2D 15, Viral Fever 12, Hyperlipidemia 10, Asthma/Migraine/Sinusitis 9 each.
- Top treatments: Hydration 21, Paracetamol 650mg 20, Amlodipine 5mg 17, Lifestyle modification 16.
- 1:1 chunk/record (Words 51–83 <500), so `chunk_record ≡ chunk_text`, 120 chunks → 120 embeddings.

## 7. Interpretation
Dataset is intentionally not a lookup table; recurring diseases/treatments/hospitals/doctors create realistic retrieval ambiguity for aggregate queries.

## 8. Limitations
- Synthetic Indian PHI; not real clinical data; not for clinical use.
- Templated per-record queries include generic summary/PHI queries with low discriminative power.

## 9. Artifact / Reproduction Path
- `data/generate_dataset.py` → `data/sample_patient_data.txt` + `reports/dataset/validation_report.md` + `reports/dataset/distribution_statistics.md`
- `benchmarks/retrieval/ground_truth.py` → `benchmarks/retrieval/ground_truth_v2.json` (605 queries, validated)
- Commands: `python3 data/generate_dataset.py`, `python3 -m benchmarks.retrieval.ground_truth`
