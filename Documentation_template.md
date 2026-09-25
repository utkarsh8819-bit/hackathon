# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity ER Team  
**Team Members:** Utkarsh  
**Submission Date:** September 2026

---

## 1. Executive Summary
We designed and implemented a two-stage, production-quality Entity Resolution (ER) system engineered specifically for the precision-weighted competition metric ($F_{0.5}$, where precision counts $2\times$ over recall). Stage A uses a multi-rule inverted index blocking engine over the combined candidate pool ($S_2 + S_3$) spanning exact names, informative token overlap, Metaphone phonetic keys, structured postal/PIN codes, and character 3-grams to maintain high candidate recall. Stage B extracts a 25-dimensional feature vector capturing non-linear string similarities, address subfield agreements, and open-set country equality, trained using a LightGBM classifier with threshold optimization directly on the Per-Entity Macro $F_{0.5}$ metric (including singleton credit).

---

## 2. Methodology

### 2.1 Problem Analysis
- **Heterogeneous Noise Profiles:** Business records from three sources ($S_1, S_2, S_3$) lack unified identifiers and exhibit severe textual noise: legal suffix variations (`LLC`, `Inc`, `Pvt Ltd`, `Limited`), embedded commas inside addresses, landmark descriptions (`Near Fortis Hospital`, `Opp SBI ATM`), and distinct address topologies between US (street/city/state/zip) and India (building/plot/area/PIN).
- **Open-Set Country Distribution:** While training records contain only `US` and `India`, test records introduce an unseen third country label (`France`). The pipeline was architected to treat country strictly as an open-set string feature with zero hardcoded enumerations or dictionary filters.
- **One-to-Many and Singleton Matches:** Source 1 entities can legitimately match zero (singletons), one, or multiple candidate records. The competition metric heavily rewards predicting true singletons with empty strings (score = 1.0) while heavily penalizing false merges on singletons (score = 0.0).

### 2.2 Solution Strategy
- **Approach Type:** Two-Stage Blocking + Supervised Tree Classifier (LightGBM) with Macro $F_{0.5}$ Threshold Sweeper.
- **Core Innovation:**
  1. *Dual-Stage Architectural Decoupling:* Candidate pairs are materialized and exported prior to any ML scoring, ensuring a clear audit trail and measurable recall ceiling.
  2. *Structured Subfield Parsing:* Rule-based regex extractors isolate 6-digit Indian PIN codes, 5-digit US/French postal codes, building/plot numbers, and landmark clauses without external geocoding or API lookups.
  3. *Grouped Disjoint Validation:* All candidate pairs for any given $S_1$ entity are strictly contained in either the train or validation partition, eliminating information leakage.
  4. *Direct Metric Optimization:* The classification decision threshold is swept over held-out validation data to directly maximize the official per-entity macro $F_{0.5}$ score.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. Exact normalized name (NFKD Unicode, lowercase, whitespace collapsed).
  2. Informative name token overlap (filtering generic legal and stopwords).
  3. Phonetic key matching using Metaphone.
  4. Structured postal code / PIN code agreement.
  5. Distinctive address token overlap.
  6. Character 3-gram signatures for typo resilience.
- **Candidate pairs generated:** Controlled via inverted index query cap (`max_candidates_per_s1 = 100`) prioritizing multi-rule agreement.
- **How true matches were not lost:**
  Candidate retrieval takes the union of all independent blocking keys. A candidate need only satisfy any one of the complementary rules to be retrieved. On validation benchmarks, this achieves near 100% blocking recall ceiling.

---

## 4. Matching Model

**Features used:**
- **Name Features:** Levenshtein edit similarity (RapidFuzz), Jaro-Winkler distance, token-set similarity, token-sort similarity, character 3-gram cosine similarity, and Metaphone phonetic similarity.
- **Address Features:** Normalized address edit ratio, distinctive address token Jaccard similarity, address token-set ratio, character 3-gram cosine similarity.
- **Structured Agreement Features:** Same postal/PIN code indicator (1 = match, 0 = mismatch, -1 = missing), same house/building number indicator (1/0/-1), open-set country equality (1/0/0.5).
- **Discrepancy & Blocking Features:** Absolute and relative length differences for name and address; multi-hot indicators for all 6 blocking rules and total count of triggered blocking rules.

**Model type:** LightGBM Binary Classifier (`class_weight='balanced'`, `n_estimators=120`, `learning_rate=0.05`, `num_leaves=31`). Tree models effectively capture feature interactions (e.g., high name similarity + matching postal code vs. high name similarity + conflicting postal code).  
**Threshold selection method:** Empirical threshold sweep across [0.05, 0.95] on the grouped validation set, maximizing the official competition formula:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
macro-averaged across all Source 1 entities with singleton credit.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Validation Macro $F_{0.5}$ evaluated on held-out grouped validation entities.
- **Common false positives (wrong merges):** Franchise branches or multiple businesses sharing identical commercial complexes / zip codes with similar generic business keywords (mitigated by distinguishing building numbers and phonetic tokens).
- **Common false negatives (missed matches):** Drastic acronyms or DBA trade names differing completely from registered corporate names (e.g. "Apex Logistics LLC" vs "ALLC Transport").

---

## 6. Conclusion
The developed system demonstrates that a modular, two-stage architecture combining multi-rule inverted index blocking with gradient boosted decision trees and precision-calibrated thresholding delivers high recall candidate retrieval and precision-heavy resolution. The pipeline is fully compliant with all competition rules, self-contained, and handles open-set country shifts cleanly.

---

## Appendix

### A. Code Artefacts
All code resides under `src/` with pinned `requirements.txt` and reproduction steps in `README.md`.
- `run_pipeline.py`: Full end-to-end execution and packaging.
- `src/train.py`: Model training and threshold selection.
- `src/inference.py`: Test inference and TSV generation.
- `src/blocking.py`: Multi-rule candidate generation.
- `src/features.py`: Pairwise feature extraction.
- `utils/validate_submission.py`: Auto-validation script.

### B. Additional Results
Interactive diagnostic workflows and error bucketing reports are provided in `notebooks/` (`01_eda.ipynb`, `02_blocking_analysis.ipynb`, `03_model_analysis.ipynb`).
