# NAC pCR Benchmark — Analysis Code

Analysis and figure-generation code for the study:

> Benchmarking pretreatment DCE-MRI, biopsy histopathology, and routine clinical variables for predicting
> pathological complete response after neoadjuvant chemotherapy in breast cancer: a retrospective study

## Contents

- `code/` — Python scripts that reproduce the tables and figures:
  - `01_reproduce_tables_figures.py` — Tables 1-3 and Figures 3-5, S1 from the derived out-of-fold predictions.
  - `02_reproduce_clinical_nested_cv.py` — nested cross-validation for the clinical reference model.
  - `03_render_workflow_architecture.py` — renders the workflow (Figure 1) and model architecture (Figure 2).
  - `04_privacy_and_checksum_check.py` — privacy/identifier scan and checksum verification.
  - `05_make_extended_visualizations.py` — extended exploratory visualization suite (EV01-EV18).
- `requirements.txt`, `requirements-training.txt` — Python dependencies.

## What is NOT included

No patient data, no raw DICOM/NIfTI/WSI images, and no model checkpoints. To protect participant privacy,
the scripts operate only on de-identified derived tables.

## How to run

1. `pip install -r requirements.txt`
2. Place the de-identified derived input files in the repository root:
   `analysis_dataset_deidentified.csv`, `out_of_fold_predictions.csv`, `per_fold_metrics.csv`,
   `outer_fold_assignments.csv`, `pairwise_auc_comparisons.csv`, `her2_coding_audit.csv`,
   `reported_values.json` (see the manuscript's Data Availability statement).
3. Run from the repository root, e.g. `python code/01_reproduce_tables_figures.py`.

## Data availability

De-identified data are available as described in the manuscript's Data Availability statement. No direct
patient identifiers are included in this repository.

## License

See the article / repository license file.
