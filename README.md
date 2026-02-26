# Grammar-Constrained (CFL) Reachability - Empirical Schema Analysis

This repository contains the empirical part of the manuscript:
"Grammar-Constrained (CFL) Reachability: Subcubic Preprocessing, Indexing Trade-offs, and Structured Decoding Semantics".

It focuses on JSON schema analysis: JSONSchemaBench loading, JSON Schema -> CFG conversion, and linear vs general grammar classification.

## What is included

- `schema_cfg_analyzer.py`: main analysis pipeline
- `plot_schema_analysis.py`: matplotlib plotting script
- `schema_analysis.csv`: per-schema metrics (generated)
- `class_distribution.png`: class distribution figure (generated)
- `schema_size_vs_is_linear.png`: schema-size vs class figure (generated)

## Why this classification looks this way

- Object properties are modeled as chained productions (linear-friendly).
- Variable-length arrays use recursive list productions (`ITEMS -> Item | Item ',' ITEMS`), which introduce non-linearity.

## Quick start

Install:

```bash
python3 -m pip install datasets jsonschema huggingface_hub matplotlib
```

Run analysis:

```bash
python3 schema_cfg_analyzer.py
```

If auth is required:

```bash
huggingface-cli login
```

or

```bash
HF_TOKEN=your_token_here python3 schema_cfg_analyzer.py
```

Generate figures:

```bash
python3 plot_schema_analysis.py
```

## Outputs

Console tables:

- Table A: linear vs general by dataset
- Table B: non-linearity sources (`array`, `nested object`, `$ref`)
- Table C: `|P|` and `|N|` distribution stats
- Table D: LaTeX-ready summary table
- Table E: schema size vs grammar class summary

CSV columns in `schema_analysis.csv`:

- `dataset`, `schema_id`, `num_productions`, `num_nonterminals`, `num_terminals`
- `max_rhs_nt`, `nonlinear_productions`, `is_linear`
- `has_recursion`, `has_array`, `has_nested_object`
- `schema_size`, `error`

## Figures

![Grammar Class Distribution](./class_distribution.png)

![Schema Size vs Grammar Class](./schema_size_vs_is_linear.png)

## Current snapshot (this workspace run)

- Total schemas analyzed: 9,558
- Linear: 801 (8.4%)
- General CFG: 8,757 (91.6%)

These values may change if the upstream dataset snapshot changes.

## Citation

If you use this code or results, cite the manuscript as:

```bibtex
@misc{alpay2026cflreachability,
  title={Grammar-Constrained (CFL) Reachability: Subcubic Preprocessing, Indexing Trade-offs, and Structured Decoding Semantics},
  author={Faruk Alpay and Levent Sarioglu},
  year={2026},
  eprint={xx},
  archivePrefix={arXiv},
  primaryClass={xx},
  url={xx}
}
```

If metadata changes later (arXiv id, class, URL), replace `xx` fields.
