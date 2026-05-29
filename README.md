# ResistAgent

ResistAgent is an evidence-constrained computational workflow for resistance-aware lead optimization. It organizes mutation priors, structural context, deterministic mechanism diagnosis, and robust counter-design into a site- and combo-aware design loop.

## Scope

This repository contains the core source code only:

- `agents/`: bounded LLM-facing agents that translate deterministic evidence into structured rationales and constrained edit plans.
- `tools/`: reusable utilities for mutation parsing, public-data handling, structural processing, scoring, counter-design, calibration, and benchmark support.
- `scripts/`: execution scripts for data audit, case selection, public-asset retrieval, mutation-prior construction, structural preparation, docking/IFP processing, model calibration, counter-design, and validation summaries.
- `configs/`, `schemas/`, `workflows/`, and `tests/`: configuration templates, validation schemas, the Snakemake entry point, and unit tests.

Generated figures, manuscripts, paper-preparation scripts, raw data, local outputs, and analysis reports are intentionally excluded.

## Requirements

Install the Python dependencies listed in `requirements.txt`. Some optional structural and physics routines require external command-line tools such as AutoDock Vina, PLIP, OpenMM-compatible force-field assets, and local structure-preparation utilities.

LLM-backed routines use an OpenAI-compatible GLM client and read credentials from environment variables:

```bash
export ZHIPU_API_KEY=...
export GLM_MODEL=glm-4.5
export GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

Most deterministic steps can be inspected or tested without live LLM calls.

## Basic Checks

```bash
python -m compileall agents tools scripts tests
pytest tests
```

## Data

The workflow expects public and licensed datasets to be staged locally under paths configured in `configs/base.yaml` and project-specific YAML files. Data files are not redistributed in this code-only repository.
