# Project instructions for the coding agent

This repository holds data loaders and a Streamlit dashboard for three combined
clinical studies (2230, 1947, 1806): skin microbiome, gut microbiome, energy
metabolism and ceramides. It runs inside a myDRE workspace with patient data.

## Python environment

- Run Python only with the project environment: `.venv\Scripts\python.exe`
  (tests: `.venv\Scripts\python.exe -m pytest`).
- Do not install packages with pip yourself. If a package is missing, stop and
  say which one; it must be added to `pyproject.toml` by the maintainer.
- The environment is created by `install-hermes.cmd` (it runs `uv sync`).

## Data safety

- Raw data is read-only. Never modify, move, rename or delete files in the data
  folders. Write all output to the `output\` folder in this repository.
- Never copy patient data into the repository, into code, tests or comments.
  Tests use small synthetic example data in `tests\data\`.
- There is no internet access for the agent. Do not try to download anything.

## Data rules for loaders

- All sample metadata (group, stage, local_severity, global_severity, ...) comes
  only from the metadata file, never from ceramide, energy or microbiome files.
- Match data to metadata by sample ID; otherwise by patient, site, timepoint
  (and lesion number).
- Lesional, non-lesional and nostril skin are separate levels. Never average them.
- Read the timepoint from each file's own column: `timepoint` (metadata),
  `Sampling_time` (energy metabolism), `timepoint_code` (ceramides).
- `Subj_ID` values in the energy files look like `58_AU`; match on the number.
- Zero concentrations mean "absent": leave them out, do not impute by default.
- Global severity belongs to one timepoint; never carry it to another timepoint.

## Working style

- Make small changes and run the tests after each one.
- Explain what you changed and why, in plain language: the users are researchers,
  not programmers.
