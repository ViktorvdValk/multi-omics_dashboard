# Skin multi-omics dashboard (studies 2230, 1947, 1806)

Skin microbiome, gut microbiome, energy metabolism and ceramides in one Streamlit app, plus an
integrated page that compares and combines their ordination spaces. A study does not need to have
every modality.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open **Data setup**. By default it scans the data drive (`Z:\`, see below) and lists every data file;
click **Load** behind a file (or **Load all**). **Manual loading (paths or uploads)** switches to the
old form with paths and **Load data**; "Try with synthetic demo data" there loads a made-up data set
(also: `python make_demo_data.py demo_data`).

## Automatic loading from the data drive

The drive (default `Z:\`; change it on Data setup or set `MULTIOMICS_DRIVE`) holds one folder per study,
named `CHDR<study number>...` (e.g. `CHDR1806_SKINERGY_PSORIASIS`, `CHDR2230_SKINERGY_CTCL`); the study
number is taken from the folder name. Each study folder has

```
CHDR1806_SKINERGY_PSORIASIS/
  metadata/                  every metadata file (.csv/.txt/.tsv/.xlsx) is read
  processed/<modality>/...   data files, also in subfolders (up to 4 levels)
```

- **Every data file is listed** (Excel: every sheet), with the modality recognised from its **content**:
  a Biotaviz tree (`#class`, `class id`, rows `phylum - Bacillota` ...) or a `.biom` file = microbiome;
  succinate / fumarate / malate / isocitrate columns = energy metabolism; a ceramide column or ceramide
  names (`Cer[NS]`, `Cer 34:1;O2` ...) = ceramides. The folder name is not used, so the modality folders
  can have any name. Change the modality or part in the list if it is wrong; "skip" files are not loaded.
- **Skin or gut** is taken from the metadata `site` of every sample (the sample columns of a Biotaviz
  file are metadata `sample_id`s). The file name (`skin`, `swab`, `gut`, `faeces`, `FC`, `stool` ...) is
  only a check: a sample whose metadata site disagrees with its file name is flagged. Energy: "blister"
  or "plasma" from the file name (blister by default); change it in the list if needed.
- **Biotaviz**: only the rows of the chosen taxonomic rank are used (higher rows are sums); reads are
  divided by the root total, and the reads of the root not assigned at that rank become "Unclassified".
- **Nothing is loaded until you click Load.** Loaded files stay loaded when you switch pages; changing
  a file on the drive (or the rank, modality or part) reloads after the next click / **Rescan**.
- **Metadata of the loaded files.** When you click Load, the metadata files that belong to the loaded data
  files are combined into one table (`combined_metadata.csv`, downloadable; removed again when nothing is
  loaded): every metadata file of their own study folders, plus a metadata file in another study folder
  that holds sample ids of a loaded file (e.g. a shared sequencing run). Data setup > "Metadata files"
  shows which files are used, per data file. Rows are combined on `study + sample_id` (`LCAB_ID` also
  counts as sample id; rows without a sample id: study + subject + site + lesion status + timepoint). A
  file can add rows and fill blanks of another file. Where files disagree on a value, **the file in the
  study's own CHDR folder wins** (if those files agree among themselves); otherwise the value is left
  missing. Rows without a `study` column get the study of their folder. A patient written differently in
  two metadata files of a study (`200_AN` in the energy metadata, `200` in the clinical metadata) is linked
  by number to the spelling of the file that has the group (listed as "patient id").
- **Microbiome sample names**: the Biotaviz sample columns are matched to the metadata `sample_id`; if
  they are not there, to the metadata column that holds them (e.g. a sequencing id column). Skin or gut
  comes from the metadata `site` (`skin` / `gut`), else from words in it or in `sample source` (swab,
  faeces ...), else from the part chosen in the list.
- **Energy metabolism sites** come from the metadata column `sample source`, matched by `LCAB_ID`:
  `blister perilesional` = lesional, `blister non-lesional` = non-lesional, `blister` = healthy volunteer,
  `plasma` = plasma. One file with both blister and plasma rows is split by it. A `Site` column in the
  data file itself is still used if it is there. A blister without a lesion score, skin type or body site
  in its own metadata row takes them from the skin row of that site and visit.
- **Discrepancies** are listed after loading (Data setup > "Metadata discrepancies", filterable,
  downloadable as CSV): values that differ between metadata files, group / stage that differ within a
  patient, global severity that differs within one visit, patient ids linked across files, microbiome
  samples whose study folder or file name contradicts the metadata, and samples in two files (the first
  file is used).
- Files that need extra packages: `.xls` needs `xlrd`, `.biom` needs `biom-format`. Emu per-sample TSV
  folders are not listed; use manual loading for those.

Default paths can be set in `views/data_setup.py` (`DEFAULTS`) or with the environment variables
`MULTIOMICS_DATA_ROOT`, `MULTIOMICS_METADATA` (the metadata file; `MULTIOMICS_CLINICAL` still works),
`MULTIOMICS_CERAMIDES`, `MULTIOMICS_ENERGY_BLISTER`, `MULTIOMICS_ENERGY_PLASMA`.

## Input files

| Modality | Source |
|---|---|
| Metadata (all modalities) | Drive: all files in the `metadata/` folders, combined (see above). Manual: ONE metadata file, `meta_data_all_complete.csv` (blank: the one in the sequencing folder) |
| Skin and gut microbiome | Drive: Biotaviz Excel / text files or `.biom` files. Manual: filtered Emu tables plus `meta_data_all3_new.csv`; or BIOM files (see below) |
| Ceramides | Long table (`study, subject, occ, location, ceramide, value`) or wide (one column per ceramide) |
| Energy metabolism (1806) | Blister fluid and plasma: `LCAB_ID, Subj_ID, Sampling_time, Succinate, Fumarate, Malate, Isocitrate`. Drive: the site from the metadata `sample source` (by `LCAB_ID`). Manual: the blister file has `Site` (lesional / non-lesional) |

Text files may be tab-, semicolon- or comma-separated, with decimal commas or points; Excel also works.
Column names are matched through the alias list in `datasets.py` (`COLUMN_ALIASES`).

## Metadata: the single source for every dataset

Group, stage / severity, PASI, global severity, local severity, skin type and body site come **only** from
the metadata file, for every dataset. Columns like these in a ceramide or energy file (`group`, `stage`,
`pasi`, `global_severity`, `local_severity`, `region`, `region_location` ...) are ignored; the loading
notes list them. The microbiome samples are rows of the metadata file. A sample of another dataset is
matched to a metadata row:

1. **by sample id**: the table's `sample_id` / `LCAB_ID` / `sample` column equals a `sample_id` of the
   metadata file (same study), or else
2. **by study + patient + site + timepoint**, and for a patient with several lesions at that visit also
   by **lesion number**: lesion 1 (code X1 / L1, or a lone lesion code) is the most severe lesion of that
   visit in the metadata file (highest `local_severity`), lesion 2 (X2 / L2) the less severe one. A
   lesional sample without a lesion code, of a patient with several lesions at that visit, gets no local
   severity (noted).

Patient-level fields (group, severity, PASI) need only the patient; the global severity is the value of
that visit (study + patient + timepoint) in the metadata file. The timepoint of a metadata row is the
mapping of the microbiome timepoint labels (same file); labels no microbiome sample uses appear in the
timepoint table as "Metadata file". Data setup > "Check every sample against the metadata file" shows,
for every sample, how it was matched and which metadata row it got.

## Microbiome from BIOM files

Data setup > "Read the microbiome from" chooses the input; the default is unchanged (filtered Emu TSVs).

- **BIOM files**: every `*.biom` file in the BIOM folder and its subfolders (blank: the `biom` folder in
  the sequencing data folder, or else the sequencing data folder itself; one `.biom` file also works),
  e.g. the per study x site files of `split_biom.py` (`emu_1806_gut.biom`, `emu_1947_skin.biom` ...). The
  study and site of a BIOM sample are read from its sample metadata (`study`, `site`, written by
  `split_biom.py`) or else from the file name, and a metadata sample is only matched to BIOM samples of its
  own study and site, so the same sample or file name (e.g. a barcode) may occur in several studies. The metadata is still the sequencing
  metadata file; a `sample_id` is matched to a BIOM sample by its BIOM sample ID or by the Emu file it
  came from (with or without folder, `.tsv` and `_rel-abundance...` suffix). Taxa are named at the chosen
  rank from the BIOM taxonomy (`k__ ... s__`); unassigned reads are left out, as with the TSVs.
- The BIOM files are used as they are: every taxon in them is analysed (the `keep_for_analysis` column
  only applies to the filtered TSVs). To analyse a filtered set from BIOM, make the BIOM file from
  filtered data.
- **Both**: the filtered TSVs first; the BIOM files add samples that have no TSV. For samples in both,
  the loading notes give the largest difference in relative abundance, a quick check that the two
  inputs agree.
- Command line: `python load_studies.py --source biom --biom-dir /path/to/biom`. Environment variables:
  `MULTIOMICS_MICROBIOME_SOURCE` (`tsv`, `biom`, `both`) and `MULTIOMICS_BIOM_DIR`.
- Needs `pip install biom-format h5py` (only for BIOM input).

## How samples are matched

Samples of different modalities are matched on **study + patient + timepoint**. Sites (lesional,
non-lesional, healthy, nostril, gut, plasma) are never averaged: each is its own view.

- **Healthy volunteers** have the site `healthy` for their skin samples (and blister fluid), in every
  dataset. They are the healthy reference in the lesional and in the non-lesional comparisons (and in
  pooled). When the label says which patient site a healthy sample matches (HV_TL / HS = target site,
  HV_NL = non-lesional) it is only used for that site. A volunteer's nostril, gut and plasma samples
  keep those sites.

- **Patient.** Ids are trimmed and `12.0` / `012` become `12`. When a table writes the subject
  differently from the metadata file (`53_AN` vs `53`), the rule that matches most patients of a
  study is used and reported under "Loading notes" on the Data setup page. Check that note.
- **Timepoint.** Every file keeps its own labels. On the Data setup page each label is assigned to
  `baseline`, `mid` or `follow-up`. The proposal takes the earliest label of a study as baseline and
  the latest as follow-up; a blank or `nvt` label is baseline. Edit the table where the proposal is
  wrong (a file with only two of three visits cannot be read automatically) and click
  "Apply timepoint mapping". Your choices are saved in `timepoint_map.csv` next to `app.py`.
- **Group, severity, PASI** come from the metadata file (see above); a patient who is not in it is
  "unspecified".

## Zero and missing concentrations

A zero is read as "below the detection limit", not as a concentration of zero. By default zeros and
missing values are left out: every ceramide or metabolite is analysed on the samples that have a
value, and whole-profile distances use the features two samples have in common. The sidebar of the
lipidomics and energy pages offers imputation instead: half of the smallest measured value (for values
missing because they are low) or k-nearest neighbours (for values missing by chance). The choice also
applies on the Integrated page. Ceramide class fractions and the mean carbon number always count a
non-detected ceramide as zero.

## Ceramide names

Class and carbon number are read from the name (`NS_C34`, `CER[NP] C44`, `Cer(d18:1/24:0)`). The carbon
number is used as written; it must mean the same thing for every ceramide in the file.

## Pages

- **Skin microbiome / Gut microbiome**: composition, alpha diversity, PCoA, single taxa, and the
  permutation statistics of `skin_engine.py`.
- **Lipidomics (ceramides)**: class composition, chain-length distribution, single ceramides, PCA,
  and the same statistics on ln concentrations and on summary indices (class fractions, mean carbon
  number).
- **Energy metabolism**: the same per metabolite, separately for lesional blister fluid, non-lesional
  blister fluid and plasma, plus ratios. Blister fluid is taken at a lesional (perilesional) and a
  non-lesional site of every patient (the `Site` column of the blister file: L / NL, lesional /
  perilesional / non-lesional ...), compared within patient (Compare by: site, and the paired plot).
  Healthy volunteers have one blister without a site label: site `healthy`, the reference for both.
  A patient's blister without a label is kept apart as "blister (patient, no site label)". Without a
  `Site` column all blisters stay one site, "blister".
  Ratios you define (for example succinate / fumarate) are added as summary indices.
- **Integrated analysis**: ordinations of several spaces side by side, coloured by any metadata
  variable or by a variable of another modality; Procrustes and Mantel tests between spaces; a
  combined space of several modalities that can be saved and compared with the others.
- **Subset explorer**: pick the samples of one dataset, narrow them down with criteria (for example
  local severity = 8, or disease group is MF), then (1) plot that dataset as an ordination of the
  selected samples coloured by any variable, or (2) plot two variables against each other. A variable
  can be metadata or a taxon / ceramide / metabolite of any dataset (relative abundance, CLR or
  present yes / no); a variable of another dataset is matched on patient and timepoint.

## Severity scores and colours

- **Severity / stage** (text): MF stage, or mild / moderate / severe from PASI. Default colours, blue
  (mildest) to red (most severe); healthy volunteers are always green.
- **Global severity** (number): the `global_severity` column of the metadata file (PASI, mSWAT ...),
  the score of one visit, so baseline, mid and follow-up keep their own values. A sample matched to a
  metadata row (by sample id, or patient + site + timepoint) keeps that row's value. Any other sample of
  that **study, patient and timepoint** gets the visit's value when all rows of the visit agree;
  otherwise it is missing (nothing is averaged or maximised; the loading notes list such visits).
  Healthy volunteers have 0 (and stay green in the plots). When timepoints are averaged per patient,
  the patient's global severity is the mean over those timepoints.
- **Local severity** (number): the `local_severity` of the matched metadata row (see "Metadata: one
  file"), a property of a lesion, so only lesional samples have it.
- **Check:** Data setup > "Check every sample against the metadata file" lists, per sample of every
  dataset, how it was matched, the value in the matched metadata row next to the value the app uses,
  and flags differences. The status line names the metadata file that was read.
- Sidebar, **Plots > Colour points by**: stage (default), local severity, global severity, or both
  (fill = local, outline = global). Scores run blue (lowest) to red (highest) with a colour bar,
  healthy stays green, samples without a value are grey. It applies to the box plots, ordinations and
  paired plots of the page. Studies that use different scales share one colour bar: select one study.

## Files

```
app.py               navigation
discovery.py         drive scan, modality recognition, combining metadata files
views/               one file per page (+ _common.py, _omics.py shared helpers)
datasets.py          common data model, matching of patients / sites / timepoints
load_studies.py      microbiome loader
ceramide_loader.py   ceramide name parsing and summary indices
energy_loader.py     energy-metabolism loader and ratios
skin_engine.py       permutation statistics (microbiome pages; reused by analysis.py)
skin_plots.py        figures of the single-modality pages
analysis.py          statistics and figures for ceramides and metabolites
integration.py       views, distances, combined spaces, Procrustes / Mantel / association tests
explore.py           variables and criteria of the Subset explorer
make_demo_data.py    synthetic demo data
```

## Method notes

- Single-modality pages: all timepoints are selected by default and nothing is mixed: one unit is one
  patient at one site and one timepoint, and comparisons between patients are made per timepoint
  ("Timepoints are: averaged per patient" gives the patient mean instead). Timepoint comparisons are
  paired within patients.
- Healthy volunteers who were sampled at one timepoint only are not tied to that timepoint: their
  sample is the healthy reference at every timepoint, in the plots and in the statistics (so the same
  volunteers appear in the baseline, mid and follow-up comparisons).

  | Compare by | Design |
  |---|---|
  | severity, group, study, skin type | between patients, per site and per timepoint |
  | timepoint | paired within patient, per site |
  | site (lesional vs non-lesional skin or blister fluid, nostril, blister vs plasma) | paired within patient, per timepoint |
  | local severity (lesional samples) | as a score: between patients (most severe lesion) and within patients with two lesions (more vs less severe) |
- A patient with two lesions at a timepoint: the most severe one (highest local severity) is used
  wherever one lesion per patient is needed; "keep both lesions" shows both in the plots.
- Integrated page: one unit is one sample, of one patient at one timepoint, and every site is its own
  view; nothing is averaged. "Patients with two lesions" (sidebar): the most severe lesion only (L1,
  default) or both, with L2 as a point of its own that is matched only to L2 samples of other
  modalities (a visit-level variable such as gut, plasma or non-lesional skin still colours it).
  Permutations move whole patients, so the timepoints and lesions of a patient are never treated as
  independent.
  A variable that changes within patients (timepoint, a metabolite) is tested within patients.
- Combined space: `D² = mean over views of (D_k / s_k)²`, with `s_k` the root-mean-square distance of
  view k, so every view carries the same total variance.
- All p-values are permutation p-values; q-values are Benjamini-Hochberg unless changed in the sidebar.
- Subset explorer: the p-values under the two-variable plot (Spearman, Mann-Whitney, Kruskal-Wallis)
  treat every sample as independent. They are a rough guide when a patient contributes several samples.
- Results are exploratory. Tested on synthetic data only.
