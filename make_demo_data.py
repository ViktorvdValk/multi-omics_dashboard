#!/usr/bin/env python3
"""
make_demo_data.py -- SYNTHETIC data in the real file layout, to try the dashboard without real data.

    python make_demo_data.py demo_data

Writes
    demo_data/meta_data_all3_new.csv                      sequencing metadata (skin, nostril and gut rows)
    demo_data/CHDR*/output_emu_99_emu_db/<sample>/...     filtered Emu tables
    demo_data/ceramides_long.csv                          studies 1806 and 1947, a subset of the patients
    demo_data/energy_blister.txt, energy_plasma.txt       study 1806: succinate, fumarate, malate, isocitrate
                                                          (tab-separated, decimal comma, Subj_ID like 201_AB)

Made difficult on purpose, like real files: subject ids are written differently per file ('012', 12,
'12_AB'), every file has its own timepoint labels (v1/v2/v3, 1/3, 0/1/2, 'nvt', blank), not every
modality has every visit, and some concentrations are zero or missing.
Every patient has a hidden 'inflammation' score that shifts lesional skin (Staphylococcus up), gut
(Faecalibacterium down), lesional ceramides (lower carbon numbers) and succinate, so the modalities are related.
"""
import os
import sys

import numpy as np
import pandas as pd

SKIN = ["Cutibacterium", "Staphylococcus", "Corynebacterium", "Streptococcus", "Micrococcus", "Moraxella",
        "Dolosigranulum", "Lawsonella", "Acinetobacter", "Pseudomonas"]
GUT = ["Bacteroides", "Faecalibacterium", "Prevotella", "Bifidobacterium", "Blautia", "Roseburia", "Akkermansia",
       "Escherichia", "Ruminococcus", "Lachnospira", "Alistipes", "Streptococcus", "Veillonella", "Enterococcus"]
SKIN_W = {"non-lesional": [8, 3, 3, 1, 1, .3, .3, 1, .5, .3], "lesional": [5, 5, 2, 1, .8, .3, .2, .8, .6, .5],
          "nostril": [1, 4, 6, 1, .5, 2, 4, .3, .3, .2]}
GUT_W = [8, 6, 4, 3, 5, 4, 2, .5, 4, 3, 3, .5, 1, .3]
CER = [(c, n) for c in ("NS", "NdS", "NP", "NH", "AS", "AP") for n in (34, 36, 38, 40, 42, 44, 46, 48)] + \
      [(c, n) for c in ("EOS", "EOP", "EOH") for n in (66, 68, 70)]
METAB = {"Succinate": (1.3, .45), "Fumarate": (.35, .25), "Malate": (1.6, .1), "Isocitrate": (180., -.1)}
SHARED = "CHDR1806_SK_FC_and_CHDR1947_FC"
INITIALS = ["AN", "AQ", "BK", "CD", "EF", "GH", "JL", "MP", "RS", "TV", "WX", "YZ"]


def main(out):
    rng = np.random.default_rng(11)
    os.makedirs(out, exist_ok=True)
    meta, cer, energy = [], [], {"blister": [], "plasma": []}

    def write_emu(folder, sample, genera, x):
        p = os.path.join(out, folder, "output_emu_99_emu_db", sample)
        os.makedirs(p, exist_ok=True)
        rows = []
        for g, a in zip(genera, x):
            if a > 0:
                for sp, f in enumerate(rng.dirichlet(np.ones(rng.integers(1, 3)))):
                    rows.append(dict(tax_id=str(abs(hash((g, sp))) % 10 ** 6), abundance=a * f,
                                     species=f"{g} sp{sp + 1}", genus=g, keep_for_analysis="True"))
        rows.append(dict(tax_id="unassigned", abundance=.01, species=None, genus=None, keep_for_analysis="True"))
        pd.DataFrame(rows).to_csv(os.path.join(p, f"{sample}_rel-abundance_filtered.tsv"), sep="\t", index=False)

    def comp(w, conc=14):
        x = rng.dirichlet(np.array(w, float) / np.sum(w) * conc + .03)
        x[x < 1e-3] = 0
        return x / x.sum()

    def skin(site, infl, region):
        w = np.array(SKIN_W[site], float)
        if site == "lesional":
            w[1] *= np.exp(1.1 * infl)
            w[0] *= np.exp(-.5 * infl)
        w[0] *= 2 if region == "sebaceous" else 1
        w[2] *= 1.8 if region == "moist" else 1
        return comp(w)

    def gut(infl):
        w = np.array(GUT_W, float)
        w[1] *= np.exp(-.6 * infl)
        w[7] *= np.exp(.9 * infl)
        return comp(w, 16)

    def add_meta(sample, study, subj, group, site, lesion, stage, pasi, region, tp):
        meta.append(dict(sample_id=sample, study=study, subject=subj, group=group, site=site, lesion_status=lesion,
                         region_location="trunk" if site == "skin" else None, region=region,
                         global_severity=pasi, stage=stage, timepoint=tp))

    def add_cer(study, subj_num, occ, loc, infl):
        for cls, n in CER:
            eo = cls.startswith("EO")
            base = 40 * np.exp(-((n - (68 if eo else 44)) / 4.5) ** 2)
            tilt = np.exp(-.10 * infl * (n - 42)) if (loc == "lesional" and not eo) else 1.0
            v = base * tilt * (np.exp(-.35 * infl) if (eo and loc == "lesional") else 1.0) * rng.lognormal(0, .25)
            u = rng.uniform()
            v = 0.0 if (v < .25 and u < .5) else np.nan if u > .99 else round(v, 3)     # low values: often 0
            cer.append(dict(study=f"CHDR{study}", subject=subj_num, occ=occ, location=loc,
                            ceramide=f"{cls}_C{n}", value=v))

    def add_energy(subj_id, time, infl, counter=[1000]):
        for site, k in (("blister", 1.0), ("plasma", .3)):
            counter[0] += 1
            row = {"LCAB_ID": f"TES0019_{counter[0]}_A", "Subj_ID": subj_id, "Sampling_time": time}
            for m, (base, eff) in METAB.items():
                v = base * np.exp(eff * k * infl) * rng.lognormal(0, .2)
                row[m] = f"{v:.3f}".replace(".", ",") if rng.uniform() > .02 else ""
            energy[site].append(row)

    regions = ["moist", "sebaceous", "dry"]
    n = 0
    # ---- 2230: MF / FMF, skin (L, NL) + nostril only; no timepoint label, a few patients with a follow-up ('fu')
    for group, stages in (("MF", ["IA", "IB", "IIB", "IV"]), ("FMF", ["IA", "IB"])):
        for si, stage in enumerate(stages):
            for j in range(6):
                n += 1
                subj, region, infl = f"{n:03d}", regions[n % 3], si * .8 + rng.normal(0, .5)
                for tp, f in ((None, 1.0), ("fu", .6)):
                    if tp == "fu" and j > 0:
                        continue
                    for lesion in ("lesional", "non_lesional", "RNOS"):
                        site = {"non_lesional": "non-lesional", "RNOS": "nostril"}.get(lesion, lesion)
                        s = f"NBD{n:03d}_{lesion}" + ("_fu" if tp else "")
                        add_meta(s, "2230", subj, group, "skin", lesion, stage, None,
                                 None if lesion == "RNOS" else region, tp)
                        write_emu("CHDR2230_SK", s, SKIN, skin(site, infl * f, region))
    # ---- 1947: mild MF + healthy; skin v1/v2/v3, gut v1/v3, ceramides occ 1/3
    for i in range(12):
        subj, region, infl = f"{100 + i:03d}", regions[i % 3], rng.normal(.3, .5)
        for tp, f in (("v1", 1.0), ("v2", .8), ("v3", .5)):
            if tp == "v2" and i >= 6:
                continue
            for lesion in ("lesional", "non_lesional"):
                s = f"S47_{subj}_{lesion}_{tp}"
                add_meta(s, "1947", subj, "MF", "skin", lesion, "IA", None, region, tp)
                write_emu("CHDR1947_SK", s, SKIN, skin(lesion.replace("_", "-"), infl * f, region))
                if i < 9 and tp != "v2":
                    add_cer("1947", 100 + i, {"v1": 1, "v3": 3}[tp], lesion.replace("_", " "), infl * f)
            if tp != "v2":
                add_meta(f"S47_{subj}_FC_{tp}", "1947", subj, "MF", "gut", None, "IA", None, None, tp)
                write_emu(SHARED, f"S47_{subj}_FC_{tp}", GUT, gut(infl * f))
    for i in range(8):
        subj, region = f"{150 + i:03d}", regions[i % 3]
        add_meta(f"S47_{subj}_SK", "1947", subj, "healthy", "skin", "healthy", None, None, region, "v1")
        write_emu("CHDR1947_SK", f"S47_{subj}_SK", SKIN, skin("non-lesional", 0, region))
        add_meta(f"S47_{subj}_FC", "1947", subj, "healthy", "gut", None, None, None, None, "v1")
        write_emu(SHARED, f"S47_{subj}_FC", GUT, gut(rng.normal(0, .3)))
        if i < 6:
            add_cer("1947", 150 + i, 1, "healthy", 0.0)
    # ---- 1806: psoriasis (PASI) + healthy; skin/gut baseline + week12, ceramides occ 1/2, energy 0/1/2
    for i in range(24):
        pasi = round(float(rng.uniform(2, 30)), 1)
        infl = (pasi - 4) / 9 + rng.normal(0, .35)
        subj, region, sid = f"{200 + i:03d}", regions[i % 3], f"{200 + i}_{INITIALS[i % 12]}"
        for tp, f in (("baseline", 1.0), ("week12", .35)):
            if tp == "week12" and i >= 16:
                continue
            for lesion in ("lesional", "non_lesional"):
                s = f"S06_{subj}_{lesion}_{tp}"
                add_meta(s, "1806", subj, "psoriasis", "skin", lesion, None, pasi, region, tp)
                write_emu(SHARED, s, SKIN, skin(lesion.replace("_", "-"), infl * f, region))
                if i < 20:
                    add_cer("1806", 200 + i, {"baseline": 1, "week12": 2}[tp], lesion.replace("_", "-"), infl * f)
            if i < 20:
                add_meta(f"S06_{subj}_FC_{tp}", "1806", subj, "psoriasis", "gut", None, None, pasi, None, tp)
                write_emu(SHARED, f"S06_{subj}_FC_{tp}", GUT, gut(infl * f))
        if i < 18:
            for time, f in ((0, 1.0), (1, .7), (2, .35)):
                if time == 0 or i < 14:
                    add_energy(sid, time, infl * f)
    for i in range(8):
        subj, region, sid = f"{250 + i:03d}", regions[i % 3], f"{250 + i}_{INITIALS[i % 12]}"
        add_meta(f"S06_{subj}_SK", "1806", subj, "healthy", "skin", "healthy", None, None, region, "baseline")
        write_emu(SHARED, f"S06_{subj}_SK", SKIN, skin("non-lesional", 0, region))
        add_meta(f"S06_{subj}_FC", "1806", subj, "healthy", "gut", None, None, None, None, "baseline")
        write_emu(SHARED, f"S06_{subj}_FC", GUT, gut(rng.normal(0, .3)))
        add_cer("1806", 250 + i, 1, "healthy", 0.0)
        add_energy(sid, 0 if i % 2 else "nvt", 0.0)

    pd.DataFrame(meta).to_csv(os.path.join(out, "meta_data_all3_new.csv"), index=False)
    pd.DataFrame(cer).to_csv(os.path.join(out, "ceramides_long.csv"), index=False)
    for site, rows in energy.items():
        pd.DataFrame(rows).to_csv(os.path.join(out, f"energy_{site}.txt"), sep="\t", index=False)
    print(f"Demo data written to {out}/  ({len(meta)} sequencing samples, {len(cer)} ceramide rows, "
          f"{len(energy['blister'])} + {len(energy['plasma'])} energy rows)")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_data")
