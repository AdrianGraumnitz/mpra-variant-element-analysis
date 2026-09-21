# =============================================================================
# element_barcode_filtering.ipynb  -- streamlined MPRAlib port
#
# HOW TO USE
#   Keep notebook cell 0 (imports) and cell 3 (pd.read_table of df_NGN2/df_WTC11
#   is now optional -- the code below reads the tables through MPRAlib directly).
#   DELETE / don't-run every cell from cell 4 to the end (the whole
#   threshold_filter / barcode_filter / dna_rna_norm / map_oligo_to_barcode /
#   reshape / merge / box-plot machinery, cells 4-113).
#   Paste the 5 cells below in their place. Runs in ~1-2 min for the full 2x2 grid.
# =============================================================================


# ---- CELL A : config + load raw tables once -------------------------------
import os, itertools
from copy import deepcopy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpralib.mpradata import MPRABarcodeData, BarcodeFilter
from mpralib.utils.plot import barcodes_per_oligo

BC_THRESHOLDS = [10, 50]     # min unique barcodes per oligo   (was 0/10/20/50)
Z_SCORES      = [1, 3]       # OLIGO_SPECIFIC RNA z-score cutoff (was 0/1/2/3)

FILES = {
    "NGN2":  "data/reporter_experiment.barcode.NGN2.bbmapMapq30StrandSensitiveAssignment.default.all.tsv.gz",
    "WTC11": "data/reporter_experiment.barcode.WTC11.bbmapMapq30StrandSensitiveAssignment.default.all.tsv.gz",
}
raw = {cell: MPRABarcodeData.from_file(path) for cell, path in FILES.items()}
for cell, d in raw.items():
    print(f"{cell}: {d.n_vars:,} barcodes, replicates {list(d.obs_names)}")
os.makedirs("data/results", exist_ok=True)


# ---- CELL B : one function that does filter + normalize + per-oligo stats --
# Replaces: threshold_filter, barcode_filter, dna_rna_norm,
#           filter_and_normalize_barcodes, map_oligo_to_barcode,
#           calculate_effect_size_per_replicate, reshape_replicate_effect_sizes,
#           aggregate_effect_size_mean_var_per_oligo, rename_aggregate_columns.
#
# MPRAlib keeps the barcode filter as a boolean mask (d.var_filter, shape
# n_barcodes x n_replicates). The normalized-count accessors already zero out
# filtered / unobserved entries, so log2(rna_norm/dna_norm) is NaN exactly where
# a barcode was dropped -> it falls out of the per-oligo mean/var automatically.

def build_oligo_stats(base, cell, bc_threshold, z_score, verbose=True):
    d = deepcopy(base)
    d.var_filter = None                       # start from unfiltered
    d.barcode_threshold = bc_threshold

    # (1) barcode filter: drop oligos with < bc_threshold barcodes
    d.apply_barcode_filter(BarcodeFilter.MIN_BCS_PER_OLIGO, {"threshold": bc_threshold})
    flagged_after_bc = int(d.var_filter.any(axis=1).sum())

    # (2) z-score filter: per-oligo RNA-count outlier barcodes
    if z_score and z_score > 0:
        d.apply_barcode_filter(
            BarcodeFilter.OLIGO_SPECIFIC,
            {"times_zscore": z_score, "apply_bc_threshold": True},
        )
    flagged_after_z = int(d.var_filter.any(axis=1).sum())

    # per-barcode, per-replicate effect size on normalised counts
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.log2(d.normalized_rna_counts / d.normalized_dna_counts)
    eff[~np.isfinite(eff)] = np.nan

    long = pd.DataFrame(
        {"oligo_name": np.tile(d.oligos.values, d.n_obs), "effect_size": eff.reshape(-1)}
    ).dropna(subset=["effect_size"])

    stats = (
        long.groupby("oligo_name")["effect_size"]
        .agg(["mean", "var", "count"])
        .rename(columns={"mean": f"mean_{cell}", "var": f"var_{cell}", "count": f"n_{cell}"})
    )

    if verbose:
        print(
            f"[{cell:5s} bc>={bc_threshold:<2d} z={z_score}]  "
            f"barcodes flagged by MIN_BCS_PER_OLIGO: {flagged_after_bc:>9,}  |  "
            f"after +OLIGO_SPECIFIC z-score: {flagged_after_z:>9,}  "
            f"(Δ {flagged_after_z - flagged_after_bc:>9,})  |  "
            f"oligos with >=1 usable barcode: {len(stats):>6,}  |  "
            f"median var: {stats[f'var_{cell}'].median():.4f}"
        )
    return d, stats


# ---- CELL C : run the 2x2 grid, write tables, PROVE filtering changed things
results = {}   # (bc_threshold, z_score) -> merged NGN2+WTC11 per-oligo table
filtered_objs = {}
print("=== filtering + per-oligo effect-size variance ===")
for bc_t in BC_THRESHOLDS:
    for z in Z_SCORES:
        d_ngn2, s_ngn2 = build_oligo_stats(raw["NGN2"], "NGN2", bc_t, z)
        d_wtc11, s_wtc11 = build_oligo_stats(raw["WTC11"], "WTC11", bc_t, z)
        merged = s_ngn2.join(s_wtc11, how="outer").reset_index()
        merged.insert(1, "bc_threshold", bc_t)
        merged.insert(2, "z_score", z)
        results[(bc_t, z)] = merged
        filtered_objs[("NGN2", bc_t, z)] = d_ngn2
        filtered_objs[("WTC11", bc_t, z)] = d_wtc11
        out = f"data/results/oligo_stats_bc{bc_t}_z{z}.tsv.gz"
        merged.to_csv(out, sep="\t", index=False)
        print(f"    -> wrote {out}   shape={merged.shape}")

# ---- PROOF 1: barcode filter actually removed barcodes (unfiltered vs filtered)
print("\n=== PROOF 1: MIN_BCS_PER_OLIGO changed the data ===")
base = raw["NGN2"]
d_chk = deepcopy(base); d_chk.var_filter = None
bc_before = int((d_chk.oligo_data.barcode_counts > 0).sum())
d_chk.barcode_threshold = 50
d_chk.apply_barcode_filter(BarcodeFilter.MIN_BCS_PER_OLIGO, {"threshold": 50})
print(f"NGN2  var_filter True cells: 0  ->  {int(d_chk.var_filter.sum()):,}")
print(f"NGN2  barcodes flagged (>=1 rep): {int(d_chk.var_filter.any(axis=1).sum()):,} / {d_chk.n_vars:,}")

# ---- PROOF 2: the z-score step removes *additional* barcodes on top of that
print("\n=== PROOF 2: OLIGO_SPECIFIC z-score removes more on top ===")
for (cell, bc_t, z), d in filtered_objs.items():
    if z == Z_SCORES[0]:
        d_hi = filtered_objs[(cell, bc_t, Z_SCORES[-1])]
        print(f"{cell:5s} bc>={bc_t:<2d}:  z={Z_SCORES[0]} flags {int(d.var_filter.sum()):>10,} cells   "
              f"vs  z={Z_SCORES[-1]} flags {int(d_hi.var_filter.sum()):>10,} cells")

# ---- PROOF 3: the written tables are genuinely different from each other
print("\n=== PROOF 3: the 4 written tables differ ===")
for a, b in itertools.combinations(results, 2):
    da, db = results[a], results[b]
    m = da.merge(db, on="oligo_name", suffixes=(f"_{a}", f"_{b}"))
    dv = (m[f"var_NGN2_{a}"] - m[f"var_NGN2_{b}"]).abs()
    assert not da.equals(db), f"{a} and {b} are identical -- filtering had no effect!"
    print(f"  bc/z {a} vs {b}: shared oligos {len(m):>6,} | "
          f"var_NGN2 differs (>1e-9) for {int((dv > 1e-9).sum()):>6,} | mean |Δvar_NGN2| {dv.mean():.4f}")
print("OK - all tables distinct.")


# ---- CELL D : the comparison plots (replaces cell_variance_box_plot etc.) ---
long_all = []
for (bc_t, z), merged in results.items():
    for cell in ("NGN2", "WTC11"):
        sub = merged[["oligo_name", f"var_{cell}"]].rename(columns={f"var_{cell}": "variance"})
        sub["cell_type"] = cell
        sub["filter"] = f"bc>={bc_t}, z={z}"
        long_all.append(sub)
long_all = pd.concat(long_all, ignore_index=True).dropna(subset=["variance"])

g = sns.catplot(
    data=long_all, x="cell_type", y="variance", col="filter", col_wrap=2,
    kind="box", showfliers=False, notch=True, height=3.5, aspect=0.8,
)
g.set_titles("{col_name}")
g.set_axis_labels("", "per-oligo effect-size variance")
plt.tight_layout()
plt.show()

# barcodes-per-oligo, before vs after, for one grid point (replaces cells 18-26)
d_before = deepcopy(raw["NGN2"]); d_before.var_filter = None
barcodes_per_oligo(d_before.oligo_data)
plt.suptitle("NGN2 barcodes per oligo -- unfiltered"); plt.show()
barcodes_per_oligo(filtered_objs[("NGN2", 50, 3)].oligo_data)
plt.suptitle("NGN2 barcodes per oligo -- bc>=50, z=3"); plt.show()


# ---- (optional) CELL E : rebuild the old-style long variance table ----------
# equivalent of the old data/results/oligo_variance_3.tsv.gz (was bc=0, z=3).
# pick whichever grid point you want:
variance_long = (
    results[(10, 3)]
    .melt(id_vars="oligo_name", value_vars=["var_NGN2", "var_WTC11"],
          var_name="cell_type", value_name="variance")
    .assign(cell_type=lambda x: x["cell_type"].str.removeprefix("var_"))
    .dropna(subset=["variance"])
)
variance_long.to_csv("data/results/oligo_variance_bc10_z3.tsv.gz", sep="\t", index=False)
print(variance_long.groupby("cell_type")["variance"].describe())
