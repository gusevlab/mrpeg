import copy
import os
import warnings

import numpy as np
from intervaltree import IntervalTree

from . import log

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import pandas as pd
from scipy.stats import norm

__all__ = [
    "_parameter_check",
    "_process_gwas",
    "_process_ref",
    "_annot_tree",
    "_summarize",
]


def _parameter_check(
    args,
) -> None:

    if not os.path.exists(args.gwas):
        raise ValueError("GWAS file is not found. Check your input.")

    if not os.path.exists(args.ref):
        raise ValueError("Reference file is not found. Check your input.")

    if not (args.keep is None or os.path.exists(args.ref)):
        raise ValueError("Keep file is not found. Check your input.")

    if args.window < 0:
        raise ValueError("Invalid window input. Choose a positive number")

    if args.threshold <= 0 or args.threshold > 1:
        raise ValueError(
            "Invalid p value threshold input. Choose a number between 0 and 1."
        )
    if args.chr is not None:
        if (np.array(args.chr) < 1).any() or (np.array(args.chr) > 22).any():
            raise ValueError(
                "Invalid chromosome input. Choose a number between 1 and 22."
            )

    return None


def _process_gwas(gwas, gwas_cols, n_chr, threshold):

    df_gwas = pd.read_csv(gwas, sep="\t").dropna()

    if not all(col in df_gwas.columns for col in gwas_cols):
        raise ValueError(
            "Specified GWAS columns are not in the GWAS data. Revisit the GWAS Data."
        )

    old_num = df_gwas.shape[0]
    log.logger.info(f"Reading GWAS with {old_num} SNPs.")

    df_gwas = (
        df_gwas[gwas_cols]
        .rename(
            columns={
                f"{gwas_cols[0]}": "CHR",
                f"{gwas_cols[1]}": "SNP",
                f"{gwas_cols[2]}": "BP",
                f"{gwas_cols[3]}": "BETA",
                f"{gwas_cols[4]}": "SE",
            }
        )
        .sort_values(by=["CHR", "BP"])
        .reset_index(drop=True)
    )

    df_gwas["CHR"] = pd.to_numeric(df_gwas["CHR"], errors="coerce")
    df_gwas["BP"] = pd.to_numeric(df_gwas["BP"], errors="coerce")
    df_gwas["BETA"] = pd.to_numeric(df_gwas["BETA"], errors="coerce")
    df_gwas["SE"] = pd.to_numeric(df_gwas["SE"], errors="coerce")

    df_gwas = df_gwas.dropna()

    if (df_gwas["SE"] <= 0).any():
        log.logger.info(f"GWAS data contains SNP with 0 or negative value of SE. Will remove these SNPs.")
        df_gwas = df_gwas[df_gwas.SE > 0]

    df_gwas[["CHR", "BP"]] = df_gwas[["CHR", "BP"]].astype(int)
    df_gwas["Z"] = df_gwas.BETA / df_gwas.SE

    # only focus on autosome
    df_gwas = df_gwas[df_gwas["CHR"].between(1, 22)]

    if n_chr is not None:
        df_gwas = df_gwas[df_gwas.CHR.isin(n_chr)]
        if df_gwas.shape[0] == 0:
            raise ValueError(
                f"GWAS data doesn't contain any SNPs on chromosome {n_chr}."
            )
        old_num = df_gwas.shape[0]
        log.logger.info(f"Filtering down to {old_num} SNPs on chromosome {n_chr}.")

    z_threshold = norm.ppf(1 - threshold / 2)
    df_gwas = df_gwas[df_gwas.Z.abs() >= z_threshold]

    if df_gwas.shape[0] == 0:
        raise ValueError(
            "GWAS data doesn't contain any SNPs after filtering on threshold."
        )

    num_diff = old_num - df_gwas.shape[0]
    if num_diff != 0:
        log.logger.info(
            f"Removing {num_diff} SNPs based on threshold, and {df_gwas.shape[0]} left."
        )

    return df_gwas


def _process_ref(ref, ref_cols, n_chr, keep, window):

    df_ref = pd.read_csv(ref, sep="\t").dropna()

    if not all(col in df_ref.columns for col in ref_cols):
        raise ValueError(
            "Specified reference columns are not in the reference data. Revisit the reference Data."
        )

    log.logger.info(f"Reading {df_ref.shape[0]} annotations.")

    df_ref = df_ref[ref_cols]

    if ref_cols[1] == ref_cols[2]:
        ref_cols[2] = f"{ref_cols[2]}_2"
        df_ref.columns = ref_cols

        log.logger.info(
            f"Constructing window around single basepairs of {ref_cols[1]}."
        )

        if window == 0:
            raise ValueError("Cannot make windows around single basepair.")
    else:
        log.logger.info(
            f"Constructing window around regions between {ref_cols[1]} and {ref_cols[2]}."
        )

    df_ref = (
        df_ref.rename(
            columns={
                f"{ref_cols[0]}": "CHR",
                f"{ref_cols[1]}": "P0",
                f"{ref_cols[2]}": "P1",
                f"{ref_cols[3]}": "ANNO",
            }
        )
        .drop_duplicates(subset=["ANNO"])
        .sort_values(by=["CHR", "P0", "P1"])
        .reset_index(drop=True)
    )

    # Convert CHR and BP to numeric, coercing errors; this turns invalid parsing into NaN
    df_ref["CHR"] = pd.to_numeric(df_ref["CHR"], errors="coerce")
    df_ref["P0"] = pd.to_numeric(df_ref["P0"], errors="coerce")
    df_ref["P1"] = pd.to_numeric(df_ref["P1"], errors="coerce")

    # Drop rows with NaN values
    df_ref = df_ref.dropna()

    df_ref[["CHR", "P0", "P1"]] = df_ref[["CHR", "P0", "P1"]].astype(int)

    # only focus on autosome
    df_ref = df_ref[df_ref["CHR"].between(1, 22)]

    if n_chr is not None:
        df_ref = df_ref[df_ref.CHR.isin(n_chr)]
        if df_ref.shape[0] == 0:
            raise ValueError(
                f"Reference data doesn't contain any SNPs on chromosome {n_chr}."
            )
        log.logger.info(
            f"Filtering to {df_ref.shape[0]} annotations on chromosome {n_chr}."
        )

    half_window = int(window * 1000 / 2)
    df_ref["P0_FLANK"] = np.maximum(df_ref["P0"] - half_window, 0)
    df_ref["P1_FLANK"] = df_ref["P1"] + half_window

    if keep is not None:
        df_keep = pd.read_csv(keep, sep="\t", header=None)
        df_ref = df_ref[df_ref["ANNO"].isin(df_keep[0])]
        if df_ref.shape[0] == 0:
            raise ValueError(
                "Annotation columns doesn't contain any annotations on the keep file."
            )

    log.logger.info(f"Prepared {df_ref.shape[0]} annotations.")

    return df_ref


def _annot_tree(df_gwas, df_ref, sep, snps_anno):

    anno_chrs = df_gwas.CHR.unique()

    res_full = []
    res_filter = []

    for n_chr in anno_chrs:
        tree = IntervalTree()
        log.logger.info(
            f"Constructing interval tree for annotations on chromosome {n_chr}."
        )
        tmp_ref = df_ref[df_ref.CHR == n_chr]
        tmp_gwas = df_gwas[df_gwas.CHR == n_chr]

        for index, row in tmp_ref.iterrows():
            start = int(row["P0_FLANK"])
            end = int(row["P1_FLANK"])
            l_annots = row["ANNO"]
            tree[start:end] = l_annots

        # tree.merge_overlaps(data_reducer=data_reducer)
        for index, row in tmp_gwas.iterrows():
            row = pd.DataFrame(row).T
            bp = int(row["BP"])
            l_annots = sep.join(entry.data for entry in tree[bp])

            tmp_split = l_annots.split(sep)

            # tmp_split = list(set(tmp_split))
            tmp_split = [x for i, x in enumerate(tmp_split) if x not in tmp_split[:i]]

            for idx in range(len(tmp_split)):
                tmp_row = copy.deepcopy(row)
                tmp_row["ANNO"] = tmp_split[idx]
                if snps_anno:
                    res_full.append(tmp_row)

                if len(tmp_split[idx]) != 0:
                    res_filter.append(tmp_row)

    if len(res_full) != 0:
        res_full = pd.concat(res_full)
    else:
        res_full = pd.DataFrame()

    if len(res_filter) != 0:
        res_filter = pd.concat(res_filter)
    else:
        res_filter = pd.DataFrame()

    return res_full, res_filter


def _summarize(anno_snps, df_ref):
    anno_snps = anno_snps[~anno_snps.duplicated(subset=["ANNO", "SNP"], keep="first")]
    result = (
        anno_snps.groupby("ANNO")["Z"]
        .apply(
            lambda x: pd.Series(
                {
                    "mean_chisq": np.mean(x ** 2),
                    "sd_chisq": np.std(x ** 2),
                    "median_chisq": np.median(x ** 2),
                    "max_chisq": np.max(x ** 2),
                    "min_chisq": np.min(x ** 2),
                    "qtl1_chisq": np.percentile(x ** 2, 25),
                    "qtl3_chisq": np.percentile(x ** 2, 75),
                    "mean_z": np.mean(x),
                    "sd_z": np.std(x),
                    "median_z": np.median(x),
                    "max_z": np.max(x),
                    "min_z": np.min(x),
                    "qtl1_z": np.percentile(x, 25),
                    "qtl3_z": np.percentile(x, 75),
                    "count": x.size,
                }
            )
        )
        .unstack()
        .reset_index()
    )

    result = df_ref[df_ref.ANNO.isin(anno_snps.ANNO)][
        ["ANNO", "CHR", "P0", "P1", "P0_FLANK", "P1_FLANK"]
    ].merge(result, how="left", on="ANNO")

    return result
