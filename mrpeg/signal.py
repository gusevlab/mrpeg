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

    return None


def _process_gwas(gwas, gwas_cols, n_chr, threshold):

    df_gwas = pd.read_csv(gwas, sep="\t").dropna()

    if not all(col in df_gwas.columns for col in gwas_cols):
        raise ValueError(
            "Specified GWAS columns are not in the GWAS data. Revisit the GWAS Data."
        )

    df_gwas = (
        df_gwas[gwas_cols]
        .rename(
            columns={
                f"{gwas_cols[0]}": "CHR",
                f"{gwas_cols[1]}": "SNP",
                f"{gwas_cols[2]}": "BP",
                f"{gwas_cols[3]}": "Z",
            }
        )
        .sort_values(by=["CHR", "BP"])
        .reset_index(drop=True)
    )

    df_gwas[["CHR", "BP"]] = df_gwas[["CHR", "BP"]].astype(int)

    if n_chr is not None:
        df_gwas = df_gwas[df_gwas.CHR == n_chr]
        if df_gwas.shape[0] == 0:
            raise ValueError(
                f"GWAS data doesn't contain any SNPs on chromosome {n_chr}."
            )

    z_threshold = norm.ppf(1 - threshold / 2)
    df_gwas = df_gwas[df_gwas.Z.abs() > z_threshold]

    if df_gwas.shape[0] == 0:
        raise ValueError(
            "GWAS data doesn't contain any SNPs after filtering on threshold."
        )

    return df_gwas


def _process_ref(ref, ref_cols, n_chr, keep, window):

    df_ref = pd.read_csv(ref, sep="\t").dropna()

    if not all(col in df_ref.columns for col in ref_cols):
        raise ValueError(
            "Specified reference columns are not in the reference data. Revisit the reference Data."
        )

    df_ref = df_ref[ref_cols]

    if ref_cols[1] == ref_cols[2]:
        ref_cols[2] = f"{ref_cols[2]}_2"
        df_ref.columns = ref_cols

    df_ref = (
        df_ref.rename(
            columns={
                f"{ref_cols[0]}": "CHR",
                f"{ref_cols[1]}": "TSS",
                f"{ref_cols[2]}": "TES",
                f"{ref_cols[3]}": "ANNO",
            }
        )
        .sort_values(by=["CHR", "TSS", "TES"])
        .reset_index(drop=True)
    )

    df_ref[["CHR", "TSS", "TES"]] = df_ref[["CHR", "TSS", "TES"]].astype(int)

    if n_chr is not None:
        df_ref = df_ref[df_ref.CHR == n_chr]
        if df_ref.shape[0] == 0:
            raise ValueError(
                f"Reference data doesn't contain any SNPs on chromosome {n_chr}."
            )

    half_window = int(window * 1000 / 2)
    df_ref["START"] = np.maximum(df_ref["TSS"] - half_window, 0)
    df_ref["END"] = df_ref["TES"] + half_window

    if keep is not None:
        df_keep = pd.read_csv(keep, sep="\t", header=None)
        df_ref = df_ref[df_ref["ANNO"].isin(df_keep[0])]
        if df_ref.shape[0] == 0:
            raise ValueError(
                "Annotation columns doesn't contain any annotations on the keep file."
            )

    return df_ref


def _annot_tree(df_gwas, df_ref, sep):

    anno_chrs = df_gwas.CHR.unique()

    # create interval tree to manage annotations
    log.logger.info("Constructing interval tree for annotations")

    tree = IntervalTree()
    res_full = []
    res_filter = []
    for n_chr in anno_chrs:
        log.logger.info(f"Annotating variants on chromosome {n_chr}")

        tmp_ref = df_ref[df_ref.CHR == n_chr]
        tmp_gwas = df_gwas[df_gwas.CHR == n_chr]

        for index, row in tmp_ref.iterrows():
            start = int(row["START"])
            end = int(row["END"])
            l_annots = row["ANNO"]
            tree[start:end] = l_annots

        # tree.merge_overlaps(data_reducer=data_reducer)
        for index, row in tmp_gwas.iterrows():
            row = pd.DataFrame(row).T
            bp = int(row["BP"])
            l_annots = sep.join(entry.data for entry in tree[bp])

            tmp_split = l_annots.split(sep)
            for idx in range(len(tmp_split)):
                row["ANNO"] = tmp_split[idx]
                res_full.append(row)
                if len(tmp_split[idx]) != 0:
                    res_filter.append(row)

    res_full = pd.concat(res_full)
    res_filter = pd.concat(res_filter)

    return res_full, res_filter


def _summarize(anno_snps):

    result = (
        anno_snps.groupby("NAME")["Z"]
        .apply(
            lambda x: pd.Series(
                {
                    "mean": np.mean(x ** 2),
                    "median": np.median(x ** 2),
                    "max": np.max(x ** 2),
                    "min": np.min(x ** 2),
                    "25%": np.percentile(x ** 2, 25),
                    "75%": np.percentile(x ** 2, 75),
                    "count": x.size,
                }
            )
        )
        .unstack()
        .reset_index()
    )

    return result
