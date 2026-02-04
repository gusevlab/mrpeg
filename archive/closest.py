import os
import warnings
from typing import List, Optional

import numpy as np
from scipy.stats import norm

from . import log

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import pandas as pd

__all__ = [
    "_parameter_check",
    "_process_potential",
    "_get_max_gwas",
    "_find_closest",
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


def _get_max_gwas(gwas: str, gwas_cols: List[str], window: float, threshold: float) -> pd.DataFrame:
    df_gwas = pd.read_csv(gwas, sep="\t").dropna()

    log.logger.info(f"Reading GWAS with {df_gwas.shape[0]} SNPs.")

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
                f"{gwas_cols[3]}": "BETA",
                f"{gwas_cols[4]}": "SE",
            }
        )
        .reset_index(drop=True)
    )

    df_gwas["CHR"] = pd.to_numeric(df_gwas["CHR"], errors="coerce")
    df_gwas["BP"] = pd.to_numeric(df_gwas["BP"], errors="coerce")
    df_gwas["BETA"] = pd.to_numeric(df_gwas["BETA"], errors="coerce")
    df_gwas["SE"] = pd.to_numeric(df_gwas["SE"], errors="coerce")

    df_gwas = df_gwas.dropna()

    if (df_gwas["SE"] <= 0).any():
        log.logger.info(
            "GWAS data contains SNP with 0 or negative value of standard error. Will remove these SNPs."
        )
        df_gwas = df_gwas[df_gwas.SE > 0]

    df_gwas[["CHR", "BP"]] = df_gwas[["CHR", "BP"]].astype(int)
    df_gwas["Z"] = df_gwas.BETA / df_gwas.SE

    # only focus on autosome
    df_gwas = df_gwas[df_gwas["CHR"].between(1, 22)]

    # add checks for Z or P
    z_threshold = norm.ppf(1 - threshold / 2)
    df_gwas = df_gwas[df_gwas["Z"].abs() > z_threshold]

    if df_gwas.shape[0] == 0:
        raise ValueError("GWAS data doesn't contain any significant hits.")
    else:
        log.logger.info(f"Identify {df_gwas.shape[0]} significant SNPs.")

    half_window = int(window * 1000 / 2)
    df_gwas["P0"] = np.maximum(df_gwas.BP - half_window, 0)
    df_gwas["P1"] = df_gwas.BP + half_window

    res = []
    for n_chr in np.sort(df_gwas.CHR.unique()):
        df_snps = (
            df_gwas[df_gwas.CHR == n_chr]
            .sort_values(by=["P0"])
            .reset_index(names="index")
        )
        # Initialize merged intervals list with the first interval
        merged = [df_snps[["P0", "P1"]].iloc[0]]
        idx: List[int] = []
        ct = 0
        # Iterate over remaining intervals
        for _, row in df_snps.iterrows():
            prev_interval = merged[-1]
            # Check for overlap between the current interval and the previous merged interval
            if row["P0"] <= prev_interval["P1"]:
                prev_interval["P1"] = max(row["P1"], prev_interval["P1"])
            else:
                merged.append(row[["P0", "P1"]])
                ct += 1
            idx = idx + [ct]

        df_snps = pd.concat(
            [
                df_snps,
                pd.DataFrame(idx, columns=["index"]).merge(
                    pd.DataFrame(merged)
                    .rename(columns={"P0": "START", "P1": "END"})
                    .reset_index(drop=True)
                    .reset_index(names="index"),
                    on="index",
                ),
            ],
            axis=1,
        )
        res.append(df_snps)

    res = pd.concat(res).drop(columns="index")

    res = res.loc[
        res.groupby(["CHR", "START", "END"])["Z"].transform(
            lambda x: abs(x) == abs(x).max()
        )
    ].reset_index(drop=True)

    res = (
        res.groupby(["CHR", "START", "END"])
        .apply(lambda x: x[~x["Z"].abs().duplicated()])
        .reset_index(drop=True)
        .drop(columns=["P0", "P1"])
    )

    log.logger.info(f"Identify {res.shape[0]} GWAS significant regions.")

    return res


def _process_potential(merge: pd.DataFrame, ref: str, ref_cols: List[str], keep: Optional[str]) -> pd.DataFrame:
    df_ref = pd.read_csv(ref, sep="\t").dropna()

    if not all(col in df_ref.columns for col in ref_cols):
        raise ValueError(
            "Specified reference columns are not in the reference data. Revisit the reference Data."
        )

    df_ref = (
        df_ref[ref_cols]
        .rename(
            columns={
                f"{ref_cols[0]}": "CHR",
                f"{ref_cols[1]}": "TSS",
                f"{ref_cols[2]}": "TES",
                f"{ref_cols[3]}": "GENE",
            }
        )
        .sort_values(by=["CHR", "TSS", "TES"])
        .reset_index(drop=True)
    )

    df_ref["CHR"] = pd.to_numeric(df_ref["CHR"], errors="coerce")
    df_ref["TSS"] = pd.to_numeric(df_ref["TSS"], errors="coerce")
    df_ref["TES"] = pd.to_numeric(df_ref["TES"], errors="coerce")

    df_ref = df_ref.dropna()

    df_ref[["CHR", "TSS", "TES"]] = df_ref[["CHR", "TSS", "TES"]].astype(int)

    df_ref = df_ref[df_ref.CHR.isin(merge.CHR)]

    if df_ref.shape[0] == 0:
        raise ValueError(
            "Reference data doesn't contain chromosomes that have GWAS significant hits."
        )

    if keep is not None:
        df_keep = pd.read_csv(keep, sep="\t", header=None)
        df_ref = df_ref[df_ref.GENE.isin(df_keep[0])]

        if df_ref.shape[0] == 0:
            raise ValueError(
                "The reference file doesn't contain any genes in the keep file."
            )

    df_ref.reset_index(drop=True, inplace=True)

    log.logger.info(f"Find closest genes from {df_ref.shape[0]} potential genes.")

    return df_ref


def _find_closest(sig_gwas: pd.DataFrame, pot_genes: pd.DataFrame) -> pd.DataFrame:
    closest = []
    for idx in range(sig_gwas.shape[0]):
        tmp_snp = sig_gwas.iloc[
            [idx],
        ]
        tmp_pot = pot_genes[pot_genes.CHR.values == tmp_snp.CHR.values].reset_index(
            drop=True
        )
        inside_genes = tmp_pot[
            np.logical_and(
                tmp_pot.TSS.values <= tmp_snp.BP.values,
                tmp_pot.TES.values >= tmp_snp.BP.values,
            )
        ]

        if inside_genes.shape[0] != 0:
            rep = inside_genes.shape[0]
            insert_genes = inside_genes.GENE.values
        else:
            tss_dist = np.abs(tmp_pot.TSS.values - tmp_snp.BP.values)
            tes_dist = np.abs(tmp_pot.TES.values - tmp_snp.BP.values)
            min_dist = np.minimum(tss_dist, tes_dist)
            min_index = np.where(min_dist == min_dist.min())[0]
            rep = len(min_index)
            insert_genes = tmp_pot.GENE[min_index].values

        rep_snp = tmp_snp.values.repeat(rep, axis=0)
        rep_snp = pd.DataFrame(rep_snp, columns=tmp_snp.columns)
        rep_snp["GENE"] = insert_genes
        closest.append(rep_snp)

    closest = pd.concat(closest)

    closest = closest.merge(pot_genes, how="left", on=["CHR", "GENE"]).rename(
        columns={"START": "REGION_START", "END": "REGION_END"}
    )
    return closest


def _find_nearby(sig_gwas: pd.DataFrame, pot_genes: pd.DataFrame, window: float) -> pd.DataFrame:
    nearby = []
    half_window = int(window * 1000 / 2)
    for idx in range(sig_gwas.shape[0]):
        tmp_snp = sig_gwas.iloc[
            [idx],
        ]

        P0 = np.maximum(tmp_snp.BP - half_window, 0)
        P1 = tmp_snp.BP + half_window

        tmp_pot = pot_genes[pot_genes.CHR.values == tmp_snp.CHR.values].reset_index(
            drop=True
        )

        overlap_pot = tmp_pot[
            (tmp_pot["TSS"] <= int(P1.iloc[0])) & (tmp_pot["TES"] >= int(P0.iloc[0]))
        ].copy()
        overlap_pot["snp"] = tmp_snp.SNP.values[0]
        nearby.append(overlap_pot)

    nearby = pd.concat(nearby)

    return nearby
