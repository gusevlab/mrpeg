import glob
import os
import warnings
from typing import Any, List, NamedTuple, Tuple

import jax.numpy as jnp
import jax.scipy.stats as stats
import pandas as pd
from jax import Array, lax
from jax._src import prng
from jax.numpy.linalg import inv
from jax.typing import ArrayLike
from scipy.linalg import block_diag
from scipy.stats import t

from . import log

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from pandas_plink import read_plink

from jax import random
from jax import vmap
__all__ = [
    "CleanData",
    "_parameter_check",
    "_prepare_gwas",
    "_prepare_eqtl",
    "_prepare_perturb",
    "_allele_check",
    "_process_raw",
    "_mrld",
    "null_result",
    "_make_null",
    "_get_p",
    "infer_peg",
]


class CleanData(NamedTuple):
    """Define the class for the prior parameter of SuShiE model.

    Attributes:
        beta: The GWAS effect size.
        inv_se: The diagonal matrix of inverse of GWAS standard error.
        eqtl: The eQTL z scores.
        perturb: The perturbation effect z scores.
        inv_ld: The inverse of the LD matrix
        gene_names: The downstream gene names.

    """

    beta: Array
    inv_se: Array
    eqtl: Array
    perturb: Array
    inv_ld: Array
    gene_names: List


def _parameter_check(
    args,
) -> None:
    if not os.path.exists(args.gwas):
        raise ValueError("No GWAS file located. Check your input.")

    if not os.path.exists(args.eqtl):
        raise ValueError("No eQTL file located. Check your input.")

    if not os.path.exists(args.perturb):
        raise ValueError("No Perturb-Seq file located. Check your input.")

    if not glob.glob(args.ref_geno + "*"):
        raise ValueError("No reference genotype file located. Check your input.")

    return None


def _prepare_gwas(gwas: str, gwas_cols: List, keep_ambiguous: bool) -> pd.DataFrame:
    df_gwas = pd.read_csv(gwas, sep="\t").dropna()

    if not all(col in df_gwas.columns for col in gwas_cols):
        raise ValueError("The specified GWAS columns are not in the GWAS data.")

    log.logger.info(f"GWAS contains {df_gwas.shape[0]} SNPs.")

    df_gwas = (
        df_gwas[gwas_cols]
        .rename(
            columns={
                gwas_cols[0]: "CHR",
                gwas_cols[1]: "SNP",
                gwas_cols[2]: "A1_gwas",
                gwas_cols[3]: "A0_gwas",
                gwas_cols[4]: "BETA",
                gwas_cols[5]: "SE",
            }
        )
        .replace([jnp.inf, -jnp.inf], jnp.nan, inplace=False)
        .dropna(inplace=False)
    )

    df_gwas[["CHR"]] = df_gwas[["CHR"]].astype(int)
    # only focus on autosome
    df_gwas = df_gwas[df_gwas["CHR"].between(1, 22)]

    # add a check on SE negative value
    if (df_gwas["SE"] <= 0).any():
        log.logger.debug(f"GWAS data contains SNP with 0 or negative value of standard error. Will remove these SNPs.")
        df_gwas = df_gwas[df_gwas.SE > 0]

    if not keep_ambiguous:
        ambiguous_snps = ["AT", "TA", "CG", "GC"]
        if_ambig = (df_gwas.A1_gwas + df_gwas.A0_gwas).isin(ambiguous_snps)
        del_num = if_ambig.sum()
        df_gwas = df_gwas[~if_ambig].reset_index(drop=True)

        if df_gwas.shape[0] == 0:
            raise ValueError(
                "All SNPs are ambiguous in GWAS data. Check the source."
            )

        if del_num != 0:
            log.logger.debug(f"Drop {del_num} ambiguous SNPs in genotype data.")
    
    return df_gwas


def _prepare_eqtl(eqtl: str, eqtl_cols: List) -> pd.DataFrame:
    df_eqtl = pd.read_csv(eqtl, sep="\t").dropna()

    if not all(col in df_eqtl.columns for col in eqtl_cols):
        raise ValueError("The specified eQTL columns are not in the eQTL data.")

    log.logger.info(f"eQTL contains {df_eqtl.shape[0]} SNPs.")

    df_eqtl = (
        df_eqtl[eqtl_cols]
        .rename(
            columns={
                eqtl_cols[0]: "CHR",
                eqtl_cols[1]: "SNP",
                eqtl_cols[2]: "A1_eqtl",
                eqtl_cols[3]: "A0_eqtl",
                eqtl_cols[4]: "Z_eqtl",
                eqtl_cols[5]: "GENE",
            }
        )
        .replace([jnp.inf, -jnp.inf], jnp.nan, inplace=False)
        .dropna(inplace=False)
    )

    df_eqtl[["CHR"]] = df_eqtl[["CHR"]].astype(int)
    # only focus on autosome
    df_eqtl = df_eqtl[df_eqtl["CHR"].between(1, 22)].reset_index(drop=True)

    return df_eqtl


def _prepare_perturb(perturb: str, top_signal: int) -> Tuple[pd.DataFrame, List]:
    df_perturb = (
        pd.read_csv(perturb, sep="\t")
        .replace([jnp.inf, -jnp.inf], jnp.nan, inplace=False)
        .dropna(inplace=False)
        .reset_index(drop=True)
    )
    df_perturb = df_perturb.rename(columns={f"{df_perturb.columns[0]}": "GENE"})
    
    # remove duplicated perturbed genes
    df_perturb = df_perturb.drop_duplicates(subset="GENE", keep="first")

    df_perturb = df_perturb.replace(jnp.nan, 0)
    ds_genes = df_perturb.columns[1 : df_perturb.shape[1]].tolist()

    if top_signal == 0:
        log.logger.debug("Inference will use all perturbed genes.")
    else:
        if top_signal > df_perturb.shape[0]:
            log.logger.warning("Specified number of top signal is larger than the" +
                               "number of perturbed genes. Will use all perturbed genes.")
    
    log.logger.info(
        f"Perturb matrix contains {df_perturb.shape[0]} perturbed genes and {len(ds_genes)} downstream genes."
    )

    return df_perturb, ds_genes


def _allele_check(
    baseA1: pd.Series,
    baseA0: pd.Series,
    compareA1: pd.Series,
    compareA0: pd.Series,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    correct = jnp.array(
        ((baseA1 == compareA1) * 1) * ((baseA0 == compareA0) * 1), dtype=int
    )
    flipped = jnp.array(
        ((baseA1 == compareA0) * 1) * ((compareA1 == baseA0) * 1), dtype=int
    )

    (correct_idx,) = jnp.where(correct == 1)
    (flipped_idx,) = jnp.where(flipped == 1)
    (wrong_idx,) = jnp.where((correct + flipped) == 0)

    return correct_idx, flipped_idx, wrong_idx

def create_diagonal(column):
    return jnp.diag(column)

def _process_raw(
    gwas: str,
    eqtl: str,
    perturb: str,
    gwas_cols: List,
    eqtl_cols: List,
    ref_geno: str,
    keep_ambiguous: bool,
    top_signal: int,
) -> CleanData:
    # read in GWAS data
    df_gwas = _prepare_gwas(gwas, gwas_cols, keep_ambiguous)

    # read in eQTL data
    df_eqtl = _prepare_eqtl(eqtl, eqtl_cols)

    # read in perturb data
    df_perturb, ds_genes = _prepare_perturb(perturb, top_signal)

    df_wk = (
        df_eqtl[df_eqtl.GENE.isin(df_perturb.GENE)]
        .sort_values(by=["CHR", "SNP"])
        .reset_index(drop=True)
    )
    num_shared = len(df_wk.GENE.unique())
    chrs = df_wk["CHR"].unique()

    if num_shared <= 2:
        raise ValueError(
            f"Only {num_shared} shared genes between eQTL and perturb data. Check your input."
        )
    elif num_shared <= 30:
        log.logger.info(
            f"eQTL and perturb data shared only {num_shared} genes on {len(chrs)} chromosomes."
            + " You may consider include more genes."
        )
    else:
        log.logger.info(
            f"eQTL and perturb data shared {num_shared} genes on {len(chrs)} chromosomes."
        )

    ref_geno = ref_geno.split("*")
    if len(ref_geno) > 1:
        ld_paths = [ref_geno[0] + str(n_chr) + ref_geno[1] for n_chr in chrs]
    else:
        raise ValueError("Ref data does not contain separator '*'.")

    keep_snps = []
    inv_ld = []

    log.logger.info("Matching SNPs in the reference genotypes.")

    for idx in range(len(chrs)):
        bim, _, bed = read_plink(f"{ld_paths[idx]}", verbose=False)
        bim.columns = ["CHR", "SNP", "CM", "BP", "A0_ref", "A1_ref", "i"]

        bim.CHR = bim.CHR.astype(int)
        df_snp = df_wk.merge(bim, how="inner", on=["CHR", "SNP"]).merge(df_gwas, how="inner", on=["CHR", "SNP"]).reset_index(drop=True)
        
        if df_snp.shape[0] == 0:
            log.logger.debug(
                f"No overlap SNPs between GWAS, eQTL, and reference data on chromosome {chrs[idx]}."
            )
            continue

        # drop wrong SNPs between rep and GWAS
        _, _, wrong_idx = _allele_check(
            df_snp["A1_ref"].values,
            df_snp["A0_ref"].values,
            df_snp["A1_gwas"].values,
            df_snp["A0_gwas"].values,
        )
        
        if len(wrong_idx) != 0:
            df_snp = df_snp.drop(wrong_idx, axis=0).reset_index(drop=True)
            if df_snp.shape[0] == 0:
                log.logger.debug(
                    f"All SNPs do not match between reference and GWAS data on chromosome {chrs[idx]}."
                )
                continue
        
        # drop wrong SNPs between ref and eqtl data 
        _, _, wrong_idx = _allele_check(
            df_snp["A1_ref"].values,
            df_snp["A0_ref"].values,
            df_snp["A1_eqtl"].values,
            df_snp["A0_eqtl"].values,
        )
        
        if len(wrong_idx) != 0:
            df_snp = df_snp.drop(wrong_idx, axis=0).reset_index(drop=True)
            if df_snp.shape[0] == 0:
                log.logger.debug(
                    f"All SNPs do not match between reference and eQTL data on chromosome {chrs[idx]}."
                )
                continue
            
        # flip alleles between ref and GWAS
        _, flip_idx, _ = _allele_check(
            df_snp["A1_ref"].values,
            df_snp["A0_ref"].values,
            df_snp["A1_gwas"].values,
            df_snp["A0_gwas"].values,
        )
        if len(flip_idx) != 0:
            df_snp.loc[flip_idx, "BETA"] = -1 * df_snp.iloc[flip_idx, :]["BETA"].values
            log.logger.debug(f"Flip {len(flip_idx)} SNPs between reference and GWAS data on chromosome {chrs[idx]}.")
        
        # flip alleles between ref and eQTL
        _, flip_idx, _ = _allele_check(
            df_snp["A1_ref"].values,
            df_snp["A0_ref"].values,
            df_snp["A1_eqtl"].values,
            df_snp["A0_eqtl"].values,
        )
        
        if len(flip_idx) != 0:
            df_snp.loc[flip_idx, "Z_eqtl"] = -1 * df_snp.iloc[flip_idx, :]["Z_eqtl"].values
            log.logger.debug(f"Flip {len(flip_idx)} SNPs between reference and eQTL data on chromosome {chrs[idx]}.")
        
        # we have cases that same SNPs are the top eQTL for multiple genes
        # to make sure we contain as many genes as possible,
        # we select top 5 eQTLs for each gene, and then remove the duplicates
        # and then pick the top eQTLs
        df_snp = (
            df_snp.groupby("GENE")
            .apply(lambda x: x.assign(abs_B=x["Z_eqtl"].abs()).nlargest(5, "abs_B"))
            .reset_index(drop=True)
        )

        df_snp = df_snp.sort_values(
            by="Z_eqtl", key=lambda x: abs(x), ascending=False
        ).drop_duplicates(subset="SNP")

        df_snp = (
            df_snp.groupby("GENE")
            .apply(lambda x: x.loc[abs(x["Z_eqtl"]).idxmax()])
            .reset_index(drop=True)
        )

        X = bed.compute().T[:, df_snp.i]
        X -= jnp.mean(X, axis=0)
        X /= jnp.std(X, axis=0)
        tmp_ld = X.T @ X / X.shape[0]
        inv_ld.append(inv(tmp_ld + 1e-3 * jnp.eye(X.shape[1])))
        keep_snps.append(df_snp[["CHR", "SNP", "BETA", "SE", "Z_eqtl", "GENE"]])

    inv_ld = block_diag(*inv_ld)

    df_wk = pd.concat(keep_snps).merge(df_perturb, how="left", on="GENE").reset_index(drop=True)
    num_diff = num_shared - df_wk.shape[0]
    log.logger.info(
        f"{num_diff} genes are removed because no eQTLs in the reference data."
    )
        
    top_signal_index = {col: df_wk[col].abs().nlargest(top_signal).index for col in df_wk.columns[6:]}

    beta_subset = jnp.column_stack([df_wk.loc[top_signal_index[col], "BETA"].values for col in df_wk.columns[6:]])
    se_subset = jnp.column_stack([df_wk.loc[top_signal_index[col], "SE"].values for col in df_wk.columns[6:]])
    eqtl_subset = jnp.column_stack([df_wk.loc[top_signal_index[col], "Z_eqtl"].values for col in df_wk.columns[6:]])
    perturb_subset = jnp.column_stack([(df_wk.loc[top_signal_index[col], col]).values for col in df_wk.columns[6:]])
    inv_ld_subset = jnp.array([inv_ld[jnp.array(indices),:][:,jnp.array(indices)] for _, indices in top_signal_index.items()])
    
    # result = CleanData(
    #     beta=jnp.array(df_wk.BETA),
    #     inv_se=jnp.diag(1 / df_wk.SE.values),
    #     eqtl=jnp.array(df_wk.Z_eqtl),
    #     perturb=jnp.array(df_wk[ds_genes]),
    #     inv_ld=jnp.array(inv_ld),
    #     gene_names=ds_genes,
    # )
    
    log.logger.info(
        f"Successfully prepared {df_wk.shape[0]} perturbed genes on {len(df_wk.CHR.unique())} chromosomes."
        + f" Start running Mr PEG on {len(ds_genes)} downstream genes."
    )
    
    result = CleanData(
        beta=beta_subset.T,
        inv_se=(1 / se_subset.T),
        eqtl=eqtl_subset.T,
        perturb=perturb_subset.T,
        inv_ld=inv_ld_subset,
        gene_names=ds_genes,
    )

    return result


def _mrld(y, X, inv_dvd):
    import pdb; pdb.set_trace()
    gamma_num = jnp.squeeze(jnp.einsum("ij,jk,km->im", X.T, inv_dvd, y[:, jnp.newaxis]))
    gamma_dem = jnp.einsum("ij,jk,ki->i", X.T, inv_dvd, X)
    mr_gamma = gamma_num / gamma_dem

    epi_hat = y[:, jnp.newaxis] - jnp.einsum("ij,j->ij", X, mr_gamma)
    df = y.shape[0] - 1
    sigma_sq_hat = (1 / df) * jnp.einsum("ij,jk,ki->i", epi_hat.T, inv_dvd, epi_hat)
    se = jnp.sqrt(sigma_sq_hat / gamma_dem)
    mr_z = mr_gamma / se

    return mr_gamma, mr_z


class null_result(NamedTuple):
    gwas_beta: Array
    eqtl: Array
    perturb: Array
    inv_dvd: Array
    rng_key: prng.PRNGKeyArray


def _make_null(result: null_result, empty: Any):
    del empty

    gwas_beta, eqtl, perturb, inv_dvd, rng_key = result

    rng_key, gamma_key = random.split(rng_key, 2)

    new_perturb = random.permutation(gamma_key, perturb, 0)
    X_perturb = jnp.einsum("i,ij->ij", eqtl, new_perturb)
    mr_gamma, _ = _mrld(gwas_beta, X_perturb, inv_dvd)

    carry = result._replace(
        rng_key=rng_key,
    )

    return carry, mr_gamma


def _get_p(mr, null):
    stats_z = (mr - jnp.mean(null, axis=0)) / jnp.std(null, axis=0)
    stats_p = 2 * stats.norm.sf(jnp.abs(stats_z))
    return stats_z, stats_p


def infer_peg(
    beta: ArrayLike,
    inv_se: ArrayLike,
    eqtl: ArrayLike,
    perturb: ArrayLike,
    inv_ld: ArrayLike,
    no_permute: bool = False,
    perm_number: int = 500,
    seed: int = 12345,
) -> Array:
    """The main inference function for running SuShiE.

    Args:
        beta: ArrayLike. GWAS effect sizes.
        inv_se: ArrayLike. The diagonal matrix of the inverse of GWAS Standard error
        eqtl: ArrayLike. eQTL Z scores.
        perturb: ArrayLike. Perturbation effect size matrix.
        inv_ld: ArrayLike. The inverse of the LD matrix.
        no_permute: bool = False. Whether to perform permutation for effect size testing.
        perm_number: int = 500. The number of permutations.
        seed: int = 12345,

    Returns:
        :py:obj:`SushieResult`: A SuShiE result object that contains prior (:py:obj:`Prior`),
        posterior (:py:obj:`Posterior`), ``cs``, ``pip``, ``elbo``, and ``elbo_increase``.

    """
    if seed <= 0:
        raise ValueError(
            "The seed specified for randomization is invalid. Choose a positive integer."
        )

    if not no_permute and perm_number <= 0:
        raise ValueError(
            "The the permutation number is invalid. Choose a positive integer."
        )

    if not no_permute and perm_number <= 100:
        log.logger.warning(
            "The number of permutation is low, and the estimate may be inaccurate."
        )

    dim_fail = (
        (beta.shape[0] != inv_se.shape[0])
        or (beta.shape[0] != eqtl.shape[0])
        or (beta.shape[0] != perturb.shape[0])
        or (beta.shape[0] != inv_ld.shape[0])
        or (beta.shape[1] != inv_se.shape[1])
        or (beta.shape[1] != eqtl.shape[1])
        or (beta.shape[1] != perturb.shape[1])
        or (beta.shape[1] != inv_ld.shape[1])
    )

    if dim_fail:
        raise ValueError(
            "The dimension of GWAS, eQTL, perturb, and the inverse of LD do not match."
        )

    rng_key = random.PRNGKey(seed)
    n_ds, n_p = perturb.shape
    import pdb; pdb.set_trace()
    X = eqtl * perturb
    updated_diag = jnp.diagonal(inv_ld, axis1=1, axis2=2) * inv_se**2
    inv_dvd = inv_ld.at[jnp.arange(n_ds)[:, None], jnp.arange(n_p), jnp.arange(n_p)].set(updated_diag)
    import pdb; pdb.set_trace()
    mr_gamma, mr_z = _mrld(beta, X, inv_dvd)
    mr_p = 2 * t.sf(jnp.abs(mr_z), beta.shape[0] - 1)

    if not no_permute:
        log.logger.info(f"Starting permutation test with {perm_number} times.")
        init_null = null_result(
            gwas_beta=beta,
            eqtl=eqtl,
            perturb=perturb,
            inv_dvd=inv_dvd,
            rng_key=rng_key,
        )

        _, null_dist = lax.scan(_make_null, init_null, xs=None, length=perm_number)
        mr_z_perm, mr_p_perm = _get_p(mr_gamma, null_dist)
    else:
        mr_z_perm = jnp.array([jnp.nan] * X.shape[1])
        mr_p_perm = jnp.array([jnp.nan] * X.shape[1])

    result = jnp.column_stack(
        (
            mr_gamma,
            mr_z,
            mr_p,
            mr_z_perm,
            mr_p_perm,
        )
    )

    return result
