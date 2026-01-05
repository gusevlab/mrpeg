import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pandas as pd
from jax import random
from jax.config import config
from pandas_plink import read_plink
from scipy import stats
from scipy.linalg import block_diag

# set key
rng_key = random.PRNGKey(1234)
config.update("jax_enable_x64", True)


def ols(X, y):
    X_inter = jnp.append(jnp.ones((X.shape[0], 1)), X, axis=1)
    y = jnp.reshape(y, (len(y), -1))
    q_matrix, r_matrix = jnp.linalg.qr(X_inter, mode="reduced")
    qty = q_matrix.T @ y
    beta = jsp.linalg.solve_triangular(r_matrix, qty)
    df = q_matrix.shape[0] - q_matrix.shape[1]
    residual = y - q_matrix @ qty
    # rss = jnp.sum(residual ** 2, axis=0)
    sigma = jnp.sqrt(jnp.sum(residual ** 2, axis=0) / df)
    se = (
        jnp.sqrt(
            jnp.diag(
                jsp.linalg.cho_solve((r_matrix, False), jnp.eye(r_matrix.shape[0]))
            )
        )[:, jnp.newaxis]
        @ sigma[jnp.newaxis, :]
    )
    t_scores = beta / se

    return beta[1:], se[1:], t_scores[1:]


def regress(Z, pheno):
    betas = []
    ses = []
    pvals = []
    zs = []
    for snp in Z.T:
        beta, inter, rval, pval, se = stats.linregress(snp, pheno)
        betas.append(beta)
        ses.append(se)
        pvals.append(pval)
        zs.append(beta / se)

    res = pd.DataFrame({"beta": betas, "se": ses, "pval": pvals, "zs": zs})

    return res


def sim_geno(L, n, key):
    p, p = L.shape
    tmp_geno = jax.random.normal(key, (n, p))
    Z = L.dot(tmp_geno.T).T
    Z -= jnp.mean(Z, axis=0)
    Z /= jnp.std(Z, axis=0)

    return Z


def compute_s2g(L, beta):
    Ltb = jnp.dot(L.T, beta)
    s2g = jnp.dot(Ltb.T, Ltb)
    return s2g


def sim_trait(g, h2g, key):
    n = len(g)

    if h2g > 0:
        s2g = jnp.var(g, ddof=1)
        s2e = s2g * ((1.0 / h2g) - 1)
        e = jax.random.normal(key, (n,)) * jnp.sqrt(s2e)
        y = g + e
    else:
        e = jax.random.normal(key, (n,))
        y = e

    # standardize
    y -= jnp.mean(y)
    y /= jnp.std(y)

    return y


# Prepare EUR PLINK triplet files (bed/bim/fam) from 1000 Genomes.
# These PLINK files include randomly sampled, significant cis-eQTLs from eQTLGen.
# For each gene, we keep the top 3 most significant cis-eQTL SNPs within ±1 Mb of the gene midpoint.

# Create a simulated example dataset:
# - Each gene has 2 cis-eQTLs in the data generation process
#   (but during inference we use only the top SNP, as done in the paper).
# - There are 400 upstream genes and two downstream focal genes:
#   * one is a mediating gene for the trait
#   * the other is a non-mediating gene
# - Both focal genes are influenced by the same set of upstream genes and eQTLs.
# - The only difference is that the mediating gene has a causal effect on the trait,
#   whereas the non-mediating gene has no effect on the trait.

# set up some parameters
npert = 400  # number of perturbed genes
eqtlct = 2  # number of eQTLs per gene
n_eqtl = 200  # eQTL sample size
gwash2g = 0.3  # GWAS heritability per each mediating gene
ngwas = 100000  # GWAS sample size
ppleio = 0  # no pleiotropy
cis = 0  # no cis mediation

# read in the perturbation effects
ref_pert = jnp.array(pd.read_csv("raw/perturb_raw.tsv.gz", sep="\t", header=None))

rng_key, row_key, col_key, rand_key = jax.random.split(rng_key, 4)
col_idx = jax.random.choice(col_key, ref_pert.shape[1], (1,), replace=False)
row_idx = jax.random.choice(row_key, ref_pert.shape[0], (npert,), replace=True)
df_pert = ref_pert[row_idx, col_idx]
tmp_noise = jax.random.normal(rand_key, (df_pert.shape[0],))
df_pert += tmp_noise

# read in the gene file to randomly select as our perturbed genes
eqtl_ref = pd.read_csv("raw/eqtl_gene.tsv.gz", sep="\t")
# read in the gene proportion file to determine how many genes to select from each chromosome
gene_prop = pd.read_csv("raw/chrom_prop.tsv.gz", sep="\t")
# link files
ref_geno = "./plink/geno_chr"
# make LD so that we can generate genotypes later

gene_prop["num"] = (gene_prop.n * npert).astype(int)
if gene_prop.num.sum() != npert:
    diff = npert - gene_prop.num.sum()
    if diff > 0:
        gene_prop.loc[0, "num"] += diff
    else:
        gene_prop.loc[0, "num"] -= diff

LD = []
all_bim = []
gene_list = []
for n_chr in range(22):
    tmp_prop = gene_prop[gene_prop.CHR == n_chr + 1]["num"].values
    tmp_ref = eqtl_ref[eqtl_ref.chrom == n_chr + 1]
    unique_gene = tmp_ref.gene.unique()
    rng_key, sel_key = jax.random.split(rng_key, 2)
    sel_idx = jax.random.choice(
        sel_key, len(unique_gene), (tmp_prop.item(),), replace=False
    )
    unique_gene = unique_gene[sel_idx]
    gene_list.append(unique_gene)
    tmp_ref = tmp_ref[tmp_ref.gene.isin(unique_gene)]
    tmp_ref = tmp_ref.groupby("gene").head(eqtlct)
    bim, _, bed = read_plink(f"{ref_geno}{n_chr + 1}", verbose=False)
    bim = bim[bim.snp.isin(tmp_ref.snp)]
    all_bim.append(bim)

    bed = jnp.array(bed)[bim.i.values].T

    bed -= jnp.mean(bed, axis=0)
    bed /= jnp.std(bed, axis=0)
    tmp_LD = bed.T @ bed / bed.shape[0] + jnp.eye(bed.shape[1]) * 0.001
    LD.append(tmp_LD)
    # inv_LD.append(inv(tmp_LD))

LD = block_diag(*LD)
L_cho = jnp.linalg.cholesky(LD)

# make eQTL summary stats
rng_key, geno_key, h2g_key = jax.random.split(rng_key, 3)

X_eqtl = sim_geno(L_cho, n_eqtl, geno_key)

# https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1006423
h2g_eqtl = jax.random.normal(h2g_key, (npert,)) * 0.023 + 0.033
h2g_eqtl = jnp.where(h2g_eqtl <= 0, h2g_eqtl[h2g_eqtl > 0].min(), h2g_eqtl)

beta_eqtl = []
hat_eqtl = []
z_eqtl = []
qtl_idx = []
for idx in range(npert):
    start = idx * eqtlct
    end = start + eqtlct
    rng_key, qtl_key, expr_key = jax.random.split(rng_key, 3)
    tmp_beta_eqtl = jax.random.normal(qtl_key, (eqtlct,))
    s2g = compute_s2g(L_cho[start:end, start:end], tmp_beta_eqtl)
    tmp_beta_eqtl *= jnp.sqrt(h2g_eqtl[idx] / s2g)
    expr = sim_trait(X_eqtl[:, start:end] @ tmp_beta_eqtl, h2g_eqtl[idx], expr_key)
    tmp_beta_hat = []
    tmp_z = []
    for jdx in range(start, end):
        beta, _, zscore = ols(X_eqtl[:, jdx][:, jnp.newaxis], expr[:, jnp.newaxis])
        tmp_beta_hat.append(beta[0])
        tmp_z.append(zscore[0])
    tmp_beta_hat = jnp.array(tmp_beta_hat)
    tmp_z = jnp.array(tmp_z)

    sel_index = jnp.argmax(tmp_z ** 2)

    qtl_idx.append(sel_index + start)
    hat_eqtl.append(tmp_beta_hat[sel_index])
    z_eqtl.append(tmp_z[sel_index])
    beta_eqtl.append(tmp_beta_eqtl)

beta_eqtl = jnp.array(beta_eqtl).flatten()
qtl_idx = jnp.array(qtl_idx)
df_qtl = pd.concat(all_bim).reset_index(drop=True).iloc[qtl_idx]

z_eqtl = jnp.array(z_eqtl).flatten()
hat_eqtl = jnp.array(hat_eqtl).flatten()

# make GWAS summary stats
(
    rng_key,
    sel_key,
    geno_key,
    sign_key,
    pleio_key1,
    pleio_key2,
    trait_key1,
    trait_key2,
    alpha_key1,
    alpha_key2,
) = jax.random.split(rng_key, 10)

sel_idx = jnp.arange(npert)
num_pert = npert

all_idx = []
for num in range(eqtlct):
    all_idx.append(sel_idx * eqtlct + num)

all_idx = jnp.array(all_idx).flatten().sort()

gene_mask = jnp.ones(df_pert.shape[0], dtype=bool)
gene_mask = gene_mask.at[sel_idx].set(False)
df_pert = df_pert.at[gene_mask].set(0)

snp_mask = jnp.ones(L_cho.shape[0], dtype=bool)
snp_mask = snp_mask.at[all_idx].set(False)
beta_eqtl = beta_eqtl.at[snp_mask].set(0)

h2g_pleio = gwash2g * ppleio
h2g_med = gwash2g * (1 - ppleio)
h2g_trans = h2g_med * (1 - cis)
h2g_cis = h2g_med * cis

all_signs = jax.random.choice(sign_key, jnp.array([-1, 1]), (4,))

# trans effects
X_gwas = sim_geno(L_cho, ngwas, geno_key)
# the last term is alpha
tmp_alpha = jax.random.normal(alpha_key1, (1,))
peg_eff = beta_eqtl * jnp.repeat(df_pert, eqtlct) * tmp_alpha
s2g = compute_s2g(L_cho, peg_eff)
alpha = tmp_alpha * (jnp.sqrt(h2g_trans / s2g) * all_signs[0])
peg_eff *= jnp.sqrt(h2g_trans / s2g) * all_signs[0]

# cis effects
rng_key, cis_key, x_key = jax.random.split(rng_key, 3)
# for med genes genotype
X_gwas2 = jax.random.normal(x_key, (ngwas,))
cis_beta = jax.random.normal(cis_key, (1,)) * jnp.sqrt(h2g_cis)

# pleiotropy effects
# pleio1 = jax.random.normal(pleio_key1, (X_gwas.shape[1],))
# pleio1 = pleio1.at[snp_mask].set(0)
pleio1_num = jax.random.normal(pleio_key1, (1,))
pleio1 = jnp.zeros((X_gwas.shape[1],))
pleio1 = pleio1.at[~snp_mask].set(pleio1_num)
s2g = compute_s2g(L_cho, pleio1)
pleio1 *= jnp.sqrt(h2g_pleio / s2g) * all_signs[1]

trait = sim_trait(X_gwas @ (peg_eff + pleio1) + X_gwas2 * cis_beta, gwash2g, trait_key1)

# make null
null_eff = beta_eqtl * jnp.repeat(jax.random.normal(alpha_key2, (num_pert,)), eqtlct)
s2g = compute_s2g(L_cho, null_eff)
null_adjuster = jnp.sqrt(h2g_med / s2g)
null_eff *= null_adjuster * all_signs[2]

pleio2_num = jax.random.normal(pleio_key2, (1,))
pleio2 = jnp.zeros((X_gwas.shape[1],))
pleio2 = pleio2.at[~snp_mask].set(pleio2_num)
s2g = compute_s2g(L_cho, pleio2)
pleio2 *= jnp.sqrt(h2g_pleio / s2g) * all_signs[3]

null_trait = sim_trait(X_gwas @ (null_eff + pleio2), gwash2g, trait_key2)

hat_gwas = []
se_gwas = []
null_hat_gwas = []
null_se_gwas = []

for idx in range(X_gwas.shape[1]):
    if idx not in qtl_idx:
        continue
    beta, se, _ = ols(X_gwas[:, idx][:, jnp.newaxis], trait[:, jnp.newaxis])
    null_beta, null_se, _ = ols(
        X_gwas[:, idx][:, jnp.newaxis], null_trait[:, jnp.newaxis]
    )
    hat_gwas.append(beta)
    se_gwas.append(se)
    null_hat_gwas.append(null_beta)
    null_se_gwas.append(null_se)

hat_gwas = jnp.array(hat_gwas).flatten()
se_gwas = jnp.array(se_gwas).flatten()
null_hat_gwas = jnp.array(null_hat_gwas).flatten()
null_se_gwas = jnp.array(null_se_gwas).flatten()

# let's save the data
# save the hat_gwas and se_gwas
df_gwas = df_qtl[["chrom", "snp", "pos", "a0", "a1"]].copy()
df_gwas["beta"] = hat_gwas
df_gwas["se"] = se_gwas
df_gwas.to_csv("example_gwas.tsv.gz", sep="\t", index=False)

df_gwas_null = df_qtl[["chrom", "snp", "pos", "a0", "a1"]].copy()
df_gwas_null["beta"] = null_hat_gwas
df_gwas_null["se"] = null_se_gwas
df_gwas_null.to_csv("example_gwas_null.tsv.gz", sep="\t", index=False)

# save the eQTL summary stats
df_eqtl = df_qtl[["chrom", "snp", "pos", "a0", "a1"]].copy()
df_eqtl["beta"] = hat_eqtl
df_eqtl["z"] = z_eqtl
df_eqtl["gene"] = np.concatenate(gene_list)
df_eqtl.to_csv("example_eqtl.tsv.gz", sep="\t", index=False)

# save the perturbation effects
df_pert_final = pd.DataFrame({"perturbations": np.concatenate(gene_list)})
df_pert_final["GENE_ABC"] = np.array(df_pert)
df_pert_final.to_csv("example_perturbation.tsv.gz", sep="\t", index=False)
