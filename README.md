[![Documentation-webpage](https://img.shields.io/badge/Docs-Available-brightgreen)](https://gusevlab.github.io/mrpeg/)
[![Github](https://img.shields.io/github/stars/gusevlab/mrpeg?style=social)](https://github.com/gusevlab/mrpeg)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# mrpeg

mrpeg is a Python software to perform association test integrating perturbational screens, eQTL, and GWAS summary data to identify mediating genes of complex traits.

``` diff
- We detest usage of our software or scientific outcome to promote racial discrimination.
```

The Mr. PEG manuscript is described in

[**Integrating perturbational screens, eQTL, and GWAS data identifies mediating genes for complex traits**](https://www.medrxiv.org/content/10.64898/2026.01.05.26343421v1)

Zeyun Lu, Yi Ding, Nathan LaPierre, Lili Wang, Douglas Yao, Nicholas Mancuso, Alexander Gusev

Check [here](https://gusevlab.github.io/mrpeg/) for full
documentation.

  [**Installation**](#installation)
  | [**Example**](#get-started-with-example)
  | [**Version History**](#version-history)
  | [**Support**](#support)


## Installation

1. **Before installation**, we *highly* recommend to create a new environment using [conda](https://docs.conda.io/en/latest/) so that it will not affect the software versions of the other projects. For example, use following codes:

    ```bash
    conda create -n env-mrpeg python=3.10
    conda activate env-mrpeg
    ```

2. If you are using a Mac with an Apple M1 or newer chip, you should initiate your conda using `miniforge` to ensure compatibility (see this [link](https://github.com/google/jax/issues/5501) for previous issue). **On most HPC systems**, this is usually not necessary.

3. Last, users can download the latest repository and then use `pip`:

    ``` bash
    git clone https://github.com/gusevlab/mrpeg.git
    cd mrpeg
    pip install .
    ```

## Get Started with Example

mrpeg software is very easy to use:

For performing inference:
``` bash
cd ./data/
mrpeg peg --gwas example_gwas.tsv.gz \
  --eqtl example_eqtl.tsv.gz \
  --perturb example_perturbation.tsv.gz \
  --gwas-cols chrom snp a1 a0 beta se \
  --eqtl-cols chrom snp a1 a0 z gene \
  --ref-geno plink/geno_chr\* \
  --trait "mediating_gene" \
  --top-signal 1 \
  -o tmp_results_mediating_gene
```

We also implement a function to compute the GWAS signals given gene annotations.

``` bash
cd ./data/
mrpeg signal --gwas example_gwas.tsv.gz \
  --gwas_cols chrom snp pos beta se \
  --ref ref_gene_info.tsv.gz \
  --ref_cols CHR P_MID_FLANK0 P_MID_FLANK1 ID2 \
  --window 0 \
  --trait example \
  --chr 1 \
  -o tmp_results_signal
```

See [here](https://gusevlab.github.io/mrpeg/) for more details on how
to use mrpeg.

If you want to use in-software mrpeg inference function, you can use
following Python code as an example:

``` python
from mrpeg.peg import infer_peg
# betas is a numpy array of GWAS effects (k by 1)
# ses is a numpy array of GWAS standard errors (k by 1)
# eqtl is a numpy array of eQTL effects (k by 1)
# perturbs is a numpy array of perturbation effects (k by t) of k perturbed genes on t downstream genes
# ld is a numpy array (k by k) of LD across SNPs
infer_peg(beta=betas, se=ses, eqtl=eqtls, perturb=perturbs, ld=lds)
```

You can customize this function with your own ideas!

## Pre-computed Perturb-seq Effect Size Matrices

We provide two pre-computed gene-to-gene perturbation effect size matrices in the [misc/](misc/) folder that users can plug directly into `mrpeg peg` via `--perturb`. Both were estimated with [FR-Perturb](https://github.com/douglasyao/fr-perturb):

- [misc/Yao_MCL.tsv.gz](misc/Yao_MCL.tsv.gz) — Macrophage cell line (MCL) screen from [Yao et al., *Nature Biotechnology*, 2023](https://www.nature.com/articles/s41587-023-01964-9). 600 upstream (perturbed) genes × 16,952 downstream genes.
- [misc/Replogle_K562.tsv.gz](misc/Replogle_K562.tsv.gz) — K562 genome-wide essential gene screen from [Replogle et al., *Cell*, 2022](https://www.cell.com/cell/fulltext/S0092-8674(22)00597-9). 2,058 upstream genes × 8,563 downstream genes.

## Version History

| Version | Description                                                                                                                                                                                                                                                                                     |
|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0.2     | Added input format specifications, troubleshooting guide, and performance guide. Fixed documentation errors carried over from a previous project. Added missing `intervaltree` dependency. Added type hints to `closest` and `signal` modules. Added test suite with 50 unit tests. **This update was completely done using Claude Code with human verification.** |
| 0.1     | Initial Release                                                                                                                                                                                                                                                                                 |

## Support

For any questions, comments, bug reporting, and feature requests, please contact Zeyun Lu (<zeyun_lu@dfci.harvard.edu>) and
Sasha Gusev (<alexander_gusev@dfci.harvard.edu>), and open a new thread in the [Issue
Tracker](https://github.com/gusevlab/mrpeg/issues).


------------------------------------------------------------------------

This project has been set up using PyScaffold 4.1.1. For details and
usage information on PyScaffold see <https://pyscaffold.org/>.
