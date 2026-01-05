[![Documentation-webpage](https://img.shields.io/badge/Docs-Available-brightgreen)](https://gusevlab.github.io/sushie/)
[![Github](https://img.shields.io/github/stars/gusevlab/mrpeg?style=social)](https://github.com/gusevlab/mrpeg)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

# mrpeg

mrpeg is a Python software to perform association test integrating perturbational screens, eQTL, and GWAS summary data to identify mediating genes of complex traits.

``` diff
- We detest usage of our software or scientific outcome to promote racial discrimination.
```

Mr. PEG manuscript is in progress.

Check [here](https://gusevlab.github.io/mrpeg/) for full
documentation.

  [**Installation**](#installation)
  | [**Example**](#get-started-with-example)
  | [**Notes**](#notes)
  | [**Version History**](#version-history)
  | [**Support**](#support)
  | [**Other Software**](#other-software)

## Installation

1. **Before installation**, we *highly* recommend to create a new environment using [conda](https://docs.conda.io/en/latest/) so that it will not affect the software versions of the other projects. For example, use following codes:

    ```bash
    conda create -n env-mrpeg python=3.8
    ```

    *We currently only support Python3.8+.*

2. If you are using a Mac with an Apple M1 or newer chip, you should install `cbgen` package or other required pacakges from conda-forge first to ensure compatibility (see this [link](https://github.com/google/jax/issues/5501) for previous issue). One easy workaround is to initiate your conda using `miniforge`. **On most HPC systems**, this is usually not necessary.

    ```bash
    conda install -c conda-forge cbgen
    ```

3. Last, users can download the latest repository and then use `pip`:

    ``` bash
    git clone https://github.com/gusevlab/mrpeg.git
    cd mrpeg
    pip install .
    ```

## Get Started with Example

mrpwg software is very easy to use:

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

We also implement two functions for gene annotation.

1. Find the GWAS closest genes:

``` bash
cd ./data/
mrpeg closest --gwas example_gwas.tsv.gz \
  --gwas_cols chrom snp pos beta se \
  --ref ref_gene_info.tsv.gz \
  --ref_cols CHR TSS TES ID2 \
  --trait example \
  -o tmp_results_closest
```

2. Compute the GWAS signals given gene annotations.

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

## Version History

| Version | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0.1     | Initial Release                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

## Support

For any questions, comments, bug reporting, and feature requests, please contact Zeyun Lu (<zeyun_lu@dfci.harvard.edu>) and
Sasha Gusev (<alexander_gusev@dfci.harvard.edu>), and open a new thread in the [Issue
Tracker](https://github.com/mancusolab/sushie/issues).


------------------------------------------------------------------------

This project has been set up using PyScaffold 4.1.1. For details and
usage information on PyScaffold see <https://pyscaffold.org/>.
