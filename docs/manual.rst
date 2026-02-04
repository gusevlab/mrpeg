.. _manual:

=================
Users Manual
=================

Installation
============

1. **Before installation**, we *highly* recommend creating a new environment using `conda <https://docs.conda.io/en/latest/>`_ so that it will not affect the software versions of other projects. For example, run:

   .. code-block:: bash

      conda create -n env-mrpeg python=3.10

2. If you are using a Mac with an Apple M1 or newer chip, you should  initialize conda using ``miniforge`` to ensure compatibility (see this `link <https://github.com/google/jax/issues/5501>`_ for a previous issue). **On most HPC systems**, this is usually not necessary.

3. Finally, download the latest repository and install via ``pip``:

   .. code-block:: bash

      git clone https://github.com/gusevlab/mrpeg.git
      cd mrpeg
      pip install .

Data Preparation
================

mrpeg requires users to input GWAS summary data, eQTL summary data, and gene-to-gene effects matrix to perform inference.

Users can check testing data in :ref:`Testing Data` section to see the format requirement.


Testing Data
------------

We provide example data in ``./data/`` folder to test out mrpeg.

We simulated GWAS summary and eQTL summary data using real cis-eQTLs identified in eQTLGen whose genotypes are from 1000G project.

Using ``./data/make_example.py``, we simulated GWAS, eQTL summary data. Users can run the script in ``./data/run_example.sh``.

Usage
=====

mrpeg software is very easy to use.

Performing inference
--------------------

.. code-block:: bash

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

Gene annotation
---------------

We also implement two functions for gene annotation.

1. Find the GWAS closest genes:

   .. code-block:: bash

      cd ./data/
      mrpeg closest --gwas example_gwas.tsv.gz \
        --gwas-cols chrom snp pos beta se \
        --ref ref_gene_info.tsv.gz \
        --ref-cols CHR TSS TES ID2 \
        --trait example \
        -o tmp_results_closest

2. Compute the GWAS signals given gene annotations:

   .. code-block:: bash

      cd ./data/
      mrpeg signal --gwas example_gwas.tsv.gz \
        --gwas-cols chrom snp pos beta se \
        --ref ref_gene_info.tsv.gz \
        --ref-cols CHR P_MID_FLANK0 P_MID_FLANK1 ID2 \
        --window 0 \
        --trait example \
        --chr 1 \
        -o tmp_results_signal
