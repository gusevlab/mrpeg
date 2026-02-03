.. _Files:

============
Output Files
============

mrpeg produces different types of output files depending on which command is run. Users can compress output files by specifying ``--compress``.

The output files consist of four main types:

#. ``*.mrpeg.tsv`` - Main inference results from ``mrpeg peg``
#. ``*.closest.tsv`` - Closest gene annotations from ``mrpeg closest``
#. ``*.nearby.tsv`` - Nearby gene annotations from ``mrpeg closest --nearby``
#. ``*.signal.tsv`` - GWAS signal summaries from ``mrpeg signal``

.. _mrpegfile:

Main Inference Results
----------------------

The ``mrpeg peg`` command outputs a ``*.mrpeg.tsv`` file containing the mediation effect estimates and p-values for each gene.

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Examples
     - Notes
   * - trait
     - String
     - height, bmi
     - The trait name specified with ``--trait``
   * - tissue
     - String
     - blood, brain
     - The tissue name specified with ``--tissue`` (default: "NA")
   * - gene_name
     - String
     - ENSG00000123456, GENE1
     - The downstream gene name from the perturbation matrix
   * - n_perturb_top
     - Integer
     - 100, 500
     - Number of top perturbation effects used (filtered by ``--top-signal``)
   * - n_perturb_all
     - Integer
     - 1000, 5000
     - Total number of perturbed genes in the analysis
   * - n_gwas_sig
     - Integer
     - 50, 200
     - Number of genome-wide significant SNPs included
   * - gamma
     - Float
     - 0.15, -0.23
     - Estimated mediation effect size
   * - gamma_se
     - Float
     - 0.05, 0.08
     - Standard error of the mediation effect
   * - gamma_p
     - Float
     - 0.001, 0.05
     - P-value based on t-test
   * - gamma_perm_mean
     - Float
     - 0.0, 0.01
     - Mean of the permutation null distribution
   * - gamma_perm_z
     - Float
     - 3.5, -2.1
     - Z-score based on permutation distribution
   * - gamma_null_p
     - Float
     - 0.0001, 0.05
     - P-value based on permutation null distribution (conservative)

.. note::
   The ``gamma_null_p`` values are typically more conservative than ``gamma_p`` as they are based on permutation testing. Use these for controlling family-wise error rate in large-scale analyses.

.. _closestfile:

Closest Gene Annotations
-------------------------

The ``mrpeg closest`` command outputs a ``*.closest.tsv`` file that identifies the closest gene to each genome-wide significant SNP.

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Examples
     - Notes
   * - CHR
     - Integer
     - 1, 22, 23
     - Chromosome number
   * - SNP
     - String
     - rs12345, snp_1_1000000
     - SNP identifier
   * - BP
     - Integer
     - 1000000
     - Base pair position
   * - BETA
     - Float
     - 0.05, -0.10
     - GWAS effect size
   * - SE
     - Float
     - 0.02
     - GWAS standard error
   * - GENE
     - String
     - ENSG00000123456, GENE1
     - Closest gene identifier
   * - trait
     - String
     - height, bmi
     - The trait name specified with ``--trait``

.. note::
   If a SNP falls within a gene body (between TSS and TES), that gene is assigned. Otherwise, the gene with the closest transcription start or end site is assigned. Multiple genes may be assigned to a single SNP if they are equidistant.

.. _nearbyfile:

Nearby Gene Annotations
-----------------------

The ``mrpeg closest --nearby`` command outputs a ``*.nearby.tsv`` file that identifies all genes within a specified window of each genome-wide significant SNP.

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Examples
     - Notes
   * - CHR
     - Integer
     - 1, 22, 23
     - Chromosome number
   * - TSS
     - Integer
     - 1000000
     - Transcription start site position
   * - TES
     - Integer
     - 1050000
     - Transcription end site position
   * - GENE
     - String
     - ENSG00000123456, GENE1
     - Gene identifier
   * - snp
     - String
     - rs12345, snp_1_1000000
     - SNP identifier within window
   * - trait
     - String
     - height, bmi
     - The trait name specified with ``--trait``

.. note::
   The window size is specified with ``--window`` in kilobases. Genes are included if their transcribed region (TSS to TES) overlaps with the window centered on each SNP.

.. _signalfile:

GWAS Signal Summaries
---------------------

The ``mrpeg signal`` command outputs a ``*.signal.tsv`` file that summarizes GWAS test statistics within each gene annotation.

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Examples
     - Notes
   * - ANNO
     - String
     - ENSG00000123456, GENE1
     - Gene or annotation identifier
   * - CHR
     - Integer
     - 1, 22, 23
     - Chromosome number
   * - P0
     - Integer
     - 1000000
     - Annotation start position (without flanking region)
   * - P1
     - Integer
     - 1050000
     - Annotation end position (without flanking region)
   * - P0_FLANK
     - Integer
     - 950000
     - Annotation start position (with flanking region)
   * - P1_FLANK
     - Integer
     - 1100000
     - Annotation end position (with flanking region)
   * - mean_chisq
     - Float
     - 5.2
     - Mean chi-square statistic (Z²) across SNPs in annotation
   * - sd_chisq
     - Float
     - 2.1
     - Standard deviation of chi-square statistics
   * - median_chisq
     - Float
     - 4.8
     - Median chi-square statistic
   * - max_chisq
     - Float
     - 15.3
     - Maximum chi-square statistic
   * - min_chisq
     - Float
     - 0.5
     - Minimum chi-square statistic
   * - qtl1_chisq
     - Float
     - 2.1
     - First quartile (25th percentile) of chi-square statistics
   * - qtl3_chisq
     - Float
     - 7.8
     - Third quartile (75th percentile) of chi-square statistics
   * - mean_z
     - Float
     - 1.5
     - Mean Z-score across SNPs in annotation
   * - sd_z
     - Float
     - 1.2
     - Standard deviation of Z-scores
   * - median_z
     - Float
     - 1.3
     - Median Z-score
   * - max_z
     - Float
     - 4.2
     - Maximum Z-score
   * - min_z
     - Float
     - -0.5
     - Minimum Z-score
   * - qtl1_z
     - Float
     - 0.5
     - First quartile (25th percentile) of Z-scores
   * - qtl3_z
     - Float
     - 2.1
     - Third quartile (75th percentile) of Z-scores
   * - count
     - Integer
     - 150
     - Number of SNPs within the annotation
   * - trait
     - String
     - height, bmi
     - The trait name specified with ``--trait``

.. note::
   The flanking region is specified with ``--window`` in kilobases. GWAS signals are summarized using Z-scores computed as BETA/SE from the input GWAS summary statistics.

.. _optionalfiles:

Optional Annotation Files
--------------------------

When running ``mrpeg signal`` with the ``--snps-anno`` flag, two additional files are generated:

#. ``*.full.anno.tsv.gz`` - All SNPs with their annotation assignments (before filtering)
#. ``*.filter.anno.tsv.gz`` - SNPs with their annotation assignments (after filtering by ``--split``)

These files contain SNP-level information including:

- CHR, BP, SNP - SNP identifiers and position
- Z - GWAS Z-score
- ANNO - Assigned annotation/gene
- trait - Trait name

The ``--split`` parameter controls whether SNPs can be assigned to multiple overlapping annotations.

.. _logger:

Logger
------

All mrpeg commands produce logging output that tracks the inference process. By default, logs are printed to the console. You can control logging verbosity with:

- ``--quiet`` or ``-q`` - Suppress most log messages
- ``--verbose`` or ``-v`` - Show detailed log messages

The logs include:

- Data loading progress
- Number of SNPs, genes, and samples processed
- Filtering statistics (e.g., ambiguous SNPs removed, LD pruning results)
- Computation progress for permutation testing
- Error messages and warnings
- Final output file locations
