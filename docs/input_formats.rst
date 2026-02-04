.. _input_formats:

=====================
Input File Formats
=====================

mrpeg requires several input files depending on the subcommand. This page
describes the expected format for each file type, including required columns,
data types, and concrete examples drawn from the bundled example data in
``data/``.

-----------------------------------------------------------
GWAS Summary Statistics
-----------------------------------------------------------

Used by: ``mrpeg peg``, ``mrpeg signal``

A tab-delimited text file (plain or gzip-compressed) containing per-SNP
effect-size estimates from a genome-wide association study.  The column names
do **not** need to match the defaults shown below — use ``--gwas-cols`` to
specify the actual header names in your file.

``mrpeg peg`` expects **six** columns (specified via ``--gwas-cols``):

.. list-table::
   :header-rows: 1
   :widths: 15 12 60

   * - Column
     - Type
     - Description
   * - CHR
     - int
     - Autosomal chromosome number (1–22).  Non-autosomal entries are dropped.
   * - SNP
     - str
     - SNP identifier (e.g. ``rs6678318``).  Used as the merge key with the eQTL file.
   * - A1
     - str
     - Effect allele (the allele whose effect ``BETA`` measures).
   * - A0
     - str
     - Non-effect (reference) allele.  Together with ``A1``, used for allele
       harmonization with the eQTL file.
   * - BETA
     - float
     - Estimated effect size for allele ``A1``.
   * - SE
     - float
     - Standard error of ``BETA``.  Must be strictly positive; rows with
       SE ≤ 0 are removed automatically.

``mrpeg signal`` expects **five** columns (the two allele columns are not
needed because no allele harmonization is performed):

.. list-table::
   :header-rows: 1
   :widths: 15 12 60

   * - Column
     - Type
     - Description
   * - CHR
     - int
     - Autosomal chromosome number (1–22).
   * - SNP
     - str
     - SNP identifier.
   * - BP
     - int
     - Genomic base-pair position on chromosome ``CHR``.
   * - BETA
     - float
     - Estimated effect size.
   * - SE
     - float
     - Standard error of ``BETA`` (> 0).

Example (``data/example_gwas.tsv.gz``, first three rows):

.. code-block:: text

   chrom  snp          pos      a0  a1  beta      se
   1      rs6678318    1030633  A   G   0.005895  0.003162
   1      rs2785581    6177898  A   G  -0.004418  0.003162
   1      rs10746477   8195724  T   C   0.011452  0.003162

To use this file with ``mrpeg peg``:

.. code-block:: bash

   mrpeg peg --gwas example_gwas.tsv.gz \
     --gwas-cols chrom snp a1 a0 beta se ...

To use with ``mrpeg signal`` (which needs a ``BP`` column instead of alleles):

.. code-block:: bash

   mrpeg signal --gwas example_gwas.tsv.gz \
     --gwas-cols chrom snp pos beta se ...

-----------------------------------------------------------
eQTL Summary Statistics
-----------------------------------------------------------

Used by: ``mrpeg peg``

A tab-delimited text file (plain or gzip-compressed) containing per-SNP,
per-gene eQTL association results.  Each row records the association between
one SNP and one gene.  Multiple rows can share the same SNP (one per gene)
or the same gene (one per SNP).

``mrpeg peg`` expects **six** columns (specified via ``--eqtl-cols``):

.. list-table::
   :header-rows: 1
   :widths: 15 12 60

   * - Column
     - Type
     - Description
   * - CHR
     - int
     - Autosomal chromosome number (1–22).
   * - SNP
     - str
     - SNP identifier.  Must match the identifiers in the GWAS file so that
       the two datasets can be merged on SNP.
   * - A1
     - str
     - Effect allele in the eQTL study.  Used together with ``A0`` to detect
       and correct allele flips relative to the GWAS file.
   * - A0
     - str
     - Non-effect allele in the eQTL study.
   * - Z
     - float
     - Z-score (or t-statistic) of the eQTL association.  If your eQTL
       results provide a beta and SE instead, compute ``Z = BETA / SE``
       before running mrpeg.
   * - GENE
     - str
     - Gene identifier (symbol or ENSEMBL ID).  Must match the row labels in
       the perturbation matrix.

Example (``data/example_eqtl.tsv.gz``, first three rows):

.. code-block:: text

   chrom  snp          pos      a0  a1  beta      z         gene
   1      rs6678318    1030633  A   G  -0.161025 -2.295787  FASLG
   1      rs2785581    6177898  A   G  -0.207129 -2.979178  TAGLN2
   1      rs10746477   8195724  T   C   0.173091  2.472928  SOX13

To use this file:

.. code-block:: bash

   mrpeg peg --eqtl example_eqtl.tsv.gz \
     --eqtl-cols chrom snp a1 a0 z gene ...

.. note::

   The ``pos`` and ``beta`` columns in the example eQTL file are informational
   and are **not** consumed by mrpeg — only the six columns specified via
   ``--eqtl-cols`` are read.

-----------------------------------------------------------
Perturbation Effect Matrix
-----------------------------------------------------------

Used by: ``mrpeg peg``

A tab-delimited text file (plain or gzip-compressed) encoding the
gene-to-gene effect matrix from a Perturb-seq (or similar CRISPR-based
perturbational screen).

* **Rows** — perturbed (upstream) genes.  The first column is a header
  (any name, e.g. ``perturbations``) followed by one gene identifier per row.
  These identifiers must match the ``GENE`` column in the eQTL file.
* **Columns** — downstream (target / focal) genes.  Each column header is a
  gene identifier.  mrpeg tests each of these genes as a potential mediator of
  the GWAS trait.
* **Values** — continuous effect-size estimates (floats).  Missing values
  should be encoded as ``NaN`` or left empty; they are handled during
  preprocessing.

Example (``data/example_perturbation.tsv.gz``, first three rows):

.. code-block:: text

   perturbations  GENE_ABC
   FASLG         -0.051150
   TAGLN2        -1.046032
   SOX13         -1.182132

In this example there are 400 perturbed genes (rows) and 1 downstream gene
(``GENE_ABC``, column).  A real-world dataset will typically have hundreds to
thousands of both.

.. note::

   The ``--top-signal`` parameter (default 0.01) keeps only the top fraction
   of perturbation effects by absolute value.  For example, with a 500 × 200
   matrix and ``--top-signal 0.01``, only the strongest 1 % of the 100 000
   entries (1 000 values) are retained; the rest are set to zero before
   inference.  Downstream genes that end up with fewer than ``--min-snps``
   (default 10) non-zero instrument SNPs are skipped entirely.

-----------------------------------------------------------
Gene Annotation / Reference File
-----------------------------------------------------------

Used by: ``mrpeg signal``

A tab-delimited file (plain or gzip-compressed) that maps genes or
genomic annotations to chromosomal coordinates.

``mrpeg signal`` expects **four** columns (specified via ``--ref-cols``)
that define annotation regions:

.. list-table::
   :header-rows: 1
   :widths: 15 12 60

   * - Column
     - Type
     - Description
   * - CHR
     - int
     - Chromosome number (1–22).
   * - START
     - int
     - Start position of the annotation region.  If ``START`` and ``END``
       refer to the same column (single base-pair annotations), set
       ``--window`` > 0 to expand around each position.
   * - END
     - int
     - End position of the annotation region.
   * - ANNO
     - str
     - Annotation name.  Used as the row identifier in the output
       ``.signal.tsv`` file.

Example (``data/ref_gene_info.tsv.gz``, first three rows):

.. code-block:: text

   ID2              NAME    CHR  TSS     TES     P_MID_FLANK0  P_MID_FLANK1
   ENSG00000187634  SAMD11  1    859307  879961  369634        1369634
   ENSG00000188976  NOC2L   1    879583  894688  387135        1387135
   ENSG00000187961  KLHL17  1    895963  901099  398531        1398531

This file contains more columns than mrpeg needs.  Use ``--ref-cols`` to
select the relevant four:

.. code-block:: bash

   mrpeg signal --ref ref_gene_info.tsv.gz \
     --ref-cols CHR P_MID_FLANK0 P_MID_FLANK1 ID2 ...

-----------------------------------------------------------
Reference Genotype Files (PLINK format)
-----------------------------------------------------------

Used by: ``mrpeg peg`` (via ``--ref-geno``)

mrpeg computes the LD (linkage disequilibrium) matrix from a set of PLINK
binary files.  You need one triplet of files per autosomal chromosome:

.. list-table::
   :header-rows: 1
   :widths: 15 60

   * - Extension
     - Description
   * - ``.bed``
     - Binary genotype matrix (individuals × SNPs).
   * - ``.bim``
     - SNP information: chromosome, SNP ID, genetic distance, base-pair
       position, allele 1, allele 2.  The ``SNP ID`` column is used to match
       SNPs to the GWAS and eQTL files.
   * - ``.fam``
     - Individual (sample) information.  The number of rows determines the
       LD sample size.

Provide the file path with a ``*`` (or ``\*``) as a placeholder for the
chromosome number:

.. code-block:: bash

   mrpeg peg --ref-geno plink/geno_chr\* ...

mrpeg will expand the wildcard for chromosomes 1–22 and read each triplet
independently.  The example data ships with 22 triplets in ``data/plink/``,
generated from 1000 Genomes EUR samples.

.. note::

   Only the SNPs that appear in **both** the GWAS and eQTL files are
   extracted from the PLINK files.  The remaining SNPs are ignored.

-----------------------------------------------------------
Keep File (optional)
-----------------------------------------------------------

Used by: ``mrpeg signal`` (``--keep``)

A single-column, tab-delimited text file with **no header**.  Each row
contains one gene or annotation name that you want to restrict the analysis
to.  Any genes or annotations in the reference file that are not present in
the keep file are dropped before processing.

Example:

.. code-block:: text

   SAMD11
   NOC2L
   KLHL17
