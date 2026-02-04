.. _troubleshooting:

=====================
Troubleshooting
=====================

This page collects common problems users encounter and how to resolve them.

-----------------------------------------------------------
Installation Issues
-----------------------------------------------------------

**Apple M1/M2 Mac: JAX installation fails**

JAX and jaxlib binaries are not always available for the ARM architecture
that Apple Silicon uses.  The recommended workaround is to initialize conda
with ``miniforge`` (which ships ARM-native builds).

**``ModuleNotFoundError`` after ``pip install .``**

Make sure you activated the conda environment before running ``pip install``:

.. code-block:: bash

   conda activate env-mrpeg
   pip install .

-----------------------------------------------------------
Runtime Errors
-----------------------------------------------------------

**``No GWAS file located. Check your input.``**

The path you passed to ``--gwas`` does not exist or is not readable.  Double-
check the path, including the working directory:

.. code-block:: bash

   ls -la path/to/your_gwas.tsv.gz

**``Specified GWAS columns are not in the GWAS data.``**

The column names you gave to ``--gwas-cols`` do not match the headers in your
file.  Print the headers and compare:

.. code-block:: bash

   zcat your_gwas.tsv.gz | head -1

Column names are **case-sensitive**.  If your file uses ``Chr`` but you wrote
``CHR``, the match will fail.

**``GWAS data doesn't contain any SNPs after filtering on threshold.``** (``mrpeg signal``)

Same cause as above.  ``mrpeg signal`` defaults ``--threshold`` to 1.0
(keep all SNPs), so this error usually means the data itself is empty or
all SE values are ≤ 0.

**``top signals must be between 0 and 1``**

``--top-signal`` accepts a value in [0, 1].  A value of ``1`` keeps all
perturbation effect pairs; ``0.01`` (the default) keeps the top 1 %.

**``The seed specified for randomization is invalid.``**

``--seed`` must be a strictly positive integer.  Use any value ≥ 1:

.. code-block:: bash

   mrpeg peg --seed 42 ...

**``The permutation number is invalid.``**

``--perm-number`` must be a positive integer.  The default is 1000.

-----------------------------------------------------------
Data Format Issues
-----------------------------------------------------------

**Ambiguous alleles are being removed**

By default mrpeg drops SNPs where the allele pair is A/T, T/A, C/G, or G/C
because strand orientation cannot be determined from alleles alone.  If you
are confident your GWAS and eQTL files are on the same strand, keep these
SNPs:

.. code-block:: bash

   mrpeg peg --keep-ambiguous ...

**Gene names don't match between eQTL and perturbation files**

mrpeg merges eQTL results with the perturbation matrix on gene name.  If the
eQTL file uses ENSEMBL IDs (``ENSG…``) and the perturbation file uses gene
symbols (``BRCA1``), the merge will yield zero shared genes.  Harmonize both
files to the same naming convention before running mrpeg.

**Allele flips between GWAS and eQTL**

mrpeg automatically detects and corrects allele flips (e.g., GWAS reports
A1=A/A0=G while eQTL reports A1=G/A0=A).  No action is needed.  However,
if the alleles are entirely different between the two files (e.g., due to
different reference panels), those SNPs will be classified as "wrong" and
dropped.  Check that both summary statistics are reported against the same
reference panel.

**SE column contains zeros or negative values**

Rows with SE ≤ 0 are removed automatically with a log message.  If a large
fraction of your data is removed, investigate the source of your summary
statistics.

**Reference file columns in wrong order**

``--ref-cols`` is **positional**: the first name maps to CHR, the second to
the start coordinate, the third to the end coordinate, and the fourth to the
annotation/gene name.  The order you specify must match this mapping
regardless of the physical column order in the file.

-----------------------------------------------------------
Performance Issues
-----------------------------------------------------------

**Inference is slow / running out of memory**

Each downstream gene is tested independently, so memory usage scales with
the number of instrument SNPs for each gene, not the full genome.  If you
are still hitting memory limits:

* Increase ``--top-signal`` filtering (e.g., ``0.001``) to reduce the number
  of non-zero perturbation entries and thus the number of instrument SNPs.
* ``--perm-number`` controls the number of permutations for the null
  distribution.  Reducing this speeds up inference linearly.
* Use 32-bit precision: ``--jax-precision 32``.  This halves memory but may
  produce slightly less accurate estimates.

**``mrpeg signal`` is slow on the full genome**

``mrpeg signal`` constructs an interval tree per chromosome and queries every
significant SNP.  Process one chromosome at a time:

.. code-block:: bash

   for chr in $(seq 1 22); do
     mrpeg signal --chr $chr ...
   done

-----------------------------------------------------------
Unexpected Results
-----------------------------------------------------------

**All gamma p-values are 1.0**

This typically means the permutation null distribution dominates the observed
effect.  Possible causes:

* The perturbation effects for the gene in question are very noisy or near
  zero after ``--top-signal`` filtering.
* The GWAS sample is underpowered for this trait/gene combination.
* The eQTL instruments are weak (few significant cis-eQTLs).

Try running with ``--top-signal 1`` (no filtering) to see whether the
perturbation signal itself is informative.

**Results differ between runs**

``infer_peg`` uses a permutation test seeded by ``--seed`` (default 12345).
Results are fully reproducible as long as the seed and all input data remain
the same.  If you changed the input data or the seed, results will differ.

**``gamma_perm_z`` is very large but ``gamma_p`` is not significant**

``gamma_p`` is the Wald test p-value (analytical), while ``gamma_null_p`` is
the permutation-based p-value.  The permutation null is intentionally
conservative.  Report ``gamma_null_p`` as the primary evidence for mediation;
use ``gamma_p`` for effect-size interpretation.
