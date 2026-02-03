.. _performance:

=====================
Performance Guide
=====================

This page describes the computational requirements of mrpeg and strategies
for running it efficiently on different hardware and dataset sizes.

-----------------------------------------------------------
Computational Overview
-----------------------------------------------------------

mrpeg has three subcommands with different computational profiles:

``mrpeg peg`` (core inference)
   The most compute-intensive command.  For each downstream (focal) gene it
   solves a weighted least-squares problem and runs a permutation test.  The
   dominant cost is the permutation loop, which is accelerated by JAX and can
   run on GPU or TPU when available.

``mrpeg closest``
   Lightweight.  Reads GWAS summary statistics, identifies significant
   regions, and finds the nearest gene in a reference annotation.  Runtime
   is dominated by file I/O.

``mrpeg signal``
   Moderate.  Constructs an interval tree per chromosome and queries every
   significant SNP.  Memory usage scales with the number of annotations and
   SNPs on the queried chromosomes.

-----------------------------------------------------------
Scaling Behaviour (``mrpeg peg``)
-----------------------------------------------------------

The key dimensions that affect runtime and memory are:

* **k** — number of instrument SNPs (shared between GWAS and eQTL after
  filtering).  Larger k increases the size of the LD matrix (k × k) and the
  per-permutation matrix operations.
* **t** — number of downstream (focal) genes to test.  Each gene is tested
  independently, so total wall-clock time scales linearly with t.
* **P** — number of permutations (``--perm-number``, default 1000).  Each
  permutation re-solves the estimator with shuffled perturbation effects.
  Runtime scales linearly with P.

Approximate scaling:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - k (SNPs)
     - t (genes)
     - P (perms)
     - Relative cost
   * - 100
     - 50
     - 1000
     - Baseline
   * - 500
     - 50
     - 1000
     - ~5×
   * - 100
     - 200
     - 1000
     - ~4×
   * - 100
     - 50
     - 5000
     - ~5×

.. note::

   The ``--top-signal`` parameter is the single most effective lever for
   controlling k.  Lowering it from 1.0 to 0.01 can reduce the effective
   number of instrument SNPs per gene by an order of magnitude.

-----------------------------------------------------------
Memory Requirements
-----------------------------------------------------------

The largest in-memory object is the LD matrix (k × k floats).  With 64-bit
precision (the default):

* k = 100  →  ~80 KB
* k = 1 000  →  ~8 MB
* k = 10 000  →  ~800 MB

If memory is tight, switch to 32-bit precision:

.. code-block:: bash

   mrpeg peg --jax-precision 32 ...

This halves all floating-point storage but may reduce numerical accuracy
slightly.  In practice the difference in final p-values is negligible for
typical effect sizes.

-----------------------------------------------------------
GPU / TPU Acceleration
-----------------------------------------------------------

JAX can offload computation to a GPU or TPU via the ``--platform`` flag:

.. code-block:: bash

   # Use GPU
   mrpeg peg --platform gpu ...

   # Use TPU (e.g., on Google Cloud)
   mrpeg peg --platform tpu ...

The default is ``cpu``.  GPU acceleration is most beneficial when k is large
(≥ a few hundred) and many permutations are requested.  For small k the
overhead of data transfer to the device can outweigh the speedup.

Install the appropriate ``jaxlib`` backend before using GPU or TPU:

* **CUDA GPU**: ``pip install jaxlib==VERSION+cu118 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html``
* **TPU**: follow the `JAX TPU guide <https://github.com/google/jax/blob/main/docs/tpu_tutorial.md>`_.

-----------------------------------------------------------
Optimisation Checklist
-----------------------------------------------------------

Apply these strategies in order of likely impact:

1. **Filter perturbation effects aggressively.**  ``--top-signal 0.01``
   (the default) already keeps only the top 1 % of effect pairs.  If you
   still have too many SNPs, try ``0.001``.

2. **Increase ``--min-snps``.**  Genes with very few instrument SNPs after
   filtering are unlikely to yield reliable estimates.  The default of 10
   already skips these; raising it to 20 or 50 can further reduce workload.

3. **Run one chromosome at a time** (for ``mrpeg signal``).  Use ``--chr``
   to limit the scope of each invocation:

   .. code-block:: bash

      for chr in $(seq 1 22); do
        mrpeg signal --chr $chr -o results_chr${chr} ...
      done

4. **Use GPU when available.**  Even a modest consumer GPU (e.g., NVIDIA
   RTX 3060) can provide a substantial speedup for the permutation loop.

5. **Lower precision if acceptable.**  ``--jax-precision 32`` halves memory
   and can speed up matrix operations on both CPU and GPU.

6. **Reduce permutation count for exploratory runs.**  Use
   ``--perm-number 100`` during development or screening; increase to 1000
   or more for final publication-quality results.

-----------------------------------------------------------
Benchmarks (Example Data)
-----------------------------------------------------------

The bundled example dataset contains 400 perturbed genes, 1 downstream gene,
and ~400 instrument SNPs distributed across 22 chromosomes.  On a standard
desktop CPU (Intel i7, 64-bit precision, 1000 permutations) the ``mrpeg peg``
command completes in under one minute.

For real-world datasets the runtime will vary significantly depending on the
number of downstream genes and the effective instrument count after
``--top-signal`` filtering.  Profile a single gene first by subsetting your
perturbation matrix to one column, then extrapolate.
