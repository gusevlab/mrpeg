.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly


    .. image:: https://readthedocs.org/projects/sushie/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://sushie.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/sushie/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/sushie

    .. image:: https://img.shields.io/conda/vn/conda-forge/sushie.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/sushie
    .. image:: https://pepy.tech/badge/sushie/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/sushie



.. image:: https://img.shields.io/badge/Docs-Available-brightgreen
        :alt: Documentation-webpage
        :target: https://gusevlab.github.io/mrpeg/

.. image:: https://img.shields.io/pypi/v/mrpeg.svg
           :alt: PyPI-Server
           :target: https://pypi.org/project/sushie/

.. image:: https://img.shields.io/github/stars/gusevlab/mrpeg?style=social
        :alt: Github
        :target: https://github.com/gusevlab/mrpeg

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :alt: License
    :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/


=======
Mr. PEG
=======
Mr. PEG is a Python software to perform association test integrating Perturb-Seq (P), cis-eQTL (E), and GWAS (G) summary data to identify mediating genes for complex traits. **The manuscript is in progress.**

Check `here <https://gusevlab.github.io/mrpeg/>`_ for full documentation.

|Installation|_ | |Example|_ | |Notes|_ | |Version|_ | |Support|_ | |Other Software|_

=================

.. _Installation:
.. |Installation| replace:: **Installation**

Installation
============

..

Users can download the latest repository and then use ``pip``:

.. code:: bash

    git clone https://github.com/gusevlab/mrpeg.git
    cd mrpeg
    pip install .

*We currently only support Python3.8+.*

Before installation, we recommend to create a new environment using `conda <https://docs.conda.io/en/latest/>`_ so that it will not affect the software versions of the other projects.

.. _Example:
.. |Example| replace:: **Example**

Get Started with Example
========================
Mr. PEG software is very easy to use. It provides three functions:

1. **peg**: Perform the association test using Perturb-Seq, cis-eQTL, and GWAS summary data.
2. **closest**: Find the GWAS closest gene given a trait (i.e., the closest gene to a significant locus).
3. **signal**: Compute the summary data of GWAS Z score (i.e., mean, median, etc.) given a genomic region.

.. _Notes:
.. |Notes| replace:: **Notes**

Notes
=====
-   Mr. PEG uses [JAX](https://github.com/google/jax) with [Just In
    Time](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
    compilation to achieve high-speed computation. However, there are
    some [issues](https://github.com/google/jax/issues/5501) for JAX
    with Mac M1 chip. To solve this, users need to initiate conda using
    [miniforge](https://github.com/conda-forge/miniforge), and then
    install SuShiE using `pip` in the desired environment.

.. _Version:
.. |Version| replace:: **Version**

Version History
===============

.. list-table::
   :header-rows: 1

   * - Version
     - Description
   * - 0.1
     - Initial Release


.. _Support:
.. |Support| replace:: **Support**

Support
========

Please report any bugs or feature requests in the `Issue Tracker <https://github.com/gusevlab/mrpeg/issues>`_. If users have any
questions or comments, please contact Zeyun Lu (zeyun_lu@dfci.harvard.edu) and Sasha Gusev (alexander_gusev@dfci.harvard.edu).

.. _OtherSoftware:
.. |Other Software| replace:: **Other Software**

Other Software
==============

Feel free to use other software developed by `Gusev Lab <http://gusevlab.org///>`_:

[FUSION]() a suite of tools for performing transcriptome-wide and regulome-wide association studies (TWAS and RWAS).

---------------------

.. _pyscaffold-notes:

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
