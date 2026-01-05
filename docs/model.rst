.. _Model:

=================
Model Description
=================

Model Overview
====================

Mr. PEG estimates the effect size :math:`\alpha \in \mathbb{R}` of a focal gene’s expression on a complex trait. The expression level of the mediating gene is modeled as a linear combination of the expression levels of perturbed genes, which in turn are modeled as a linear combination of genotyped variants.

This hierarchical framework captures the relationships between eQTLs, perturbed genes, and mediating genes, ultimately linking genetic variants to complex traits.

The mathematical representation is given by:

.. math::

   \mathbf{y} = \mathbf{X}\mathbf{\Delta}\boldsymbol{\gamma}\alpha + \boldsymbol{\epsilon}
   = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}

where

- :math:`\mathbf{y} \in \mathbb{R}^{n \times 1}` is the normalized complex trait measured across :math:`n` individuals, with mean 0 and standard deviation 1.
- :math:`\mathbf{X} \in \mathbb{R}^{n \times k}` is the normalized genotype matrix.
- :math:`k` is the number of eQTLs across :math:`t` perturbed genes.
- :math:`\mathbf{\Delta} \in \mathbb{R}^{k \times t}` represents the eQTL effect sizes on the :math:`t` perturbed genes.
- :math:`\boldsymbol{\gamma} \in \mathbb{R}^{t \times 1}` denotes the gene-to-gene effect sizes on the mediating gene.
- :math:`\alpha \in \mathbb{R}` is the mediating effect size.
- :math:`\boldsymbol{\beta} = \mathbf{\Delta}\boldsymbol{\gamma}\alpha \in \mathbb{R}^{k \times 1}` denotes the SNP effects on the complex trait.
- :math:`\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma_\epsilon^2 \mathbf{I}_{n \times n})` is the environmental noise.

Statistical Inference
====================

Our goal is to test the mediating effect :math:`\alpha` for the focal gene. We obtain the marginal effect size estimate :math:`\hat{\boldsymbol{\beta}}^{*}` from GWASs, the marginal effect size estimate :math:`\hat{\boldsymbol{\Delta}}^{*}` from cis-eQTL studies, and the perturbation effect size estimate :math:`\hat{\boldsymbol{\gamma}}` from perturbational screening experiments.

We assume :math:`\hat{\boldsymbol{\beta}}^{*}` has the sampling distribution

.. math::

   \hat{\boldsymbol{\beta}}^{*} \sim \mathcal{N}(\hat{\mathbf{V}}\boldsymbol{\beta}, \sigma^2 \hat{\mathbf{D}}\hat{\mathbf{V}}\hat{\mathbf{D}})

and derive the unbiased and maximum likelihood estimator (MLE) for :math:`\alpha` as

.. math::

   \hat{\alpha}
   =
   \frac{
     (\hat{\boldsymbol{\Delta}}^{*}\hat{\boldsymbol{\gamma}})^{\mathsf{T}}
     (\hat{\mathbf{D}}\hat{\mathbf{V}}\hat{\mathbf{D}})^{-1}
     \hat{\boldsymbol{\beta}}^{*}
   }{
     (\hat{\boldsymbol{\Delta}}^{*}\hat{\boldsymbol{\gamma}})^{\mathsf{T}}
     (\hat{\mathbf{D}}\hat{\mathbf{V}}\hat{\mathbf{D}})^{-1}
     (\hat{\boldsymbol{\Delta}}^{*}\hat{\boldsymbol{\gamma}})
   }.

where

- :math:`\hat{\mathbf{V}} \in \mathbb{R}^{k \times k}` is the estimated SNP correlation (LD) matrix.
- :math:`\sigma^2 \in \mathbb{R}_{+}` is a heterogeneity parameter accounting for noise due to potential horizontal pleiotropy.
- :math:`\hat{\mathbf{D}} \in \mathbb{R}^{k \times k}` is a diagonal matrix containing the standard errors of :math:`\hat{\boldsymbol{\beta}}^{*}` from GWASs.

To estimate the standard error of :math:`\hat{\alpha}`, we permute the perturbation effects :math:`\hat{\boldsymbol{\gamma}}` to construct the null distribution of :math:`\hat{\alpha}`, providing a conservative estimate. This null assumes no association between complex traits and mediating genes, meaning any observed relationship is due solely to random perturbation effects rather than regulation by the perturbed genes and their cis-eQTLs.
