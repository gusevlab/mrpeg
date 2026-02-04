"""Shared pytest fixtures for mrpeg tests."""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np


@pytest.fixture
def data_dir():
    """Return path to the data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def example_gwas(data_dir):
    """Return path to example GWAS file."""
    return str(data_dir / "example_gwas.tsv.gz")


@pytest.fixture
def example_eqtl(data_dir):
    """Return path to example eQTL file."""
    return str(data_dir / "example_eqtl.tsv.gz")


@pytest.fixture
def example_perturb(data_dir):
    """Return path to example perturbation file."""
    return str(data_dir / "example_perturbation.tsv.gz")


@pytest.fixture
def ref_geno(data_dir):
    """Return path pattern to reference genotype files."""
    return str(data_dir / "plink" / "geno_chr*")


@pytest.fixture
def ref_gene_info(data_dir):
    """Return path to reference gene info file."""
    return str(data_dir / "ref_gene_info.tsv.gz")


@pytest.fixture
def small_gwas_data():
    """Generate small synthetic GWAS data for testing."""
    np.random.seed(42)
    n_snps = 10
    return pd.DataFrame({
        'CHR': [1] * n_snps,
        'SNP': [f'rs{i}' for i in range(n_snps)],
        'BP': np.arange(1000000, 1000000 + n_snps * 10000, 10000),
        'A1': ['A'] * n_snps,
        'A0': ['G'] * n_snps,
        'BETA': np.random.randn(n_snps) * 0.1,
        'SE': np.ones(n_snps) * 0.05,
    })


@pytest.fixture
def small_eqtl_data():
    """Generate small synthetic eQTL data for testing."""
    np.random.seed(42)
    n_snps = 10
    return pd.DataFrame({
        'CHR': [1] * n_snps,
        'SNP': [f'rs{i}' for i in range(n_snps)],
        'A1': ['A'] * n_snps,
        'A0': ['G'] * n_snps,
        'Z': np.random.randn(n_snps),
        'GENE': ['GENE1'] * n_snps,
    })


@pytest.fixture
def small_perturb_data():
    """Generate small synthetic perturbation matrix for testing."""
    np.random.seed(42)
    n_genes = 3
    genes = [f'GENE{i}' for i in range(1, n_genes + 1)]
    data = pd.DataFrame(
        np.random.randn(n_genes, n_genes),
        index=genes,
        columns=genes
    )
    return data


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Create temporary output directory for tests."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return output_dir
