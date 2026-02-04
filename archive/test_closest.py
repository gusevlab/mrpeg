"""Tests for mrpeg.closest module (gene annotation)."""

import pytest
import numpy as np
import pandas as pd

from mrpeg import closest


class TestParameterCheck:
    """Test parameter validation for closest command."""

    def test_parameter_check_missing_gwas(self, tmp_path):
        """Test error when GWAS file is missing."""
        class Args:
            gwas = str(tmp_path / "nonexistent.tsv.gz")
            ref = str(tmp_path / "ref.tsv.gz")
            keep = None
            window = 500
            threshold = 5e-8

        with pytest.raises(ValueError, match="GWAS file is not found"):
            closest._parameter_check(Args())

    def test_parameter_check_missing_ref(self, tmp_path, example_gwas):
        """Test error when reference file is missing."""
        class Args:
            gwas = example_gwas
            ref = str(tmp_path / "nonexistent_ref.tsv.gz")
            keep = None
            window = 500
            threshold = 5e-8

        with pytest.raises(ValueError, match="Reference file is not found"):
            closest._parameter_check(Args())

    def test_parameter_check_negative_window(self, tmp_path, example_gwas, ref_gene_info):
        """Test error when window is negative."""
        class Args:
            gwas = example_gwas
            ref = ref_gene_info
            keep = None
            window = -100
            threshold = 5e-8

        with pytest.raises(ValueError, match="Invalid window input"):
            closest._parameter_check(Args())

    def test_parameter_check_invalid_threshold_zero(self, tmp_path, example_gwas, ref_gene_info):
        """Test error when threshold is 0."""
        class Args:
            gwas = example_gwas
            ref = ref_gene_info
            keep = None
            window = 500
            threshold = 0

        with pytest.raises(ValueError, match="Invalid p value threshold"):
            closest._parameter_check(Args())

    def test_parameter_check_invalid_threshold_above_one(self, tmp_path, example_gwas, ref_gene_info):
        """Test error when threshold is greater than 1."""
        class Args:
            gwas = example_gwas
            ref = ref_gene_info
            keep = None
            window = 500
            threshold = 1.5

        with pytest.raises(ValueError, match="Invalid p value threshold"):
            closest._parameter_check(Args())

    def test_parameter_check_valid_inputs(self, example_gwas, ref_gene_info):
        """Test that valid inputs pass parameter check."""
        class Args:
            gwas = example_gwas
            ref = ref_gene_info
            keep = None
            window = 500
            threshold = 5e-8

        # Should not raise any exception
        result = closest._parameter_check(Args())
        assert result is None


class TestGetMaxGWAS:
    """Test GWAS significance filtering and region merging."""

    def test_get_max_gwas_basic(self, tmp_path):
        """Test basic GWAS significant SNP selection."""
        # Create GWAS data with clear significant signal
        gwas_data = pd.DataFrame({
            'CHR': [1, 1, 1, 2],
            'SNP': ['rs1', 'rs2', 'rs3', 'rs4'],
            'BP': [1000000, 1010000, 5000000, 1000000],
            'BETA': [10.0, 0.001, 10.0, 10.0],  # Large BETA = significant
            'SE': [0.05, 0.05, 0.05, 0.05],
        })
        gwas_file = tmp_path / "gwas_sig.tsv.gz"
        gwas_data.to_csv(gwas_file, sep='\t', index=False, compression='gzip')

        # Very permissive threshold to ensure all large-beta SNPs pass
        df = closest._get_max_gwas(
            str(gwas_file),
            ['CHR', 'SNP', 'BP', 'BETA', 'SE'],
            window=500,   # 500kb window
            threshold=0.5  # Very permissive threshold for testing
        )

        assert len(df) > 0
        assert 'CHR' in df.columns
        assert 'SNP' in df.columns

    def test_get_max_gwas_no_significant_snps(self, tmp_path):
        """Test error when no significant SNPs are found."""
        gwas_data = pd.DataFrame({
            'CHR': [1, 1, 1],
            'SNP': ['rs1', 'rs2', 'rs3'],
            'BP': [1000000, 2000000, 3000000],
            'BETA': [0.001, 0.001, 0.001],  # All tiny BETAs
            'SE': [1.0, 1.0, 1.0],  # Large SE -> Z ~= 0
        })
        gwas_file = tmp_path / "gwas_nonsig.tsv.gz"
        gwas_data.to_csv(gwas_file, sep='\t', index=False, compression='gzip')

        with pytest.raises(ValueError, match="GWAS data doesn't contain any significant hits"):
            closest._get_max_gwas(
                str(gwas_file),
                ['CHR', 'SNP', 'BP', 'BETA', 'SE'],
                window=500,
                threshold=5e-8  # Strict threshold
            )

    def test_get_max_gwas_missing_columns(self, tmp_path):
        """Test error when GWAS columns don't match."""
        gwas_data = pd.DataFrame({
            'chromosome': [1, 1],
            'variant': ['rs1', 'rs2'],
            'position': [1000000, 2000000],
            'effect': [0.1, 0.2],
            'stdev': [0.05, 0.05],
        })
        gwas_file = tmp_path / "gwas_wrongcols.tsv.gz"
        gwas_data.to_csv(gwas_file, sep='\t', index=False, compression='gzip')

        with pytest.raises(ValueError, match="Specified GWAS columns are not in the GWAS data"):
            closest._get_max_gwas(
                str(gwas_file),
                ['CHR', 'SNP', 'BP', 'BETA', 'SE'],
                window=500,
                threshold=0.5
            )

    def test_get_max_gwas_filters_non_autosomal(self, tmp_path):
        """Test that non-autosomal chromosomes (e.g., X=23) are filtered out."""
        gwas_data = pd.DataFrame({
            'CHR': [1, 23, 1],
            'SNP': ['rs1', 'rs_chrX', 'rs3'],
            'BP': [1000000, 5000000, 3000000],
            'BETA': [10.0, 10.0, 10.0],
            'SE': [0.05, 0.05, 0.05],
        })
        gwas_file = tmp_path / "gwas_chrX.tsv.gz"
        gwas_data.to_csv(gwas_file, sep='\t', index=False, compression='gzip')

        df = closest._get_max_gwas(
            str(gwas_file),
            ['CHR', 'SNP', 'BP', 'BETA', 'SE'],
            window=500,
            threshold=0.5
        )

        # Chr 23 should be excluded
        if len(df) > 0:
            assert all(df['CHR'].between(1, 22))


class TestFindClosest:
    """Test closest gene finding logic."""

    def test_find_closest_snp_inside_gene(self):
        """Test that SNP inside a gene body is assigned to that gene."""
        sig_gwas = pd.DataFrame({
            'CHR': [1],
            'SNP': ['rs1'],
            'BP': [1005000],  # Between TSS=1000000 and TES=1010000
            'BETA': [0.1],
            'SE': [0.05],
        })

        pot_genes = pd.DataFrame({
            'CHR': [1, 1],
            'TSS': [1000000, 2000000],
            'TES': [1010000, 2010000],
            'GENE': ['GENE_A', 'GENE_B'],
        })

        result = closest._find_closest(sig_gwas, pot_genes)

        assert len(result) == 1
        assert result['GENE'].values[0] == 'GENE_A'

    def test_find_closest_snp_outside_genes(self):
        """Test closest gene when SNP is between genes."""
        sig_gwas = pd.DataFrame({
            'CHR': [1],
            'SNP': ['rs1'],
            'BP': [1500000],  # Between GENE_A (TES=1100000) and GENE_B (TSS=2000000)
            'BETA': [0.1],
            'SE': [0.05],
        })

        pot_genes = pd.DataFrame({
            'CHR': [1, 1],
            'TSS': [1000000, 2000000],
            'TES': [1100000, 2100000],
            'GENE': ['GENE_A', 'GENE_B'],
        })

        result = closest._find_closest(sig_gwas, pot_genes)

        assert len(result) == 1
        # SNP at 1500000: distance to GENE_A TES (1100000) = 400000
        # distance to GENE_B TSS (2000000) = 500000
        # GENE_A should be closer
        assert result['GENE'].values[0] == 'GENE_A'


class TestFindNearby:
    """Test nearby gene finding logic."""

    def test_find_nearby_basic(self):
        """Test that all genes within window are found."""
        sig_gwas = pd.DataFrame({
            'CHR': [1],
            'SNP': ['rs1'],
            'BP': [1500000],
            'BETA': [0.1],
            'SE': [0.05],
        })

        pot_genes = pd.DataFrame({
            'CHR': [1, 1, 1],
            'TSS': [1000000, 1400000, 5000000],
            'TES': [1100000, 1600000, 5100000],
            'GENE': ['GENE_A', 'GENE_B', 'GENE_C'],
        })

        # Window of 2000kb centered on SNP: [1500000 - 1000000, 1500000 + 1000000] = [500000, 2500000]
        result = closest._find_nearby(sig_gwas, pot_genes, window=2000)

        # GENE_A and GENE_B should be within window, GENE_C should not
        assert 'GENE_A' in result['GENE'].values
        assert 'GENE_B' in result['GENE'].values
        assert 'GENE_C' not in result['GENE'].values
