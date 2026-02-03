"""Tests for mrpeg.peg module (core inference)."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import gzip

from mrpeg import peg


class TestParameterCheck:
    """Test parameter validation."""

    def test_parameter_check_missing_gwas(self, tmp_path):
        """Test error when GWAS file is missing."""
        class Args:
            gwas = str(tmp_path / "nonexistent.tsv.gz")
            eqtl = "dummy.tsv.gz"
            perturb = "dummy.tsv.gz"
            ref_geno = "dummy"

        with pytest.raises(ValueError, match="No GWAS file located"):
            peg._parameter_check(Args())

    def test_parameter_check_missing_eqtl(self, tmp_path, example_gwas):
        """Test error when eQTL file is missing."""
        class Args:
            gwas = example_gwas
            eqtl = str(tmp_path / "nonexistent.tsv.gz")
            perturb = "dummy.tsv.gz"
            ref_geno = "dummy"

        with pytest.raises(ValueError, match="No eQTL file located"):
            peg._parameter_check(Args())

    def test_parameter_check_missing_perturb(self, tmp_path, example_gwas, example_eqtl):
        """Test error when perturbation file is missing."""
        class Args:
            gwas = example_gwas
            eqtl = example_eqtl
            perturb = str(tmp_path / "nonexistent.tsv.gz")
            ref_geno = "dummy"

        with pytest.raises(ValueError, match="No Perturb-Seq file located"):
            peg._parameter_check(Args())


class TestPrepareGWAS:
    """Test GWAS data preparation."""

    def test_prepare_gwas_basic(self, tmp_path):
        """Test basic GWAS data loading."""
        # Use only non-ambiguous allele pairs (avoid AT, TA, CG, GC)
        gwas_data = pd.DataFrame({
            'CHR': [1, 1, 1],
            'SNP': ['rs1', 'rs2', 'rs3'],
            'A1': ['A', 'C', 'G'],
            'A0': ['G', 'T', 'T'],
            'BETA': [0.1, 0.2, -0.1],
            'SE': [0.05, 0.05, 0.05],
        })
        gwas_file = tmp_path / "test_gwas.tsv.gz"
        gwas_data.to_csv(gwas_file, sep='\t', index=False, compression='gzip')

        df = peg._prepare_gwas(
            str(gwas_file),
            ['CHR', 'SNP', 'A1', 'A0', 'BETA', 'SE'],
            keep_ambiguous=False
        )

        assert len(df) == 3
        assert 'CHR' in df.columns
        assert 'SNP' in df.columns
        assert all(df['CHR'] == 1)

    def test_prepare_gwas_removes_ambiguous(self, tmp_path):
        """Test that ambiguous SNPs are removed when keep_ambiguous=False."""
        # Ambiguous pairs: AT, TA, CG, GC. Non-ambiguous: AG, GA, CT, TC, AC, CA, GT, TG
        gwas_data = pd.DataFrame({
            'CHR': [1, 1, 1, 1],
            'SNP': ['rs1', 'rs2_ambig_AT', 'rs3_ambig_CG', 'rs4'],
            'A1': ['A', 'A', 'C', 'A'],
            'A0': ['G', 'T', 'G', 'C'],  # rs1=AG, rs2=AT(ambig), rs3=CG(ambig), rs4=AC
            'BETA': [0.1, 0.2, -0.1, 0.15],
            'SE': [0.05, 0.05, 0.05, 0.05],
        })
        gwas_file = tmp_path / "test_gwas_ambig.tsv.gz"
        gwas_data.to_csv(gwas_file, sep='\t', index=False, compression='gzip')

        df = peg._prepare_gwas(
            str(gwas_file),
            ['CHR', 'SNP', 'A1', 'A0', 'BETA', 'SE'],
            keep_ambiguous=False
        )

        # Should remove the ambiguous SNPs (rs2 and rs3), keep rs1 and rs4
        assert len(df) == 2
        assert 'rs2_ambig_AT' not in df['SNP'].values
        assert 'rs3_ambig_CG' not in df['SNP'].values
        assert 'rs1' in df['SNP'].values
        assert 'rs4' in df['SNP'].values

    def test_prepare_gwas_keeps_ambiguous(self, tmp_path):
        """Test that ambiguous SNPs are kept when keep_ambiguous=True."""
        gwas_data = pd.DataFrame({
            'CHR': [1, 1],
            'SNP': ['rs1', 'rs2_ambig'],
            'A1': ['A', 'A'],
            'A0': ['G', 'T'],  # rs2 is A/T
            'BETA': [0.1, 0.2],
            'SE': [0.05, 0.05],
        })
        gwas_file = tmp_path / "test_gwas_keep.tsv.gz"
        gwas_data.to_csv(gwas_file, sep='\t', index=False, compression='gzip')

        df = peg._prepare_gwas(
            str(gwas_file),
            ['CHR', 'SNP', 'A1', 'A0', 'BETA', 'SE'],
            keep_ambiguous=True
        )

        assert len(df) == 2
        assert 'rs2_ambig' in df['SNP'].values


class TestPrepareEQTL:
    """Test eQTL data preparation."""

    def test_prepare_eqtl_basic(self, tmp_path):
        """Test basic eQTL data loading."""
        eqtl_data = pd.DataFrame({
            'CHR': [1, 1, 1],
            'SNP': ['rs1', 'rs2', 'rs3'],
            'A1': ['A', 'C', 'G'],
            'A0': ['G', 'T', 'C'],
            'Z': [2.5, -1.8, 3.2],
            'GENE': ['GENE1', 'GENE1', 'GENE2'],
        })
        eqtl_file = tmp_path / "test_eqtl.tsv.gz"
        eqtl_data.to_csv(eqtl_file, sep='\t', index=False, compression='gzip')

        df = peg._prepare_eqtl(
            str(eqtl_file),
            ['CHR', 'SNP', 'A1', 'A0', 'Z', 'GENE']
        )

        assert len(df) == 3
        assert 'CHR' in df.columns
        assert 'Z_eqtl' in df.columns
        assert 'GENE' in df.columns


class TestPreparePerturb:
    """Test perturbation data preparation."""

    def test_prepare_perturb_basic(self, tmp_path):
        """Test basic perturbation matrix loading with enough genes."""
        # _prepare_perturb requires >= 10 perturbed genes per target gene
        np.random.seed(42)
        n_perturb = 15
        n_target = 3
        genes = [f'GENE{i}' for i in range(1, n_perturb + 1)]
        targets = [f'TARGET{j}' for j in range(1, n_target + 1)]

        perturb_data = pd.DataFrame(
            np.random.randn(n_perturb, n_target),
            index=genes,
            columns=targets
        )
        perturb_file = tmp_path / "test_perturb.tsv.gz"
        perturb_data.to_csv(perturb_file, sep='\t', compression='gzip')

        df = peg._prepare_perturb(str(perturb_file), top_signal=1.0)

        # With top_signal=1.0 (keep all), should have all perturbed genes
        # and at least 1 target gene (those with >=10 perturbed genes passing filter)
        assert df.shape[0] > 0
        assert df.shape[1] > 1  # GENE column + at least 1 target

    def test_prepare_perturb_top_signal_filtering(self, tmp_path):
        """Test that top_signal filters perturbation effects correctly."""
        np.random.seed(42)
        n_perturb = 20
        n_target = 3
        genes = [f'GENE{i}' for i in range(1, n_perturb + 1)]
        targets = [f'TARGET{j}' for j in range(1, n_target + 1)]

        perturb_data = pd.DataFrame(
            np.random.randn(n_perturb, n_target),
            index=genes,
            columns=targets
        )
        perturb_file = tmp_path / "test_perturb_filter.tsv.gz"
        perturb_data.to_csv(perturb_file, sep='\t', compression='gzip')

        # top_signal=1.0 keeps all, top_signal=0.5 keeps top 50%
        df_all = peg._prepare_perturb(str(perturb_file), top_signal=1.0)
        df_half = peg._prepare_perturb(str(perturb_file), top_signal=0.5)

        # With top_signal=1.0 (all) should have more or equal genes than 0.5
        assert df_all.shape[0] >= df_half.shape[0]

    def test_prepare_perturb_invalid_top_signal(self, tmp_path):
        """Test that invalid top_signal values are caught."""
        perturb_data = pd.DataFrame(
            np.random.randn(10, 3),
            index=[f'G{i}' for i in range(10)],
            columns=['T1', 'T2', 'T3']
        )
        perturb_file = tmp_path / "test_perturb_inv.tsv.gz"
        perturb_data.to_csv(perturb_file, sep='\t', compression='gzip')

        with pytest.raises(ValueError, match="top signals must be"):
            peg._prepare_perturb(str(perturb_file), top_signal=2.0)

        with pytest.raises(ValueError, match="top signals must be"):
            peg._prepare_perturb(str(perturb_file), top_signal=-0.1)


class TestAlleleCheck:
    """Test allele checking and harmonization."""

    def test_allele_check_correct_alleles(self):
        """Test allele checking with correctly matched alleles."""
        base_a1 = pd.Series(['A', 'C', 'G'])
        base_a0 = pd.Series(['G', 'T', 'C'])
        compare_a1 = pd.Series(['A', 'C', 'G'])
        compare_a0 = pd.Series(['G', 'T', 'C'])

        correct, flipped, wrong = peg._allele_check(base_a1, base_a0, compare_a1, compare_a0)

        # Returns index arrays: correct=indices where match, flipped=indices where flipped
        assert len(correct) == 3
        assert len(flipped) == 0
        assert len(wrong) == 0

    def test_allele_check_flipped_alleles(self):
        """Test allele checking with flipped alleles."""
        base_a1 = pd.Series(['A', 'C'])
        base_a0 = pd.Series(['G', 'T'])
        compare_a1 = pd.Series(['G', 'T'])  # Flipped
        compare_a0 = pd.Series(['A', 'C'])  # Flipped

        correct, flipped, wrong = peg._allele_check(base_a1, base_a0, compare_a1, compare_a0)

        assert len(correct) == 0
        assert len(flipped) == 2
        assert len(wrong) == 0

    def test_allele_check_wrong_alleles(self):
        """Test allele checking with mismatched alleles."""
        base_a1 = pd.Series(['A', 'C'])
        base_a0 = pd.Series(['G', 'T'])
        compare_a1 = pd.Series(['T', 'G'])  # Wrong alleles
        compare_a0 = pd.Series(['C', 'A'])  # Wrong alleles

        correct, flipped, wrong = peg._allele_check(base_a1, base_a0, compare_a1, compare_a0)

        assert len(correct) == 0
        assert len(flipped) == 0
        assert len(wrong) == 2


class TestInferPeg:
    """Test main inference function."""

    def test_infer_peg_basic(self):
        """Test basic inference with small synthetic data."""
        np.random.seed(42)
        k = 10  # number of SNPs
        t = 5   # number of downstream genes

        beta = np.random.randn(k) * 0.1
        se = np.ones(k) * 0.05
        eqtl = np.random.randn(k)
        perturb = np.random.randn(k, t)
        ld = np.eye(k)  # No LD

        result = peg.infer_peg(
            beta=beta,
            se=se,
            eqtl=eqtl,
            perturb=perturb,
            ld=ld,
            perm_number=10,  # Small number for testing
            seed=42
        )

        # Should return array with shape (t, 6)
        # Columns: gamma, gamma_se, gamma_p, gamma_perm_mean, gamma_perm_z, gamma_null_p
        assert result.shape == (t, 6)
        assert not np.any(np.isnan(result))

    def test_infer_peg_dimension_mismatch(self):
        """Test that dimension mismatches are handled."""
        k = 10
        t = 5

        beta = np.random.randn(k)
        se = np.ones(k)
        eqtl = np.random.randn(k + 1)  # Wrong size!
        perturb = np.random.randn(k, t)
        ld = np.eye(k)

        # Should raise an error due to dimension mismatch
        with pytest.raises((ValueError, AssertionError, IndexError)):
            peg.infer_peg(beta, se, eqtl, perturb, ld, perm_number=10)

    def test_infer_peg_invalid_seed(self):
        """Test that invalid seed values are caught."""
        k = 5
        t = 3

        beta = np.random.randn(k)
        se = np.ones(k)
        eqtl = np.random.randn(k)
        perturb = np.random.randn(k, t)
        ld = np.eye(k)

        # Seed must be positive
        with pytest.raises(ValueError, match="seed specified for randomization is invalid"):
            peg.infer_peg(beta, se, eqtl, perturb, ld, seed=-1)

        with pytest.raises(ValueError, match="seed specified for randomization is invalid"):
            peg.infer_peg(beta, se, eqtl, perturb, ld, seed=0)

    def test_infer_peg_reproducibility(self):
        """Test that results are reproducible with the same seed."""
        np.random.seed(42)
        k = 10
        t = 5

        beta = np.random.randn(k)
        se = np.ones(k) * 0.05
        eqtl = np.random.randn(k)
        perturb = np.random.randn(k, t)
        ld = np.eye(k)

        result1 = peg.infer_peg(beta, se, eqtl, perturb, ld, perm_number=10, seed=42)
        result2 = peg.infer_peg(beta, se, eqtl, perturb, ld, perm_number=10, seed=42)

        # Results should be identical
        np.testing.assert_array_almost_equal(result1, result2)
