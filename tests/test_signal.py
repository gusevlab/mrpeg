"""Tests for mrpeg.signal module (GWAS signal computation)."""

import pytest
import numpy as np
import pandas as pd

from mrpeg import signal


class TestParameterCheck:
    """Test parameter validation for signal command."""

    def test_parameter_check_missing_gwas(self, tmp_path):
        """Test error when GWAS file is missing."""
        class Args:
            gwas = str(tmp_path / "nonexistent.tsv.gz")
            ref = str(tmp_path / "ref.tsv.gz")
            keep = None
            window = 500
            threshold = 5e-8
            chr = None

        with pytest.raises(ValueError, match="GWAS file is not found"):
            signal._parameter_check(Args())

    def test_parameter_check_missing_ref(self, tmp_path, example_gwas):
        """Test error when reference file is missing."""
        class Args:
            gwas = example_gwas
            ref = str(tmp_path / "nonexistent_ref.tsv.gz")
            keep = None
            window = 500
            threshold = 5e-8
            chr = None

        with pytest.raises(ValueError, match="Reference file is not found"):
            signal._parameter_check(Args())

    def test_parameter_check_negative_window(self, tmp_path, example_gwas, ref_gene_info):
        """Test error when window is negative."""
        class Args:
            gwas = example_gwas
            ref = ref_gene_info
            keep = None
            window = -100
            threshold = 5e-8
            chr = None

        with pytest.raises(ValueError, match="Invalid window input"):
            signal._parameter_check(Args())

    def test_parameter_check_invalid_threshold(self, tmp_path, example_gwas, ref_gene_info):
        """Test error when threshold is out of range."""
        class Args:
            gwas = example_gwas
            ref = ref_gene_info
            keep = None
            window = 500
            threshold = 0
            chr = None

        with pytest.raises(ValueError, match="Invalid p value threshold"):
            signal._parameter_check(Args())

    def test_parameter_check_invalid_chr(self, example_gwas, ref_gene_info):
        """Test error when chromosome number is out of range."""
        class Args:
            gwas = example_gwas
            ref = ref_gene_info
            keep = None
            window = 500
            threshold = 5e-8
            chr = [25]  # Invalid: >22

        with pytest.raises(ValueError, match="Invalid chromosome input"):
            signal._parameter_check(Args())

    def test_parameter_check_chr_zero(self, example_gwas, ref_gene_info):
        """Test error when chromosome is 0."""
        class Args:
            gwas = example_gwas
            ref = ref_gene_info
            keep = None
            window = 500
            threshold = 5e-8
            chr = [0]  # Invalid: <1

        with pytest.raises(ValueError, match="Invalid chromosome input"):
            signal._parameter_check(Args())

    def test_parameter_check_valid_inputs(self, example_gwas, ref_gene_info):
        """Test that valid inputs pass parameter check."""
        class Args:
            gwas = example_gwas
            ref = ref_gene_info
            keep = None
            window = 500
            threshold = 5e-8
            chr = [1]

        result = signal._parameter_check(Args())
        assert result is None


class TestProcessGWAS:
    """Test GWAS data processing for signal computation."""

    def test_process_gwas_basic(self, tmp_path):
        """Test basic GWAS data loading and filtering."""
        gwas_data = pd.DataFrame({
            'CHR': [1, 1, 1, 2],
            'SNP': ['rs1', 'rs2', 'rs3', 'rs4'],
            'BP': [1000000, 2000000, 3000000, 4000000],
            'BETA': [10.0, 10.0, 10.0, 10.0],
            'SE': [0.05, 0.05, 0.05, 0.05],
        })
        gwas_file = tmp_path / "gwas_process.tsv.gz"
        gwas_data.to_csv(gwas_file, sep='\t', index=False, compression='gzip')

        df = signal._process_gwas(
            str(gwas_file),
            ['CHR', 'SNP', 'BP', 'BETA', 'SE'],
            n_chr=None,
            threshold=0.5  # Permissive threshold
        )

        assert len(df) > 0
        assert 'Z' in df.columns
        assert all(df['CHR'].between(1, 22))

    def test_process_gwas_chr_filtering(self, tmp_path):
        """Test chromosome-specific filtering."""
        gwas_data = pd.DataFrame({
            'CHR': [1, 1, 2, 2],
            'SNP': ['rs1', 'rs2', 'rs3', 'rs4'],
            'BP': [1000000, 2000000, 3000000, 4000000],
            'BETA': [10.0, 10.0, 10.0, 10.0],
            'SE': [0.05, 0.05, 0.05, 0.05],
        })
        gwas_file = tmp_path / "gwas_chr.tsv.gz"
        gwas_data.to_csv(gwas_file, sep='\t', index=False, compression='gzip')

        df = signal._process_gwas(
            str(gwas_file),
            ['CHR', 'SNP', 'BP', 'BETA', 'SE'],
            n_chr=[1],
            threshold=0.5
        )

        assert all(df['CHR'] == 1)

    def test_process_gwas_missing_chromosome(self, tmp_path):
        """Test error when filtering for chromosome with no data."""
        gwas_data = pd.DataFrame({
            'CHR': [1, 1],
            'SNP': ['rs1', 'rs2'],
            'BP': [1000000, 2000000],
            'BETA': [10.0, 10.0],
            'SE': [0.05, 0.05],
        })
        gwas_file = tmp_path / "gwas_misschr.tsv.gz"
        gwas_data.to_csv(gwas_file, sep='\t', index=False, compression='gzip')

        with pytest.raises(ValueError, match="GWAS data doesn't contain any SNPs on chromosome"):
            signal._process_gwas(
                str(gwas_file),
                ['CHR', 'SNP', 'BP', 'BETA', 'SE'],
                n_chr=[5],  # No data for chr 5
                threshold=0.5
            )

    def test_process_gwas_removes_zero_se(self, tmp_path):
        """Test that SNPs with SE <= 0 are removed."""
        gwas_data = pd.DataFrame({
            'CHR': [1, 1, 1],
            'SNP': ['rs1', 'rs2', 'rs3'],
            'BP': [1000000, 2000000, 3000000],
            'BETA': [10.0, 10.0, 10.0],
            'SE': [0.05, 0.0, 0.05],  # rs2 has SE=0
        })
        gwas_file = tmp_path / "gwas_zero_se.tsv.gz"
        gwas_data.to_csv(gwas_file, sep='\t', index=False, compression='gzip')

        df = signal._process_gwas(
            str(gwas_file),
            ['CHR', 'SNP', 'BP', 'BETA', 'SE'],
            n_chr=None,
            threshold=0.5
        )

        assert 'rs2' not in df['SNP'].values

    def test_process_gwas_wrong_columns(self, tmp_path):
        """Test error when columns don't match."""
        gwas_data = pd.DataFrame({
            'chromosome': [1],
            'variant': ['rs1'],
            'position': [1000000],
            'effect': [10.0],
            'stdev': [0.05],
        })
        gwas_file = tmp_path / "gwas_badcols.tsv.gz"
        gwas_data.to_csv(gwas_file, sep='\t', index=False, compression='gzip')

        with pytest.raises(ValueError, match="Specified GWAS columns are not in the GWAS data"):
            signal._process_gwas(
                str(gwas_file),
                ['CHR', 'SNP', 'BP', 'BETA', 'SE'],
                n_chr=None,
                threshold=0.5
            )


class TestProcessRef:
    """Test reference annotation processing."""

    def test_process_ref_basic(self, tmp_path):
        """Test basic reference data loading."""
        ref_data = pd.DataFrame({
            'CHR': [1, 1, 2],
            'TSS': [1000000, 2000000, 3000000],
            'TES': [1100000, 2100000, 3100000],
            'GENE': ['GENE1', 'GENE2', 'GENE3'],
        })
        ref_file = tmp_path / "ref_basic.tsv.gz"
        ref_data.to_csv(ref_file, sep='\t', index=False, compression='gzip')

        df = signal._process_ref(
            str(ref_file),
            ['CHR', 'TSS', 'TES', 'GENE'],
            n_chr=None,
            keep=None,
            window=100  # 100kb
        )

        assert len(df) == 3
        assert 'ANNO' in df.columns
        assert 'P0_FLANK' in df.columns
        assert 'P1_FLANK' in df.columns

    def test_process_ref_flanking_window(self, tmp_path):
        """Test that flanking regions are correctly computed."""
        ref_data = pd.DataFrame({
            'CHR': [1],
            'TSS': [1000000],
            'TES': [1100000],
            'GENE': ['GENE1'],
        })
        ref_file = tmp_path / "ref_flank.tsv.gz"
        ref_data.to_csv(ref_file, sep='\t', index=False, compression='gzip')

        window_kb = 100  # 100kb flanking = 50kb on each side
        df = signal._process_ref(
            str(ref_file),
            ['CHR', 'TSS', 'TES', 'GENE'],
            n_chr=None,
            keep=None,
            window=window_kb
        )

        half_window = int(window_kb * 1000 / 2)
        assert df['P0_FLANK'].values[0] == max(1000000 - half_window, 0)
        assert df['P1_FLANK'].values[0] == 1100000 + half_window

    def test_process_ref_chr_filtering(self, tmp_path):
        """Test chromosome-specific reference filtering."""
        ref_data = pd.DataFrame({
            'CHR': [1, 1, 2, 2],
            'TSS': [1000000, 2000000, 3000000, 4000000],
            'TES': [1100000, 2100000, 3100000, 4100000],
            'GENE': ['GENE1', 'GENE2', 'GENE3', 'GENE4'],
        })
        ref_file = tmp_path / "ref_chr.tsv.gz"
        ref_data.to_csv(ref_file, sep='\t', index=False, compression='gzip')

        df = signal._process_ref(
            str(ref_file),
            ['CHR', 'TSS', 'TES', 'GENE'],
            n_chr=[2],
            keep=None,
            window=0
        )

        assert all(df['CHR'] == 2)
        assert len(df) == 2


class TestAnnotTree:
    """Test annotation tree construction."""

    def test_annot_tree_basic(self):
        """Test that SNPs are correctly assigned to overlapping annotations."""
        df_gwas = pd.DataFrame({
            'CHR': [1, 1],
            'SNP': ['rs1', 'rs2'],
            'BP': [1050000, 3000000],  # rs1 falls in GENE1, rs2 in GENE2
            'Z': [3.5, -2.1],
        })

        df_ref = pd.DataFrame({
            'CHR': [1, 1],
            'P0': [1000000, 2900000],
            'P1': [1100000, 3100000],
            'P0_FLANK': [900000, 2800000],
            'P1_FLANK': [1200000, 3200000],
            'ANNO': ['GENE1', 'GENE2'],
        })

        _, res_filter = signal._annot_tree(df_gwas, df_ref, sep=":", snps_anno=False)

        assert len(res_filter) >= 2
        assert 'GENE1' in res_filter['ANNO'].values
        assert 'GENE2' in res_filter['ANNO'].values

    def test_annot_tree_snp_not_in_annotation(self):
        """Test that SNPs outside all annotations get empty annotation."""
        df_gwas = pd.DataFrame({
            'CHR': [1],
            'SNP': ['rs1'],
            'BP': [9000000],  # Far from any annotation
            'Z': [3.5],
        })

        df_ref = pd.DataFrame({
            'CHR': [1],
            'P0': [1000000],
            'P1': [1100000],
            'P0_FLANK': [900000],
            'P1_FLANK': [1200000],
            'ANNO': ['GENE1'],
        })

        res_full, res_filter = signal._annot_tree(df_gwas, df_ref, sep=":", snps_anno=True)

        # SNP is outside all annotations, filter should only contain non-empty annots
        if len(res_filter) > 0:
            assert all(res_filter['ANNO'] != '')


class TestSummarize:
    """Test GWAS signal summarization."""

    def test_summarize_basic(self):
        """Test basic signal summarization."""
        anno_snps = pd.DataFrame({
            'ANNO': ['GENE1', 'GENE1', 'GENE1', 'GENE2', 'GENE2'],
            'SNP': ['rs1', 'rs2', 'rs3', 'rs4', 'rs5'],
            'Z': [3.5, 2.1, -1.8, 4.2, -0.5],
            'CHR': [1, 1, 1, 1, 1],
        })

        df_ref = pd.DataFrame({
            'CHR': [1, 1],
            'P0': [1000000, 2000000],
            'P1': [1100000, 2100000],
            'P0_FLANK': [900000, 1900000],
            'P1_FLANK': [1200000, 2200000],
            'ANNO': ['GENE1', 'GENE2'],
        })

        result = signal._summarize(anno_snps, df_ref)

        assert len(result) == 2
        assert 'mean_chisq' in result.columns
        assert 'sd_chisq' in result.columns
        assert 'count' in result.columns
        assert result[result['ANNO'] == 'GENE1']['count'].values[0] == 3
        assert result[result['ANNO'] == 'GENE2']['count'].values[0] == 2

    def test_summarize_chisq_values(self):
        """Test that chi-square statistics are computed correctly as Z^2."""
        anno_snps = pd.DataFrame({
            'ANNO': ['GENE1', 'GENE1'],
            'SNP': ['rs1', 'rs2'],
            'Z': [2.0, 3.0],
            'CHR': [1, 1],
        })

        df_ref = pd.DataFrame({
            'CHR': [1],
            'P0': [1000000],
            'P1': [1100000],
            'P0_FLANK': [900000],
            'P1_FLANK': [1200000],
            'ANNO': ['GENE1'],
        })

        result = signal._summarize(anno_snps, df_ref)

        # Z^2 values: 4.0 and 9.0
        expected_mean = (4.0 + 9.0) / 2.0
        np.testing.assert_almost_equal(
            result[result['ANNO'] == 'GENE1']['mean_chisq'].values[0],
            expected_mean
        )

    def test_summarize_deduplicates_snps(self):
        """Test that duplicate ANNO-SNP pairs are properly deduplicated."""
        anno_snps = pd.DataFrame({
            'ANNO': ['GENE1', 'GENE1'],  # Same SNP appearing twice for GENE1
            'SNP': ['rs1', 'rs1'],
            'Z': [2.0, 2.0],
            'CHR': [1, 1],
        })

        df_ref = pd.DataFrame({
            'CHR': [1],
            'P0': [1000000],
            'P1': [1100000],
            'P0_FLANK': [900000],
            'P1_FLANK': [1200000],
            'ANNO': ['GENE1'],
        })

        result = signal._summarize(anno_snps, df_ref)

        # After dedup, only 1 SNP should be counted
        assert result[result['ANNO'] == 'GENE1']['count'].values[0] == 1
