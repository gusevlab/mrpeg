#!/usr/bin/env python

from __future__ import division

import argparse
import logging
import os
import sys
import warnings
from importlib import metadata

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from jax import config

from . import closest, log, peg, signal

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

__all__ = [
    "run_peg",
    "run_closest",
    "run_signal",
    "_get_command_string",
]


def _get_command_string(args):
    base = f"mrpeg {args[0]}{os.linesep}"
    rest = args[1:]
    rest_strs = []
    needs_tab = True
    for cmd in rest:
        if "-" == cmd[0]:
            if cmd in ["--quiet", "-q", "--verbose", "-v"]:
                rest_strs.append(f"\t{cmd}{os.linesep}")
                needs_tab = True
            else:
                rest_strs.append(f"\t{cmd}")
                needs_tab = False
        else:
            if needs_tab:
                rest_strs.append(f"\t{cmd}{os.linesep}")
                needs_tab = True
            else:
                rest_strs.append(f" {cmd}{os.linesep}")
                needs_tab = True

    return base + "".join(rest_strs) + os.linesep


def run_peg(args):
    """The umbrella function to run Mr PEG.

    Args:
        args: The command line parameter input.

    """

    try:
        if args.jax_precision == 64:
            config.update("jax_enable_x64", True)

        config.update("jax_platform_name", args.platform)

        peg._parameter_check(args)

        clean_data = peg._process_raw(
            args.gwas,
            args.eqtl,
            args.perturb,
            args.gwas_cols,
            args.eqtl_cols,
            args.ref_geno,
        )

        infer_result = peg.infer_peg(
            clean_data.beta,
            clean_data.inv_se,
            clean_data.eqtl,
            clean_data.perturb,
            clean_data.inv_ld,
            args.no_permute,
            args.perm_number,
            args.seed,
        )

        df_infer = pd.DataFrame(
            infer_result,
            columns=[
                "mr_gamma",
                "mr_z",
                "mr_p",
                "mr_z_perm",
                "mr_p_perm",
            ],
        )

        df_result = pd.DataFrame(
            {
                "trait": args.trait,
                "gene_name": clean_data.gene_names,
                "n_perturb": clean_data.beta.shape[0],
                "n_perturb_sig": np.sum(np.abs(clean_data.perturb) > 1.96, axis=0),
            }
        )
        df_final = pd.concat([df_result, df_infer], axis=1)
        log.logger.info("Saving results.")
        suffix = ".gz" if args.compress else ""
        df_final.to_csv(f"{args.output}.mrpeg.tsv{suffix}", sep="\t", index=False)

    except Exception as err:
        import traceback

        print(
            "".join(
                traceback.format_exception(
                    etype=type(err), value=err, tb=err.__traceback__
                )
            )
        )
        log.logger.error(err)

    finally:
        log.logger.info(
            "Finished Mr PEG running. Thanks for using our software."
            + " For bug reporting, suggestions, and comments, please go to https://github.com/gusevlab/mrpeg.",
        )
    return 0


def run_closest(args):
    """The umbrella function to find closest GWAS genes.

    Args:
        args: The command line parameter input.

    """

    try:
        closest._parameter_check(args)

        sig_gwas = closest._get_max_gwas(
            args.gwas, args.gwas_cols, args.window, args.threshold
        )
        pot_genes = closest._process_potential(
            sig_gwas, args.ref, args.ref_cols, args.keep
        )
        closest_genes = closest._find_closest(sig_gwas, pot_genes)
        closest_genes["trait"] = args.trait
        suffix = ".gz" if args.compress else ""
        closest_genes.to_csv(
            f"{args.output}.closest.tsv{suffix}", sep="\t", index=False
        )

    except Exception as err:
        import traceback

        print(
            "".join(
                traceback.format_exception(
                    etype=type(err), value=err, tb=err.__traceback__
                )
            )
        )
        log.logger.error(err)

    finally:
        log.logger.info(
            "Finished Mr PEG finding closest gene. Thanks for using our software."
            + " For bug reporting, suggestions, and comments, please go to https://github.com/gusevlab/mrpeg.",
        )
    return 0


def run_signal(args):
    """The umbrella function to calcucate GWAS signals.

    Args:
        args: The command line parameter input.

    """

    try:
        signal._parameter_check(args)

        df_gwas = signal._process_gwas(
            args.gwas, args.gwas_cols, args.chr, args.threshold
        )

        df_ref = signal._process_ref(
            args.ref, args.ref_cols, args.chr, args.keep, args.window
        )

        anno_full, anno_filter = signal._annot_tree(df_gwas, df_ref, args.split, args.snps_anno)

        signal_summary = signal._summarize(anno_filter, df_ref)

        anno_filter["trait"] = args.trait
        signal_summary["trait"] = args.trait
        suffix = ".gz" if args.compress else ""
        signal_summary.to_csv(
            f"{args.output}.signal.tsv{suffix}", sep="\t", index=False
        )

        if args.snps_anno:
            anno_full["trait"] = args.trait
            anno_full.to_csv(f"{args.output}.full.anno.tsv.gz", sep="\t", index=False)
            anno_filter.to_csv(
                f"{args.output}.filter.anno.tsv.gz", sep="\t", index=False
            )

    except Exception as err:
        import traceback

        print(
            "".join(
                traceback.format_exception(
                    etype=type(err), value=err, tb=err.__traceback__
                )
            )
        )
        log.logger.error(err)

    finally:
        log.logger.info(
            "Finished Mr PEG GWAS signals for annotations. Thanks for using our software."
            + " For bug reporting, suggestions, and comments, please go to https://github.com/gusevlab/mrpeg.",
        )
    return 0


def build_peg_parser(subp):
    # add imputation parser
    peg = subp.add_parser(
        "peg",
        description=("Run Mr PEG framework",),
    )

    peg.add_argument(
        "--gwas",
        type=str,
        required=True,
        help=("GWAS data ",),
    )

    # main arguments
    peg.add_argument(
        "--eqtl",
        type=str,
        required=True,
        help=(
            "Genotype data in plink 1 format. The plink triplet (bed, bim, and fam) should be",
            " in the same folder with the same prefix.",
            " Use 'space' to separate ancestries if more than two.",
            " Keep the same ancestry order as phenotype's.",
        ),
    )

    peg.add_argument(
        "--perturb",
        type=str,
        required=True,
        help=(
            "Genotype data in vcf format. Use 'space' to separate ancestries if more than two.",
            " Keep the same ancestry order as phenotype's.",
        ),
    )

    peg.add_argument(
        "--ref_geno",
        default=None,
        type=str,
        help=("'*' to split",),
    )

    peg.add_argument(
        "--gwas_cols",
        nargs=6,
        default=["CHR", "SNP", "A1", "A0", "BETA", "SE"],
        type=str,
        help=(
            "Single file that contains subject ID across all ancestries that are used for fine-mapping."
        ),
    )

    peg.add_argument(
        "--eqtl_cols",
        nargs=6,
        default=["CHR", "SNP", "A1", "A0", "Z", "GENE"],
        type=str,
        help=(
            "Single file that contains subject ID across all ancestries that are used for fine-mapping."
        ),
    )

    peg.add_argument(
        "--perm_number",
        default=500,
        type=int,
        help=(
            "Integer number of shared effects pre-specified.",
            " Default is 5. Larger number may cause slow inference.",
        ),
    )

    peg.add_argument(
        "--no_permute",
        default=False,
        action="store_true",
        help=(
            "Indicator to scale the genotype and phenotype data by standard deviation.",
            " Default is False (to scale)."
            " Specify --no_scale will store 'True' value, and may cause different inference.",
        ),
    )

    peg.add_argument(
        "--seed",
        default=12345,
        type=int,
        help=(
            "The seed for randomization. It can be used to cut data sets in cross validation. ",
            " It can also be used to randomly select SNPs in the credible sets to calculate the purity."
            " Default is 12345. It has to be positive integer number.",
        ),
    )

    peg.add_argument(
        "--trait",
        default="Trait",
        help=(
            "Trait, tissue, gene name of the phenotype for better indexing in post-hoc analysis. Default is 'Trait'.",
        ),
    )

    # misc options
    peg.add_argument(
        "--quiet",
        default=False,
        action="store_true",
        help="Indicator to not print message to console. Default is False. Specify --numpy will store 'True' value.",
    )

    peg.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help=(
            "Indicator to include debug information in the log. Default is False.",
            " Specify --numpy will store 'True' value.",
        ),
    )

    peg.add_argument(
        "-c",
        "--compress",
        default=False,
        action="store_true",
        help=(
            "Indicator to compress all output tsv files in tsv.gz.",
            " Default is False. Specify --compress will store 'True' value to save disk space.",
            " This command will not compress *.npy files.",
        ),
    )

    peg.add_argument(
        "--platform",
        default="cpu",
        type=str,
        choices=["cpu", "gpu", "tpu"],
        help=(
            "Indicator for the JAX platform. It has to be 'cpu', 'gpu', or 'tpu'. Default is cpu.",
        ),
    )

    peg.add_argument(
        "--jax_precision",
        default=64,
        type=int,
        choices=[32, 64],
        help=(
            "Indicator for the JAX precision: 64-bit or 32-bit.",
            " Default is 64-bit. Choose 32-bit may cause 'elbo decreases' warning.",
        ),
    )

    peg.add_argument(
        "-o",
        "--output",
        default="mrpeg_results",
        help=("Prefix for output files. Default is 'mrpeg_results'.",),
    )

    return peg


def build_closest_parser(subp):
    # add imputation parser
    closest = subp.add_parser(
        "closest",
        description=("Get closest gene",),
    )

    closest.add_argument(
        "--gwas",
        type=str,
        required=True,
        help=("Phenotype data. It has to be a tsv file that contains at least two",),
    )

    # main arguments
    closest.add_argument(
        "--ref",
        type=str,
        required=True,
        help=(
            "Genotype data in plink 1 format. The plink triplet (bed, bim, and fam) should be",
        ),
    )

    closest.add_argument(
        "--gwas_cols",
        nargs=5,
        default=["CHR", "SNP", "BP", "BETA", "SE"],
        type=str,
        help=(
            "Single file that contains subject ID across all ancestries that are used for fine-mapping."
        ),
    )
    closest.add_argument(
        "--ref_cols",
        nargs=4,
        default=["CHR", "P0", "P1", "ANNO"],
        type=str,
        help=(
            "Single file that contains subject ID across all ancestries that are used for fine-mapping."
        ),
    )

    closest.add_argument(
        "--keep",
        default=None,
        type=str,
        help=("keep file."),
    )
    # need to change this name to expand window
    closest.add_argument(
        "--window",
        default=1000,
        type=int,
        help=(
            "Integer number of shared effects pre-specified.",
            " Default is 5. Larger number may cause slow inference.",
        ),
    )

    closest.add_argument(
        "--threshold",
        default=5e-8,
        type=float,
        help=(
            "Integer number of shared effects pre-specified.",
            " Default is 5. Larger number may cause slow inference.",
        ),
    )

    closest.add_argument(
        "--trait",
        default="Trait",
        help=(
            "Trait, tissue, gene name of the phenotype for better indexing in post-hoc analysis. Default is 'Trait'.",
        ),
    )

    # misc options
    closest.add_argument(
        "--quiet",
        default=False,
        action="store_true",
        help="Indicator to not print message to console. Default is False. Specify --numpy will store 'True' value.",
    )

    closest.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help=(
            "Indicator to include debug information in the log. Default is False.",
            " Specify --numpy will store 'True' value.",
        ),
    )

    closest.add_argument(
        "-c",
        "--compress",
        default=False,
        action="store_true",
        help=(
            "Indicator to compress all output tsv files in tsv.gz.",
            " Default is False. Specify --compress will store 'True' value to save disk space.",
            " This command will not compress *.npy files.",
        ),
    )

    closest.add_argument(
        "-o",
        "--output",
        default="mrpeg_results",
        help=("Prefix for output files. Default is 'mrpeg_results'.",),
    )

    return closest


def build_signal_parser(subp):
    # add imputation parser
    signal = subp.add_parser(
        "signal",
        description=("Get closest gene",),
    )

    signal.add_argument(
        "--gwas",
        type=str,
        required=True,
        help=("Phenotype data. It has to be a tsv file that contains at least two",),
    )

    signal.add_argument(
        "--ref",
        type=str,
        required=True,
        help=(
            "Genotype data in plink 1 format. The plink triplet (bed, bim, and fam) should be",
        ),
    )

    signal.add_argument(
        "--gwas_cols",
        nargs=5,
        default=["CHR", "SNP", "BP", "BETA", "SE"],
        type=str,
        help=(
            "Single file that contains subject ID across all ancestries that are used for fine-mapping."
        ),
    )

    signal.add_argument(
        "--ref_cols",
        nargs=4,
        default=["CHR", "START", "END", "ANNO"],
        type=str,
        help=(
            "Single file that contains subject ID across all ancestries that are used for fine-mapping."
        ),
    )

    # main arguments
    signal.add_argument(
        "--chr",
        nargs="+",
        type=int,
        default=None,
        help=("keep file."),
    )

    signal.add_argument(
        "--keep",
        type=str,
        default=None,
        help=("keep file."),
    )

    signal.add_argument(
        "--window",
        default=1000,
        type=int,
        help=(
            "Integer number of shared effects pre-specified.",
            " Default is 5. Larger number may cause slow inference.",
        ),
    )

    signal.add_argument(
        "--split",
        default=",",
        type=str,
        help=(
            "Integer number of shared effects pre-specified.",
            " Default is 5. Larger number may cause slow inference.",
        ),
    )

    signal.add_argument(
        "--threshold",
        default=1.0,
        type=float,
        help=(
            "Integer number of shared effects pre-specified.",
            " Default is 5. Larger number may cause slow inference.",
        ),
    )

    signal.add_argument(
        "--snps_anno",
        default=False,
        type=bool,
        help=(
            "Trait, tissue, gene name of the phenotype for better indexing in post-hoc analysis. Default is 'Trait'.",
        ),
    )

    signal.add_argument(
        "--trait",
        default="Trait",
        type=str,
        help=(
            "Trait, tissue, gene name of the phenotype for better indexing in post-hoc analysis. Default is 'Trait'.",
        ),
    )

    # misc options

    signal.add_argument(
        "--quiet",
        default=False,
        action="store_true",
        help="Indicator to not print message to console. Default is False. Specify --numpy will store 'True' value.",
    )

    signal.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help=(
            "Indicator to include debug information in the log. Default is False.",
            " Specify --numpy will store 'True' value.",
        ),
    )

    signal.add_argument(
        "-c",
        "--compress",
        default=False,
        action="store_true",
        help=(
            "Indicator to compress all output tsv files in tsv.gz.",
            " Default is False. Specify --compress will store 'True' value to save disk space.",
            " This command will not compress *.npy files.",
        ),
    )

    signal.add_argument(
        "-o",
        "--output",
        default="mrpeg_results",
        help=("Prefix for output files. Default is 'mrpeg_results'.",),
    )

    return signal


def _main(argsv):
    # setup main parser
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subp = argp.add_subparsers(
        help="Subcommands: Mendelian Randomization framework integrating Perturb-Seq, eQTL, and GWAS data"
    )

    peg = build_peg_parser(subp)
    peg.set_defaults(func=run_peg)

    closest = build_closest_parser(subp)
    closest.set_defaults(func=run_closest)

    signal = build_signal_parser(subp)
    signal.set_defaults(func=run_signal)

    # parse arguments
    args = argp.parse_args(argsv)

    cmd_str = _get_command_string(argsv)

    version = metadata.version("mrpeg")

    masthead = "===================================" + os.linesep
    masthead += f"             Mr PEG v{version}             " + os.linesep
    masthead += "===================================" + os.linesep

    # setup logging
    log_format = "[%(asctime)s - %(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    if args.verbose:
        log.logger.setLevel(logging.DEBUG)
    else:
        log.logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt=log_format, datefmt=date_format)
    log.logger.propagate = False

    # write to stdout unless quiet is set
    if not args.quiet:
        sys.stdout.write(masthead)
        sys.stdout.write(cmd_str)
        sys.stdout.write("Starting log..." + os.linesep)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)
        log.logger.addHandler(stdout_handler)

    # setup log file, but write PLINK-style command first
    disk_log_stream = open(f"{args.output}.log", "w")
    disk_log_stream.write(masthead)
    disk_log_stream.write(cmd_str)
    disk_log_stream.write("Starting log..." + os.linesep)

    disk_handler = logging.StreamHandler(disk_log_stream)
    disk_handler.setFormatter(fmt)
    log.logger.addHandler(disk_handler)

    # launch finemap
    args.func(args)

    return 0


def run_cli():
    return _main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
