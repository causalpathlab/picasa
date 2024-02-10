r"""
Genomics operations from GLUE paper 
https://github.com/gao-lab/GLUE

"""

import os
import re
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import pybedtools
from pybedtools import BedTool
from pybedtools.cbedtools import Interval
from .util.base import ConstrainedDataFrame


class Bed(ConstrainedDataFrame):

    r"""
    BED format data frame
    """

    COLUMNS = pd.Index([
        "chrom", "chromStart", "chromEnd", "name", "score",
        "strand", "thickStart", "thickEnd", "itemRgb",
        "blockCount", "blockSizes", "blockStarts"
    ])

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super(Bed, cls).rectify(df)
        COLUMNS = cls.COLUMNS.copy(deep=True)
        for item in COLUMNS:
            if item in df:
                if item in ("chromStart", "chromEnd"):
                    df[item] = df[item].astype(int)
                else:
                    df[item] = df[item].astype(str)
            elif item not in ("chrom", "chromStart", "chromEnd"):
                df[item] = "."
            else:
                raise ValueError(f"Required column {item} is missing!")
        return df.loc[:, COLUMNS]

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        super(Bed, cls).verify(df)
        if len(df.columns) != len(cls.COLUMNS) or np.any(df.columns != cls.COLUMNS):
            raise ValueError("Invalid BED format!")

    @classmethod
    def read_bed(cls, fname: os.PathLike) -> "Bed":
        r"""
        Read BED file

        Parameters
        ----------
        fname
            BED file

        Returns
        -------
        bed
            Loaded :class:`Bed` object
        """
        COLUMNS = cls.COLUMNS.copy(deep=True)
        loaded = pd.read_csv(fname, sep="\t", header=None, comment="#")
        loaded.columns = COLUMNS[:loaded.shape[1]]
        return cls(loaded)

    def write_bed(self, fname: os.PathLike, ncols: Optional[int] = None) -> None:
        r"""
        Write BED file

        Parameters
        ----------
        fname
            BED file
        ncols
            Number of columns to write (by default write all columns)
        """
        if ncols and ncols < 3:
            raise ValueError("`ncols` must be larger than 3!")
        df = self.df.iloc[:, :ncols] if ncols else self
        df.to_csv(fname, sep="\t", header=False, index=False)

    def to_bedtool(self) -> pybedtools.BedTool:
        r"""
        Convert to a :class:`pybedtools.BedTool` object

        Returns
        -------
        bedtool
            Converted :class:`pybedtools.BedTool` object
        """
        return BedTool(Interval(
            row["chrom"], row["chromStart"], row["chromEnd"],
            name=row["name"], score=row["score"], strand=row["strand"]
        ) for _, row in self.iterrows())

    def nucleotide_content(self, fasta: os.PathLike) -> pd.DataFrame:
        r"""
        Compute nucleotide content in the BED regions

        Parameters
        ----------
        fasta
            Genomic sequence file in FASTA format

        Returns
        -------
        nucleotide_stat
            Data frame containing nucleotide content statistics for each region
        """
        result = self.to_bedtool().nucleotide_content(fi=os.fspath(fasta), s=True)  # pylint: disable=unexpected-keyword-arg
        result = pd.DataFrame(
            np.stack([interval.fields[6:15] for interval in result]),
            columns=[
                r"%AT", r"%GC",
                r"#A", r"#C", r"#G", r"#T", r"#N",
                r"#other", r"length"
            ]
        ).astype({
            r"%AT": float, r"%GC": float,
            r"#A": int, r"#C": int, r"#G": int, r"#T": int, r"#N": int,
            r"#other": int, r"length": int
        })
        pybedtools.cleanup()
        return result

    def strand_specific_start_site(self) -> "Bed":
        r"""
        Convert to strand-specific start sites of genomic features

        Returns
        -------
        start_site_bed
            A new :class:`Bed` object, containing strand-specific start sites
            of the current :class:`Bed` object
        """
        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromEnd"] = df.loc[pos_strand, "chromStart"] + 1
        df.loc[neg_strand, "chromStart"] = df.loc[neg_strand, "chromEnd"] - 1
        return type(self)(df)

    def strand_specific_end_site(self) -> "Bed":
        r"""
        Convert to strand-specific end sites of genomic features

        Returns
        -------
        end_site_bed
            A new :class:`Bed` object, containing strand-specific end sites
            of the current :class:`Bed` object
        """
        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromStart"] = df.loc[pos_strand, "chromEnd"] - 1
        df.loc[neg_strand, "chromEnd"] = df.loc[neg_strand, "chromStart"] + 1
        return type(self)(df)

    def expand(
            self, upstream: int, downstream: int,
            chr_len: Optional[Mapping[str, int]] = None
    ) -> "Bed":
        r"""
        Expand genomic features towards upstream and downstream

        Parameters
        ----------
        upstream
            Number of bps to expand in the upstream direction
        downstream
            Number of bps to expand in the downstream direction
        chr_len
            Length of each chromosome

        Returns
        -------
        expanded_bed
            A new :class:`Bed` object, containing expanded features
            of the current :class:`Bed` object

        Note
        ----
        Starting position < 0 after expansion is always trimmed.
        Ending position exceeding chromosome length is trimed only if
        ``chr_len`` is specified.
        """
        if upstream == downstream == 0:
            return self
        df = pd.DataFrame(self, copy=True)
        if upstream == downstream:  # symmetric
            df["chromStart"] -= upstream
            df["chromEnd"] += downstream
        else:  # asymmetric
            if set(df["strand"]) != set(["+", "-"]):
                raise ValueError("Not all features are strand specific!")
            pos_strand = df.query("strand == '+'").index
            neg_strand = df.query("strand == '-'").index
            if upstream:
                df.loc[pos_strand, "chromStart"] -= upstream
                df.loc[neg_strand, "chromEnd"] += upstream
            if downstream:
                df.loc[pos_strand, "chromEnd"] += downstream
                df.loc[neg_strand, "chromStart"] -= downstream
        df["chromStart"] = np.maximum(df["chromStart"], 0)
        if chr_len:
            chr_len = df["chrom"].map(chr_len)
            df["chromEnd"] = np.minimum(df["chromEnd"], chr_len)
        return type(self)(df)


class Gtf(ConstrainedDataFrame):  # gffutils is too slow

    r"""
    GTF format data frame
    """

    COLUMNS = pd.Index([
        "seqname", "source", "feature", "start", "end",
        "score", "strand", "frame", "attribute"
    ])  # Additional columns after "attribute" is allowed

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super(Gtf, cls).rectify(df)
        COLUMNS = cls.COLUMNS.copy(deep=True)
        for item in COLUMNS:
            if item in df:
                if item in ("start", "end"):
                    df[item] = df[item].astype(int)
                else:
                    df[item] = df[item].astype(str)
            elif item not in ("seqname", "start", "end"):
                df[item] = "."
            else:
                raise ValueError(f"Required column {item} is missing!")
        return df.sort_index(axis=1, key=cls._column_key)

    @classmethod
    def _column_key(cls, x: pd.Index) -> np.ndarray:
        x = cls.COLUMNS.get_indexer(x)
        x[x < 0] = x.max() + 1  # Put additional columns after "attribute"
        return x

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        super(Gtf, cls).verify(df)
        if len(df.columns) < len(cls.COLUMNS) or \
                np.any(df.columns[:len(cls.COLUMNS)] != cls.COLUMNS):
            raise ValueError("Invalid GTF format!")

    @classmethod
    def read_gtf(cls, fname: os.PathLike) -> "Gtf":
        r"""
        Read GTF file

        Parameters
        ----------
        fname
            GTF file

        Returns
        -------
        gtf
            Loaded :class:`Gtf` object
        """
        COLUMNS = cls.COLUMNS.copy(deep=True)
        loaded = pd.read_csv(fname, sep="\t", header=None, comment="#")
        loaded.columns = COLUMNS[:loaded.shape[1]]
        return cls(loaded)

    def split_attribute(self) -> "Gtf":
        r"""
        Extract all attributes from the "attribute" column
        and append them to existing columns

        Returns
        -------
        splitted
            Gtf with splitted attribute columns appended
        """
        pattern = re.compile(r'([^\s]+) "([^"]+)";')
        splitted = pd.DataFrame.from_records(np.vectorize(lambda x: {
            key: val for key, val in pattern.findall(x)
        })(self["attribute"]), index=self.index)
        if set(self.COLUMNS).intersection(splitted.columns):
            self.logger.warning(
                "Splitted attribute names overlap standard GTF fields! "
                "The standard fields are overwritten!"
            )
        return self.assign(**splitted)

    def to_bed(self, name: Optional[str] = None) -> Bed:
        r"""
        Convert GTF to BED format

        Parameters
        ----------
        name
            Specify a column to be converted to the "name" column in bed format,
            otherwise the "name" column would be filled with "."

        Returns
        -------
        bed
            Converted :class:`Bed` object
        """
        bed_df = pd.DataFrame(self, copy=True).loc[
            :, ("seqname", "start", "end", "score", "strand")
        ]
        bed_df.insert(3, "name", np.repeat(
            ".", len(bed_df)
        ) if name is None else self[name])
        bed_df["start"] -= 1  # Convert to zero-based
        bed_df.columns = (
            "chrom", "chromStart", "chromEnd", "name", "score", "strand"
        )
        return Bed(bed_df)


# Aliases
read_bed = Bed.read_bed
read_gtf = Gtf.read_gtf
