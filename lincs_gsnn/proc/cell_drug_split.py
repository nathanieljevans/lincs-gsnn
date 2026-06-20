"""Cell-drug train/validation split for LINCS-GSNN training."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SPLIT_COLUMNS = ("pert_id", "cell_iname", "partition")
VALID_PARTITIONS = frozenset({"train", "val"})


def build_cell_drug_split(
    pert_ids: Sequence[str],
    cell_inames: Sequence[str],
    n_val: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign each (pert_id, cell_iname) pair to train or val.

    For every drug, ``n_val`` cell lines are held out for validation and the
    remainder go to training. Requires ``len(cell_inames) > n_val``.

    Parameters
    ----------
    pert_ids
        Drug identifiers (LINCS ``pert_id`` values).
    cell_inames
        Cell-line identifiers (LINCS ``cell_iname`` values).
    n_val
        Number of validation cells per drug (default 1 of 11).
    seed
        RNG seed for reproducible per-drug shuffles.

    Returns
    -------
    pd.DataFrame
        Columns: ``pert_id``, ``cell_iname``, ``partition``.
    """
    pert_ids = [str(p) for p in pert_ids]
    cell_inames = [str(c) for c in cell_inames]
    n_val = int(n_val)
    n_cells = len(cell_inames)

    if n_val < 1:
        raise ValueError(f"n_val must be >= 1, got {n_val}")
    if n_val >= n_cells:
        raise ValueError(
            f"n_val ({n_val}) must be < number of cell lines ({n_cells})"
        )
    if not pert_ids:
        raise ValueError("pert_ids must be non-empty")
    if not cell_inames:
        raise ValueError("cell_inames must be non-empty")

    rng = np.random.default_rng(seed)
    rows = []
    for pert_id in pert_ids:
        order = rng.permutation(n_cells)
        val_cells = {cell_inames[i] for i in order[:n_val]}
        for cell in cell_inames:
            partition = "val" if cell in val_cells else "train"
            rows.append({"pert_id": pert_id, "cell_iname": cell, "partition": partition})

    split_df = pd.DataFrame(rows, columns=list(SPLIT_COLUMNS))
    _validate_split(split_df, pert_ids, cell_inames, n_val)
    return split_df


def _validate_split(
    split_df: pd.DataFrame,
    pert_ids: Sequence[str],
    cell_inames: Sequence[str],
    n_val: int,
) -> None:
    expected_pairs = len(pert_ids) * len(cell_inames)
    if len(split_df) != expected_pairs:
        raise ValueError(
            f"split has {len(split_df)} rows, expected {expected_pairs} "
            f"({len(pert_ids)} drugs x {len(cell_inames)} cells)"
        )
    per_drug = split_df.groupby("pert_id")["partition"].value_counts().unstack(fill_value=0)
    n_train = len(cell_inames) - n_val
    bad = per_drug[(per_drug.get("val", 0) != n_val) | (per_drug.get("train", 0) != n_train)]
    if len(bad):
        raise ValueError(
            f"split validation failed for {len(bad)} drug(s); "
            f"expected {n_train} train and {n_val} val cells per drug. "
            f"First offenders:\n{bad.head()}"
        )


def save_cell_drug_split(path: str, split_df: pd.DataFrame) -> None:
    """Write split table to CSV."""
    missing = set(SPLIT_COLUMNS) - set(split_df.columns)
    if missing:
        raise ValueError(f"split_df missing columns: {sorted(missing)}")
    bad = set(split_df["partition"].unique()) - VALID_PARTITIONS
    if bad:
        raise ValueError(f"invalid partition values: {bad}")
    split_df[list(SPLIT_COLUMNS)].to_csv(path, index=False)


def load_cell_drug_split(path: str) -> pd.DataFrame:
    """Load split table from CSV."""
    df = pd.read_csv(path)
    missing = set(SPLIT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    bad = set(df["partition"].unique()) - VALID_PARTITIONS
    if bad:
        raise ValueError(f"{path}: invalid partition values {bad}")
    return df


def filter_meta_by_partition(
    meta: pd.DataFrame,
    split_df: pd.DataFrame,
    partition: str,
) -> pd.DataFrame:
    """Filter metadata rows to one partition via (pert_id, cell_iname) join."""
    if partition not in VALID_PARTITIONS:
        raise ValueError(f"partition must be one of {VALID_PARTITIONS}, got {partition!r}")
    keys = split_df.loc[split_df["partition"] == partition, ["pert_id", "cell_iname"]]
    merged = meta.merge(keys, on=["pert_id", "cell_iname"], how="inner")
    return merged.reset_index(drop=True)


def summarize_split(split_df: pd.DataFrame) -> Mapping[str, float | int]:
    """Return summary counts for logging."""
    n_pairs = len(split_df)
    n_val = int((split_df["partition"] == "val").sum())
    n_train = int((split_df["partition"] == "train").sum())
    return {
        "n_pairs": n_pairs,
        "n_drugs": int(split_df["pert_id"].nunique()),
        "n_cells": int(split_df["cell_iname"].nunique()),
        "n_train_pairs": n_train,
        "n_val_pairs": n_val,
        "val_fraction": n_val / n_pairs if n_pairs else 0.0,
    }
