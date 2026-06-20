"""Unit tests for :mod:`lincs_gsnn.proc.cell_drug_split`."""

from __future__ import annotations

import tempfile

import pandas as pd
import pytest

from lincs_gsnn.proc.cell_drug_split import (
    build_cell_drug_split,
    filter_meta_by_partition,
    load_cell_drug_split,
    save_cell_drug_split,
)


CELLS_11 = [f"CELL{i}" for i in range(11)]
DRUGS_3 = ["D0", "D1", "D2"]


def test_build_split_11_cells_3_drugs():
    split = build_cell_drug_split(DRUGS_3, CELLS_11, n_val=1, seed=42)
    assert len(split) == 33
    assert (split["partition"] == "val").sum() == 3
    assert (split["partition"] == "train").sum() == 30
    per_drug = split.groupby("pert_id")["partition"].value_counts().unstack(fill_value=0)
    assert (per_drug["val"] == 1).all()
    assert (per_drug["train"] == 10).all()


def test_reproducible_with_same_seed():
    a = build_cell_drug_split(DRUGS_3, CELLS_11, n_val=1, seed=7)
    b = build_cell_drug_split(DRUGS_3, CELLS_11, n_val=1, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_val_cells():
    a = build_cell_drug_split(DRUGS_3, CELLS_11, n_val=1, seed=1)
    b = build_cell_drug_split(DRUGS_3, CELLS_11, n_val=1, seed=2)
    val_a = set(zip(a.loc[a.partition == "val", "pert_id"], a.loc[a.partition == "val", "cell_iname"]))
    val_b = set(zip(b.loc[b.partition == "val", "pert_id"], b.loc[b.partition == "val", "cell_iname"]))
    assert val_a != val_b


def test_filter_meta_keeps_all_rows_per_pair():
    split = build_cell_drug_split(["D0"], ["A", "B"], n_val=1, seed=0)
    meta = pd.DataFrame({
        "pert_id": ["D0", "D0", "D0", "D0"],
        "cell_iname": ["A", "A", "B", "B"],
        "dose": [0.0, 1.0, 0.0, 1.0],
        "file_name": ["f0", "f1", "f2", "f3"],
    })
    train = filter_meta_by_partition(meta, split, "train")
    val = filter_meta_by_partition(meta, split, "val")
    assert len(train) + len(val) == len(meta)
    # one cell held out -> 2 rows in val partition, 2 in train
    assert len(val) == 2
    assert len(train) == 2


def test_save_and_load_roundtrip():
    split = build_cell_drug_split(DRUGS_3, CELLS_11, n_val=1, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/cell_drug_split.csv"
        save_cell_drug_split(path, split)
        loaded = load_cell_drug_split(path)
    pd.testing.assert_frame_equal(split.reset_index(drop=True), loaded.reset_index(drop=True))


def test_n_val_must_be_less_than_n_cells():
    with pytest.raises(ValueError, match="n_val"):
        build_cell_drug_split(["D0"], ["A"], n_val=1, seed=0)


def test_n_val_invariant_holds_for_many_seeds():
    for seed in range(10):
        split = build_cell_drug_split(DRUGS_3, CELLS_11, n_val=1, seed=seed)
        per_drug = split.groupby("pert_id")["partition"].value_counts().unstack(fill_value=0)
        assert (per_drug["val"] == 1).all()
        assert (per_drug["train"] == 10).all()
