"""Helpers for dx/dt metadata tables used by :class:`DXDTDataset`."""

from __future__ import annotations

import pandas as pd


def filter_min_dose(meta: pd.DataFrame, min_dose_um: float | None) -> pd.DataFrame:
    """Keep rows with ``dose >= min_dose_um`` (µM).

    Parameters
    ----------
    meta :
        dxdt_meta table with a ``dose`` column.
    min_dose_um :
        Minimum dose in µM. When ``None``, returns ``meta`` unchanged.
    """
    if min_dose_um is None:
        return meta.reset_index(drop=True)

    if 'dose' not in meta.columns:
        raise ValueError('dxdt_meta is missing required column: dose')

    out = meta[meta['dose'] >= float(min_dose_um)]
    return out.reset_index(drop=True)


def subsample(meta: pd.DataFrame, frac: float = 1.0, *, seed: int = 0) -> pd.DataFrame:
    """Randomly subsample rows of a dxdt_meta table by proportion.

    Parameters
    ----------
    meta :
        Table with at least ``file_name``, ``pert_id``, ``dose``, ``cell_iname``,
        and ``time`` columns (same schema as ``dxdt_meta.csv``).
    frac :
        Fraction of rows to keep in ``(0, 1]``. ``1.0`` returns all rows.
    seed :
        Random seed passed to :meth:`pandas.DataFrame.sample`.

    Returns
    -------
    pandas.DataFrame
        Subsampled copy with reset index.
    """
    if frac == 1.0:
        return meta.reset_index(drop=True)

    if not 0.0 < frac < 1.0:
        raise ValueError(f'subsample frac must be in (0, 1]; got {frac}')

    if len(meta) == 0:
        return meta.reset_index(drop=True)

    return meta.sample(frac=frac, random_state=seed).reset_index(drop=True)
