"""Console table logging helpers for GSNN training scripts."""

from __future__ import annotations

import time

import torch


def peak_mem_gb(device) -> float:
    if isinstance(device, torch.device):
        dev = device.type
    else:
        dev = str(device)
    if dev == 'cuda' and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def reset_peak_mem(device) -> None:
    if isinstance(device, torch.device):
        dev = device.type
    else:
        dev = str(device)
    if dev == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def erase_progress_line(width=160) -> None:
    """Clear the in-progress status line before printing a table row."""
    print('\r' + ' ' * width + '\r', end='', flush=True)


def format_duration(seconds: float) -> str:
    """Compact duration string for progress lines (e.g. ``45s``, ``2m03s``)."""
    seconds = max(0.0, float(seconds))
    total = int(seconds + 0.5)
    if total < 60:
        return f'{total}s'
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f'{minutes}m{secs:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h{minutes:02d}m'


def format_train_batch_progress(
    batch_idx: int,
    n_batches: int,
    *,
    epoch_start: float,
    loss: float,
    r2: float,
) -> str:
    """In-epoch status line with per-batch loss/R² and elapsed/ETA timing."""
    done = batch_idx + 1
    elapsed = time.perf_counter() - epoch_start
    timing = f'elapsed: {format_duration(elapsed)}'
    if done < n_batches:
        eta = elapsed / done * (n_batches - done)
        timing = f'{timing}, eta: {format_duration(eta)}'
    return (
        f'[train batch {batch_idx}/{n_batches} -> '
        f'loss: {loss:.2E}, r2: {r2:.2f}, {timing}]'
    )


# Table column: (value_key, header_label, width, printf-style fmt).
TableCol = tuple[str, str, int, str]

# .3E values (e.g. 1.234E+05, -2.500E+00) fit in 10 columns; .4E needs 11.
SCIENTIFIC_COL_WIDTH = 10
SCIENTIFIC_FMT = '.3E'


def fmt_cell(value, width, *, fmt='.3f', missing='-'):
    if value is None:
        return f'{missing:>{width}}'
    if isinstance(value, float) and value != value:
        return f'{"nan":>{width}}'
    if fmt == 'd':
        return f'{int(value):>{width}d}'
    text = format(value, fmt)
    if len(text) > width and 'E' in fmt.upper():
        for prec in (2, 1, 0):
            text = format(value, f'.{prec}E')
            if len(text) <= width:
                break
    return f'{text:>{width}}'


def format_header(cols: list[tuple[str, int]]) -> str:
    """Format a table header from (label, width) pairs."""
    return ' '.join(f'{label:>{width}}' for label, width in cols)


def format_row(cells: list[str]) -> str:
    return ' '.join(cells)


def table_header(cols: list[TableCol]) -> str:
    """Format a header row from :data:`TableCol` specs."""
    return format_header([(label, width) for _, label, width, _ in cols])


def table_row(cols: list[TableCol], values: dict[str, object]) -> str:
    """Format one aligned data row using the same column specs as :func:`table_header`."""
    cells = [
        fmt_cell(values[key], width, fmt=fmt)
        for key, _, width, fmt in cols
    ]
    return format_row(cells)


def pretrain_epoch_table_columns(*, use_gamma_prior: bool) -> list[TableCol]:
    """Column layout shared by pretrain header and epoch rows."""
    cols: list[TableCol] = [
        ('epoch', 'ep', 4, 'd'),
        ('train_nll', 'tr_nll', SCIENTIFIC_COL_WIDTH, SCIENTIFIC_FMT),
        ('train_mse', 'tr_mse', SCIENTIFIC_COL_WIDTH, SCIENTIFIC_FMT),
        ('train_r2', 'tr_r2', 7, '.3f'),
        ('val_nll', 'v_nll', SCIENTIFIC_COL_WIDTH, SCIENTIFIC_FMT),
        ('val_mse', 'v_mse', SCIENTIFIC_COL_WIDTH, SCIENTIFIC_FMT),
        ('val_r2', 'v_r2', 7, '.3f'),
    ]
    if use_gamma_prior:
        cols.append(('train_gamma_prior', 'g_prior', SCIENTIFIC_COL_WIDTH, SCIENTIFIC_FMT))
    cols.extend([
        ('lr', 'lr', 9, '.2E'),
        ('best_val', 'best_v', SCIENTIFIC_COL_WIDTH, SCIENTIFIC_FMT),
        ('time_s', 'secs', 6, '.1f'),
        ('max_mem_gb', 'memGB', 6, '.2f'),
    ])
    return cols


def train_epoch_table_columns(*, use_gamma_prior: bool) -> list[TableCol]:
    """Column layout shared by odeint-train header and epoch rows."""
    cols: list[TableCol] = [
        ('epoch', 'ep', 4, 'd'),
        ('train_nll', 'tr_nll', SCIENTIFIC_COL_WIDTH, SCIENTIFIC_FMT),
        ('train_mse', 'tr_mse', SCIENTIFIC_COL_WIDTH, SCIENTIFIC_FMT),
        ('train_r2', 'tr_r2', 7, '.3f'),
        ('train_time_series_r', 'tr_tsr', 7, '.3f'),
        ('val_nll', 'v_nll', SCIENTIFIC_COL_WIDTH, SCIENTIFIC_FMT),
        ('val_mse', 'v_mse', SCIENTIFIC_COL_WIDTH, SCIENTIFIC_FMT),
        ('val_r2', 'v_r2', 7, '.3f'),
        ('val_time_series_r', 'v_tsr', 7, '.3f'),
    ]
    if use_gamma_prior:
        cols.append(('train_gamma_prior', 'g_prior', SCIENTIFIC_COL_WIDTH, SCIENTIFIC_FMT))
    cols.extend([
        ('lr', 'lr', 9, '.2E'),
        ('best_val', 'best_v', SCIENTIFIC_COL_WIDTH, SCIENTIFIC_FMT),
        ('time_s', 'secs', 6, '.1f'),
        ('max_mem_gb', 'memGB', 6, '.2f'),
    ])
    return cols


def configure_cuda_performance(enable_tf32: bool) -> None:
    if not enable_tf32:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
