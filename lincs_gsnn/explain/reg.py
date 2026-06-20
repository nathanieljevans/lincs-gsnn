'''Config-parameter encoding and GP surrogate models for experiment comparison.

Encodes flattened workflow config columns (``config__*`` from eval CSVs) into a
numeric design matrix, fits a Gaussian process surrogate, and ranks unseen
Cartesian-product configurations by Expected Improvement.
'''

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import itertools
import math

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ParamKind = Literal['boolean', 'binary', 'categorical', 'numeric', 'json_list']

LOG10_PARAMS = frozenset({
    'config__pretrain__lr',
    'config__pretrain__wd',
    'config__train__lr',
    'config__train__wd',
    'config__train__tol',
    'config__node_activity__alpha_decay',
})

DEFAULT_EXCLUDE_PREFIXES = (
    'config__dirs__',
    'config__scripts__',
    'config__contrastive_explanations__',
)

DEFAULT_EXCLUDE_EXACT = frozenset({
    'config__run_id',
    'config__snakemake_workdir',
    'config__make_bio_network__gene_stats',
})

JSON_LIST_FEATURE_COLS = ('has_expr', 'has_mut', 'has_cell_line')


@dataclass
class ParamSpec:
    '''Metadata for one hyperparameter (used for encoding and future GP search).'''

    name: str
    kind: ParamKind
    n_unique: int
    observed_values: list[Any] = field(default_factory=list)
    encoded_columns: list[str] = field(default_factory=list)
    bounds: tuple[float, float] | None = None
    log_transform: bool = False
    categories: list[str] = field(default_factory=list)


@dataclass
class EncodedConfig:
    '''Design matrix and metadata from ``encode_config_frame``.'''

    X: pd.DataFrame
    column_names: list[str]
    param_groups: dict[str, list[str]]
    param_specs: dict[str, ParamSpec]
    formula_names: dict[str, str]


@dataclass
class GPSurrogateResult:
    '''Output of ``fit_gp_surrogate``.'''

    gp: GaussianProcessRegressor
    scaler_X: StandardScaler
    scaler_y: StandardScaler
    design_columns: list[str]
    param_cols: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    kernel_: str
    loo_predictions: np.ndarray
    loo_std: np.ndarray
    loo_r2: float
    target: str
    encoded: EncodedConfig
    feature_fill_values: dict[str, float]


def _impute_design_frame(
    X: pd.DataFrame,
    fill_values: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    '''Fill missing / non-finite encoded features with column medians.'''
    fills: dict[str, float] = {}
    X_out = X.copy()
    for col in X_out.columns:
        series = X_out[col].replace([np.inf, -np.inf], np.nan)
        if fill_values is not None and col in fill_values:
            fill = fill_values[col]
        elif series.notna().any():
            fill = float(series.median())
        else:
            fill = 0.0
        fills[col] = fill
        X_out[col] = series.fillna(fill)
    return X_out, fills


def _is_path_like(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith('/') or value.startswith('../')


def _binary_encode(series: pd.Series) -> pd.Series:
    '''Encode a two-level categorical as 0/1 (sorted level order).'''
    levels = sorted(series.dropna().astype(str).unique())
    if len(levels) != 2:
        return series.map(bool_from_config_value)
    mapping = {levels[0]: 0.0, levels[1]: 1.0}
    return series.astype(str).map(mapping)


def bool_from_config_value(value: Any) -> float:
    '''Map YAML / CLI config scalars to 0/1.

    Empty strings, NaN, ``false``, ``0``, and ``--no-*`` flags -> 0.
    Non-empty CLI flags, ``true``, and other truthy values -> 1.
    '''
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, np.integer)):
        return float(value != 0)
    if isinstance(value, (float, np.floating)):
        return float(value != 0.0 and not np.isnan(value))
    text = str(value).strip()
    if text == '':
        return 0.0
    if text.lower() in {'false', '0', 'none', 'null', 'nan'}:
        return 0.0
    if text.startswith('--no-'):
        return 0.0
    return 1.0


def _short_param_name(col: str, *, prefix: str = 'config__') -> str:
    if col.startswith(prefix):
        return col[len(prefix):].replace('__', '_')
    return col.replace('__', '_')


def _infer_param_kind(series: pd.Series) -> ParamKind:
    non_null = series.dropna()
    if non_null.empty:
        return 'categorical'

    if non_null.name in LOG10_PARAMS or pd.api.types.is_numeric_dtype(non_null):
        unique = non_null.nunique()
        if unique <= 2 and non_null.name not in LOG10_PARAMS:
            vals = set(non_null.unique())
            if vals <= {0, 1, 0.0, 1.0, True, False}:
                return 'boolean'
        return 'numeric'

    sample = non_null.iloc[0]
    if isinstance(sample, str) and sample.startswith('['):
        return 'json_list'

    unique_vals = non_null.astype(str).unique()
    if len(unique_vals) == 2:
        bool_vals = [bool_from_config_value(v) for v in unique_vals]
        if len(set(bool_vals)) == 1:
            return 'binary'
        return 'boolean'
    return 'categorical'


def select_varying_config_columns(
    df: pd.DataFrame,
    *,
    prefix: str = 'config__',
    min_unique: int = 2,
    exclude_prefixes: tuple[str, ...] = DEFAULT_EXCLUDE_PREFIXES,
    exclude_exact: frozenset[str] = DEFAULT_EXCLUDE_EXACT,
) -> tuple[list[str], pd.DataFrame]:
    '''Return hyperparameter columns that vary across runs plus a metadata table.'''
    config_cols = [c for c in df.columns if c.startswith(prefix)]
    rows: list[dict[str, Any]] = []
    selected: list[str] = []

    for col in sorted(config_cols):
        if col in exclude_exact:
            continue
        if any(col.startswith(p) for p in exclude_prefixes):
            continue

        series = df[col]
        n_unique = series.nunique(dropna=False)
        if n_unique < min_unique:
            continue

        sample_vals = series.dropna().unique()[:5]
        if any(_is_path_like(v) for v in sample_vals):
            continue

        kind = _infer_param_kind(series)
        rows.append({
            'param': col,
            'short_name': _short_param_name(col, prefix=prefix),
            'n_unique': n_unique,
            'kind': kind,
            'example_values': list(sample_vals),
        })
        selected.append(col)

    meta = pd.DataFrame(rows)
    return selected, meta


def _parse_json_list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(v) for v in parsed]
    return []


def encode_config_frame(
    df: pd.DataFrame,
    param_cols: list[str],
    *,
    prefix: str = 'config__',
) -> EncodedConfig:
    '''Encode selected config columns into a numeric design matrix.'''
    encoded_parts: list[pd.DataFrame] = []
    param_groups: dict[str, list[str]] = {}
    param_specs: dict[str, ParamSpec] = {}
    formula_names: dict[str, str] = {}

    for col in param_cols:
        series = df[col]
        short = _short_param_name(col, prefix=prefix)
        kind = _infer_param_kind(series)
        observed = list(series.dropna().unique())

        if kind == 'json_list':
            feature_frame = pd.DataFrame(index=df.index)
            col_names: list[str] = []
            for feat, label in zip(
                ('expr', 'mut', 'cell_line'),
                JSON_LIST_FEATURE_COLS,
                strict=True,
            ):
                cname = f'{short}__{label}'
                feature_frame[cname] = series.map(
                    lambda v, f=feat: float(f in _parse_json_list(v)),
                )
                col_names.append(cname)
            encoded_parts.append(feature_frame)
            param_groups[col] = col_names
            param_specs[col] = ParamSpec(
                name=col,
                kind='json_list',
                n_unique=series.nunique(),
                observed_values=observed,
                encoded_columns=col_names,
                categories=list(JSON_LIST_FEATURE_COLS),
            )
            formula_names[col] = ' + '.join(col_names)
            continue

        if kind == 'numeric':
            numeric = pd.to_numeric(series, errors='coerce')
            use_log = col in LOG10_PARAMS
            cname = f'{short}__log10' if use_log else short
            if use_log:
                values = np.log10(numeric.where(numeric > 0))
            else:
                values = numeric
            part = pd.DataFrame({cname: values}, index=df.index)
            encoded_parts.append(part)
            param_groups[col] = [cname]
            finite = values.replace([np.inf, -np.inf], np.nan).dropna()
            bounds = (float(finite.min()), float(finite.max())) if len(finite) else None
            param_specs[col] = ParamSpec(
                name=col,
                kind='numeric',
                n_unique=series.nunique(),
                observed_values=observed,
                encoded_columns=[cname],
                bounds=bounds,
                log_transform=use_log,
            )
            formula_names[col] = cname
            continue

        if kind in {'boolean', 'binary'}:
            if kind == 'boolean':
                values = series.map(bool_from_config_value)
            else:
                values = _binary_encode(series)
            cname = short
            part = pd.DataFrame({cname: values}, index=df.index)
            encoded_parts.append(part)
            param_groups[col] = [cname]
            param_specs[col] = ParamSpec(
                name=col,
                kind=kind,
                n_unique=series.nunique(),
                observed_values=observed,
                encoded_columns=[cname],
                categories=[str(v) for v in observed],
            )
            formula_names[col] = cname
            continue

        # categorical: one-hot, drop first level
        cat = series.astype(str).fillna('__missing__')
        dummies = pd.get_dummies(cat, prefix=short, drop_first=True, dtype=float)
        encoded_parts.append(dummies)
        dummy_cols = list(dummies.columns)
        param_groups[col] = dummy_cols
        param_specs[col] = ParamSpec(
            name=col,
            kind='categorical',
            n_unique=series.nunique(),
            observed_values=observed,
            encoded_columns=dummy_cols,
            categories=sorted(cat.unique()),
        )
        formula_names[col] = f'C({short})'

    if encoded_parts:
        X = pd.concat(encoded_parts, axis=1)
    else:
        X = pd.DataFrame(index=df.index)

    return EncodedConfig(
        X=X,
        column_names=list(X.columns),
        param_groups=param_groups,
        param_specs=param_specs,
        formula_names=formula_names,
    )


def prepare_regression_frame(
    res: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    target: str = 'train__mean_best_val_r2',
    require_train_enabled: bool = True,
) -> pd.DataFrame:
    '''Merge eval results with summary targets and drop invalid rows.'''
    merged = summary_df.merge(res, on='run_id', how='inner', suffixes=('', '_res'))

    if require_train_enabled and 'config__train__enabled' in merged.columns:
        enabled = merged['config__train__enabled'].map(bool_from_config_value) > 0
        merged = merged.loc[enabled]

    y = pd.to_numeric(merged[target], errors='coerce')
    valid = y.notna() & np.isfinite(y)
    if 'r2' in target.lower():
        valid &= y.between(-1.0, 1.0)
    frame = merged.loc[valid].copy()
    frame[target] = y.loc[valid].astype(float)
    return frame.reset_index(drop=True)


def param_specs_to_search_space(
    param_specs: dict[str, ParamSpec],
) -> pd.DataFrame:
    '''Summarize observed hyperparameter ranges for future GP / BO.'''
    rows: list[dict[str, Any]] = []
    for spec in param_specs.values():
        row: dict[str, Any] = {
            'param': spec.name,
            'kind': spec.kind,
            'n_unique': spec.n_unique,
            'observed_values': spec.observed_values,
            'encoded_columns': spec.encoded_columns,
        }
        if spec.bounds is not None:
            row['lower'] = spec.bounds[0]
            row['upper'] = spec.bounds[1]
        if spec.categories:
            row['categories'] = spec.categories
        if spec.log_transform:
            row['log_transform'] = True
        rows.append(row)
    return pd.DataFrame(rows)


def decode_config_row(
    row: pd.Series,
    param_cols: list[str],
    *,
    prefix: str = 'config__',
) -> dict[str, Any]:
    '''Decode one ranking-table row back to config parameter values.'''
    return {
        col: row[_short_param_name(col, prefix=prefix)]
        for col in param_cols
        if _short_param_name(col, prefix=prefix) in row.index
    }


def _default_gp_kernel():
    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-6, 1e-1))
    )


def _encode_candidates_aligned(
    frame: pd.DataFrame,
    candidates: pd.DataFrame,
    param_cols: list[str],
    design_columns: list[str],
    *,
    fill_values: dict[str, float] | None = None,
    prefix: str = 'config__',
) -> pd.DataFrame:
    '''Encode candidates with the same dummy columns as the training frame.'''
    combined = pd.concat(
        [frame[param_cols], candidates[param_cols]],
        ignore_index=True,
    )
    encoded = encode_config_frame(combined, param_cols, prefix=prefix)
    X_cand = encoded.X.iloc[len(frame):].copy()
    X_cand = X_cand.reindex(columns=design_columns, fill_value=0.0)
    X_cand, _ = _impute_design_frame(X_cand, fill_values=fill_values)
    return X_cand


def _leave_one_out_gp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel,
    alpha: float,
    n_restarts: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    '''Leave-one-out GP predictions for diagnostics.'''
    n = len(y)
    loo_mu = np.empty(n, dtype=float)
    loo_std = np.empty(n, dtype=float)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=False,
            n_restarts_optimizer=n_restarts,
            random_state=random_state,
        )
        gp.fit(X[mask], y[mask])
        mu_i, std_i = gp.predict(X[i:i + 1], return_std=True)
        loo_mu[i] = float(mu_i[0])
        loo_std[i] = float(std_i[0])

    return loo_mu, loo_std


def fit_gp_surrogate(
    frame: pd.DataFrame,
    param_cols: list[str],
    *,
    target: str = 'train__mean_best_val_r2',
    kernel=None,
    alpha: float = 1e-6,
    n_restarts: int = 10,
    random_state: int = 0,
    prefix: str = 'config__',
) -> GPSurrogateResult:
    '''Fit a Gaussian process surrogate on the encoded hyperparameter design matrix.'''
    encoded = encode_config_frame(frame, param_cols, prefix=prefix)
    X_df, feature_fill_values = _impute_design_frame(encoded.X)
    X_raw = X_df.astype(float).values
    y_raw = frame[target].astype(float).values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_raw)
    y_train = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    if kernel is None:
        kernel = _default_gp_kernel()

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=False,
        n_restarts_optimizer=n_restarts,
        random_state=random_state,
    )
    gp.fit(X_train, y_train)

    loo_mu_std, loo_std_std = _leave_one_out_gp(
        X_train,
        y_train,
        kernel=kernel,
        alpha=alpha,
        n_restarts=n_restarts,
        random_state=random_state,
    )
    loo_mu = scaler_y.inverse_transform(loo_mu_std.reshape(-1, 1)).ravel()
    loo_std = loo_std_std * float(scaler_y.scale_[0])
    loo_r2 = float(r2_score(y_raw, loo_mu))

    return GPSurrogateResult(
        gp=gp,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        design_columns=list(encoded.column_names),
        param_cols=list(param_cols),
        X_train=X_train,
        y_train=y_raw,
        kernel_=str(gp.kernel_),
        loo_predictions=loo_mu,
        loo_std=loo_std,
        loo_r2=loo_r2,
        target=target,
        encoded=encoded,
        feature_fill_values=feature_fill_values,
    )


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    y_best: float,
    *,
    xi: float = 0.01,
    maximize: bool = True,
) -> np.ndarray:
    '''Expected Improvement for a maximization objective.'''
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    improvement = mu - y_best - xi if maximize else y_best - mu - xi

    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(sigma > 0, improvement / sigma, 0.0)
        pdf = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * z ** 2)
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))

    ei = np.where(
        sigma > 0,
        improvement * cdf + sigma * pdf,
        np.maximum(improvement, 0.0),
    )
    return ei


def select_feasible_grid_params(
    frame: pd.DataFrame,
    param_cols: list[str],
    *,
    max_candidates: int = 200_000,
) -> tuple[list[str], int]:
    '''Greedy subset of hyperparameters whose observed grid fits *max_candidates*.'''
    if not param_cols:
        return [], 0

    ranked = sorted(
        param_cols,
        key=lambda col: len(frame[col].dropna().unique()),
    )
    selected: list[str] = []
    product = 1

    for col in ranked:
        n_levels = len(frame[col].dropna().unique())
        if n_levels <= 1:
            continue
        if selected and product * n_levels > max_candidates:
            break
        if not selected and n_levels > max_candidates:
            return [col], n_levels
        selected.append(col)
        product *= n_levels

    return selected, product


def _expand_partial_candidates(
    candidates: pd.DataFrame,
    frame: pd.DataFrame,
    grid_param_cols: list[str],
    all_param_cols: list[str],
) -> pd.DataFrame:
    '''Fill non-grid hyperparameters with the most common observed training value.'''
    if set(grid_param_cols) == set(all_param_cols):
        return candidates

    defaults: dict[str, Any] = {}
    for col in all_param_cols:
        if col in grid_param_cols:
            continue
        mode = frame[col].mode(dropna=True)
        defaults[col] = mode.iloc[0] if not mode.empty else frame[col].dropna().iloc[0]

    expanded = candidates.copy()
    for col, value in defaults.items():
        expanded[col] = value
    return expanded


def enumerate_observed_grid(
    frame: pd.DataFrame,
    param_cols: list[str],
    *,
    max_candidates: int = 200_000,
    drop_observed: bool = True,
    prefix: str = 'config__',
) -> pd.DataFrame:
    '''Cartesian product over observed levels of each hyperparameter.'''
    if not param_cols:
        return pd.DataFrame()

    level_lists: list[list[Any]] = []
    for col in param_cols:
        levels = list(frame[col].dropna().unique())
        if not levels:
            raise ValueError(f'No observed levels for hyperparameter {col!r}')
        level_lists.append(levels)

    n_candidates = math.prod(len(levels) for levels in level_lists)
    if n_candidates > max_candidates:
        raise ValueError(
            f'Cartesian product has {n_candidates:,} candidates, exceeding '
            f'max_candidates={max_candidates:,}. Reduce varying hyperparameters '
            'or increase max_candidates.',
        )

    rows: list[dict[str, Any]] = []
    short_names = [_short_param_name(col, prefix=prefix) for col in param_cols]
    for combo in itertools.product(*level_lists):
        row = {col: val for col, val in zip(param_cols, combo, strict=True)}
        row.update({
            short: val
            for short, val in zip(short_names, combo, strict=True)
        })
        rows.append(row)

    candidates = pd.DataFrame(rows)
    if not drop_observed or candidates.empty:
        return candidates.reset_index(drop=True)

    observed_keys = {
        tuple(row[col] for col in param_cols)
        for _, row in frame[param_cols].iterrows()
    }
    key_series = candidates.apply(
        lambda row: tuple(row[col] for col in param_cols),
        axis=1,
    )
    unseen = ~key_series.isin(observed_keys)
    return candidates.loc[unseen].reset_index(drop=True)


def rank_candidates_by_ei(
    result: GPSurrogateResult,
    candidates: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    xi: float = 0.01,
    top_k: int | None = 20,
    prefix: str = 'config__',
) -> pd.DataFrame:
    '''Rank candidate configs by Expected Improvement against the best observed run.'''
    if candidates.empty:
        return pd.DataFrame(columns=['ei', 'mu', 'sigma', 'mu_minus_best'])

    X_raw = _encode_candidates_aligned(
        frame,
        candidates,
        result.param_cols,
        result.design_columns,
        fill_values=result.feature_fill_values,
        prefix=prefix,
    ).astype(float).values
    X_scaled = result.scaler_X.transform(X_raw)

    mu_std, sigma_std = result.gp.predict(X_scaled, return_std=True)
    mu = result.scaler_y.inverse_transform(mu_std.reshape(-1, 1)).ravel()
    sigma = sigma_std * float(result.scaler_y.scale_[0])
    y_best = float(result.y_train.max())

    ei = expected_improvement(mu, sigma, y_best, xi=xi, maximize=True)
    short_names = [_short_param_name(col, prefix=prefix) for col in result.param_cols]

    ranked = candidates.copy()
    ranked['ei'] = ei
    ranked['mu'] = mu
    ranked['sigma'] = sigma
    ranked['mu_minus_best'] = mu - y_best
    ranked = ranked.sort_values('ei', ascending=False).reset_index(drop=True)

    display_cols = ['ei', 'mu', 'sigma', 'mu_minus_best', *short_names]
    display_cols = [c for c in display_cols if c in ranked.columns]
    ranked = ranked[display_cols]

    if top_k is not None:
        ranked = ranked.head(top_k).reset_index(drop=True)
    return ranked


def propose_next_configs(
    frame: pd.DataFrame,
    param_cols: list[str],
    *,
    target: str = 'train__mean_best_val_r2',
    xi: float = 0.01,
    max_candidates: int = 200_000,
    grid_param_cols: list[str] | None = None,
    prefix: str = 'config__',
    **gp_kwargs: Any,
) -> tuple[GPSurrogateResult, pd.DataFrame]:
    '''Fit GP, enumerate unseen grid points, and rank by Expected Improvement.'''
    gp_result = fit_gp_surrogate(
        frame,
        param_cols,
        target=target,
        prefix=prefix,
        **gp_kwargs,
    )

    if grid_param_cols is None:
        grid_param_cols, grid_size = select_feasible_grid_params(
            frame,
            param_cols,
            max_candidates=max_candidates,
        )
    else:
        grid_size = math.prod(
            len(frame[col].dropna().unique()) for col in grid_param_cols
        )

    if not grid_param_cols:
        return gp_result, pd.DataFrame(columns=['ei', 'mu', 'sigma', 'mu_minus_best'])

    candidates = enumerate_observed_grid(
        frame,
        grid_param_cols,
        max_candidates=max_candidates,
        prefix=prefix,
    )
    candidates = _expand_partial_candidates(
        candidates,
        frame,
        grid_param_cols,
        param_cols,
    )
    ranking = rank_candidates_by_ei(
        gp_result,
        candidates,
        frame,
        xi=xi,
        top_k=None,
        prefix=prefix,
    )
    ranking.attrs['grid_param_cols'] = grid_param_cols
    ranking.attrs['grid_size'] = grid_size
    return gp_result, ranking


def plot_gp_diagnostics(
    result: GPSurrogateResult,
    *,
    ax=None,
):
    '''Observed vs leave-one-out GP predictions with uncertainty bars.'''
    import matplotlib.pyplot as plt

    observed = result.y_train
    predicted = result.loo_predictions
    std = result.loo_std

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.errorbar(
        observed,
        predicted,
        yerr=2.0 * std,
        fmt='o',
        ms=8,
        capsize=4,
        elinewidth=1.2,
        color='steelblue',
        ecolor='gray',
        markeredgecolor='k',
        markeredgewidth=0.5,
    )
    lims = [
        min(observed.min(), predicted.min()) - 0.02,
        max(observed.max(), predicted.max()) + 0.02,
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f'Observed {result.target}')
    ax.set_ylabel('LOO predicted')
    ax.set_title(f'GP surrogate (LOO R²={result.loo_r2:.3f})')
    ax.set_aspect('equal', adjustable='box')
    return ax


def plot_ei_marginal(
    ranking: pd.DataFrame,
    param_cols: list[str],
    *,
    top_n_params: int | None = None,
    prefix: str = 'config__',
):
    '''Boxplot of EI grouped by hyperparameter level across ranked candidates.'''
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ranking.empty:
        raise ValueError('ranking is empty; nothing to plot')

    short_names = [_short_param_name(col, prefix=prefix) for col in param_cols]
    plot_params = short_names
    if top_n_params is not None:
        plot_params = plot_params[:top_n_params]

    n_params = len(plot_params)
    ncols = min(3, n_params)
    nrows = math.ceil(n_params / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    for idx, short in enumerate(plot_params):
        ax = axes[idx // ncols][idx % ncols]
        if short not in ranking.columns:
            ax.set_visible(False)
            continue
        plot_df = ranking[[short, 'ei']].copy()
        plot_df[short] = plot_df[short].astype(str)
        sns.boxplot(data=plot_df, x=short, y='ei', ax=ax, color='steelblue')
        ax.set_xlabel(short)
        ax.set_ylabel('Expected Improvement')
        ax.set_title(short)
        ax.tick_params(axis='x', rotation=45)

    for idx in range(n_params, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('EI by hyperparameter level (candidate pool)', y=1.02)
    fig.tight_layout()
    return fig, axes


def param_specs_table(encoded: EncodedConfig) -> pd.DataFrame:
    '''Convenience wrapper around ``param_specs_to_search_space``.'''
    return param_specs_to_search_space(encoded.param_specs)
