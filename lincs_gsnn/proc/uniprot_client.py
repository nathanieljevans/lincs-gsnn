"""Batch UniProt REST client for landmark gene ID mapping.

Uses the UniProt ID mapping service (``/idmapping/run``) and optional
``/uniprotkb/accessions`` field retrieval. Replaces per-gene PyPath lookups
for building the canonical landmark table.
"""

from __future__ import annotations

import io
import time
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

try:  # urllib3 ships with requests; import path is stable but guard just in case.
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover - extremely old urllib3
    from requests.packages.urllib3.util.retry import Retry  # type: ignore

_UNIPROT_API = "https://rest.uniprot.org"
_HUMAN_TAXON = "9606"
_DEFAULT_CHUNK = 200
_POLL_INTERVAL_S = 1.0
_POLL_TIMEOUT_S = 300.0
_FINISHED_STATUSES = frozenset({None, "FINISHED", "COMPLETE", "SUCCESS"})

# (connect, read) timeouts. A short connect timeout fails fast on an
# unreachable host so the retry/backoff logic can kick in promptly.
_CONNECT_TIMEOUT_S = 15.0
_READ_TIMEOUT_S = 120.0
_DEFAULT_TIMEOUT = (_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S)

# Network-level errors worth catching when degrading gracefully. ``OSError``
# covers the bare ``TimeoutError: timed out`` raised from socket connect.
NetworkError = (requests.exceptions.RequestException, OSError)


def _build_session(
    total_retries: int = 5,
    backoff_factor: float = 1.0,
) -> requests.Session:
    """requests session that retries connect/read/5xx errors with backoff.

    backoff sleeps are ``backoff_factor * (2 ** (attempt - 1))`` seconds, i.e.
    ~1s, 2s, 4s, 8s, 16s for the default factor — so transient blips and brief
    UniProt rate-limits recover without aborting the whole bionetwork build.
    """
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _build_session()


def _post_idmapping_run(
    ids: Sequence[str],
    from_db: str,
    to_db: str = "UniProtKB",
    organism: str = _HUMAN_TAXON,
) -> str:
    resp = _SESSION.post(
        f"{_UNIPROT_API}/idmapping/run",
        data={
            "from": from_db,
            "to": to_db,
            "ids": ",".join(ids),
            "organism": organism,
        },
        timeout=_DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    job_id = resp.json().get("jobId")
    if not job_id:
        raise RuntimeError(f"UniProt idmapping: no jobId in response: {resp.text[:200]}")
    return job_id


def _poll_idmapping_job(job_id: str) -> None:
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    status_url = f"{_UNIPROT_API}/idmapping/status/{job_id}"
    while time.monotonic() < deadline:
        resp = _SESSION.get(status_url, timeout=_DEFAULT_TIMEOUT)
        resp.raise_for_status()
        status = resp.json().get("jobStatus")
        if status in _FINISHED_STATUSES:
            return
        if status == "RUNNING":
            time.sleep(_POLL_INTERVAL_S)
            continue
        if status in ("ERROR", "FAILED"):
            raise RuntimeError(f"UniProt idmapping job {job_id} ended with status={status!r}")
        # Unknown terminal state: try fetching results below.
        return
    raise TimeoutError(f"UniProt idmapping job {job_id} did not finish within {_POLL_TIMEOUT_S}s")


def _fetch_idmapping_results(job_id: str) -> pd.DataFrame:
    resp = _SESSION.get(
        f"{_UNIPROT_API}/idmapping/results/{job_id}",
        params={"format": "json"},
        timeout=_DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    results = payload.get("results", [])
    if not results:
        return pd.DataFrame(columns=["from", "to"])
    rows = [{"from": r.get("from"), "to": r.get("to")} for r in results]
    return pd.DataFrame(rows)


def idmapping_batch(
    ids: Sequence[str],
    from_db: str,
    to_db: str = "UniProtKB",
    organism: str = _HUMAN_TAXON,
    chunk_size: int = _DEFAULT_CHUNK,
    verbose: bool = False,
) -> pd.DataFrame:
    """Map *ids* to UniProt accessions via the ID mapping service.

    Parameters
    ----------
    ids
        Source identifiers (gene symbols, Ensembl IDs, etc.).
    from_db
        UniProt ``from`` database label (e.g. ``Gene_Name``, ``Ensembl``).
    chunk_size
        Maximum IDs per API job (UniProt allows large jobs; 500 is conservative).

    Returns
    -------
    DataFrame with columns ``from``, ``to`` (one row per mapping pair).
    """
    clean = [str(x).strip() for x in ids if str(x).strip() and str(x).lower() != "nan"]
    unique = list(dict.fromkeys(clean))
    if not unique:
        return pd.DataFrame(columns=["from", "to"])

    frames: List[pd.DataFrame] = []
    for start in range(0, len(unique), chunk_size):
        chunk = unique[start : start + chunk_size]
        if verbose:
            print(f"  UniProt idmapping {from_db}: {start + 1}-{start + len(chunk)} / {len(unique)}")
        job_id = _post_idmapping_run(chunk, from_db=from_db, to_db=to_db, organism=organism)
        _poll_idmapping_job(job_id)
        frames.append(_fetch_idmapping_results(job_id))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["from", "to"])
    return out.drop_duplicates()


def mapping_to_dict(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Group idmapping results: source id -> list of target accessions."""
    if df.empty:
        return {}
    out: Dict[str, List[str]] = {}
    for src, group in df.groupby("from", sort=False):
        accs = [str(x).strip() for x in group["to"].tolist() if str(x).strip()]
        out[str(src)] = list(dict.fromkeys(accs))
    return out


def fetch_gene_names_for_accessions(
    accessions: Sequence[str],
    chunk_size: int = 100,
    verbose: bool = False,
) -> Dict[str, List[str]]:
    """Return accession -> gene names (primary + synonyms) from UniProtKB."""
    accs = list(dict.fromkeys(str(a).strip() for a in accessions if str(a).strip()))
    if not accs:
        return {}

    acc_to_names: Dict[str, List[str]] = {}
    for start in range(0, len(accs), chunk_size):
        chunk = accs[start : start + chunk_size]
        if verbose:
            print(f"  UniProt accessions fields: {start + 1}-{start + len(chunk)} / {len(accs)}")
        resp = _SESSION.get(
            f"{_UNIPROT_API}/uniprotkb/accessions",
            params={
                "accessions": ",".join(chunk),
                "fields": "accession,gene_primary,gene_synonym",
            },
            headers={"Accept": "text/plain; format=tsv"},
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        table = pd.read_csv(io.StringIO(resp.text), sep="\t")
        if table.empty:
            continue
        # column names vary slightly; normalize
        cols = {c.lower(): c for c in table.columns}
        acc_col = cols.get("entry", cols.get("accession", table.columns[0]))
        primary_col = cols.get("gene names (primary)", cols.get("gene_primary", None))
        syn_col = cols.get("gene names (synonym)", cols.get("gene_synonym", None))

        for _, row in table.iterrows():
            acc = str(row[acc_col]).strip()
            names: List[str] = []
            if primary_col is not None and pd.notna(row.get(primary_col)):
                names.append(str(row[primary_col]).strip())
            if syn_col is not None and pd.notna(row.get(syn_col)):
                for part in str(row[syn_col]).replace(" ", "").split(","):
                    part = part.strip()
                    if part:
                        names.append(part)
            if names:
                acc_to_names.setdefault(acc, [])
                for n in names:
                    if n and n not in acc_to_names[acc]:
                        acc_to_names[acc].append(n)

    return acc_to_names
