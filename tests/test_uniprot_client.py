"""Tests for :mod:`lincs_gsnn.proc.uniprot_client` (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from lincs_gsnn.proc.uniprot_client import idmapping_batch, mapping_to_dict


@patch("lincs_gsnn.proc.uniprot_client._SESSION.post")
@patch("lincs_gsnn.proc.uniprot_client._SESSION.get")
def test_idmapping_batch(mock_get, mock_post):
    mock_post.return_value = MagicMock(status_code=200, json=lambda: {"jobId": "job1"})
    mock_get.side_effect = [
        MagicMock(status_code=200, json=lambda: {"jobStatus": "FINISHED"}),
        MagicMock(
            status_code=200,
            json=lambda: {
                "results": [
                    {"from": "TP53", "to": "P04637"},
                    {"from": "MAP2K1", "to": "Q02750"},
                ]
            },
        ),
    ]

    df = idmapping_batch(["TP53", "MAP2K1"], from_db="Gene_Name")
    assert len(df) == 2
    m = mapping_to_dict(df)
    assert m["TP53"] == ["P04637"]
    assert m["MAP2K1"] == ["Q02750"]
