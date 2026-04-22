from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_CANONICAL_COLUMNS = ['qty', 'unit_price', 'freight_price']

COLUMN_CANDIDATES = {
    'qty': ['qty', 'quantity', 'volume', 'units', 'amount sold', 'sold qty'],
    'unit_price': ['unit_price', 'unit price', 'price', 'retail', 'selling price'],
    'total_price': ['total_price', 'total price', 'total', 'revenue', 'gross'],
    'freight_price': ['freight_price', 'freight', 'cost', 'wholesale', 'purchase', 'purchase price', 'cogs'],
}


def safe_numeric(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().replace('\u00a0', '').replace(' ', '')
    if not text:
        return np.nan

    if ',' in text and '.' in text:
        if text.rfind(',') > text.rfind('.'):
            text = text.replace('.', '').replace(',', '.')
        else:
            text = text.replace(',', '')
    else:
        text = text.replace(',', '.')

    try:
        return float(text)
    except ValueError:
        return np.nan


def find_columns(columns: Iterable[str]) -> dict[str, str | None]:
    names = {str(col).lower().strip(): str(col) for col in columns}

    def get_like(keywords: list[str]) -> str | None:
        for kw in keywords:
            for lower_name, original_name in names.items():
                if kw in lower_name:
                    return original_name
        return None

    return {canonical: get_like(candidates) for canonical, candidates in COLUMN_CANDIDATES.items()}


def load_retail_data(csv_path: str | Path) -> tuple[pd.DataFrame, dict[str, str | None]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f'Input CSV not found: {csv_path}')

    df = pd.read_csv(csv_path)
    mapping = find_columns(df.columns)

    if not all(mapping.get(col) for col in REQUIRED_CANONICAL_COLUMNS):
        raise ValueError(
            'Could not detect required columns. '
            f'Found columns: {list(df.columns)}; detected mapping: {mapping}'
        )

    out = df.copy()
    out['qty'] = out[mapping['qty']].apply(safe_numeric)
    out['unit_price'] = out[mapping['unit_price']].apply(safe_numeric)
    if mapping.get('freight_price'):
        out['freight_price'] = out[mapping['freight_price']].apply(safe_numeric)
    else:
        out['freight_price'] = 0.0

    out = out.dropna(subset=['qty', 'unit_price', 'freight_price']).reset_index(drop=True)
    if out.empty:
        raise ValueError('After numeric cleaning the dataset is empty.')

    return out, mapping
