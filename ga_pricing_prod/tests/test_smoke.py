from __future__ import annotations

import pandas as pd

from src.config import GAConfig
from src.io_utils import load_retail_data
from src.pricing_optimizer import PricingOptimizer


def test_smoke(tmp_path):
    csv_path = tmp_path / 'retail_price.csv'
    pd.DataFrame(
        {
            'qty': [100, 120, 80, 150],
            'unit_price': [10, 20, 30, 15],
            'freight_price': [4, 8, 12, 6],
        }
    ).to_csv(csv_path, index=False)

    df, mapping = load_retail_data(csv_path)
    assert mapping['qty'] == 'qty'
    cfg = GAConfig(input_csv=csv_path, generations=5, pop_size=20)
    artifacts = PricingOptimizer(cfg).optimize(df)
    assert artifacts.summary.best_profit > 0
    assert len(artifacts.result_df) == 4
