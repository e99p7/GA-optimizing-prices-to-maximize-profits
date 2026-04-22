from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import GAConfig
from src.io_utils import load_retail_data
from src.pricing_optimizer import PricingOptimizer, save_outputs


class OptimizePathRequest(BaseModel):
    input_csv: str = Field(default='data/input/retail_price.csv')
    output_csv: str = Field(default='data/output/retail_price_suggested_prices.csv')
    summary_json: str = Field(default='data/output/optimization_summary.json')
    pop_size: int = 200
    generations: int = 80
    tournament_k: int = 3
    cxpb: float = 0.6
    mutpb: float = 0.3
    mut_std: float = 0.1
    mult_min: float = 0.5
    mult_max: float = 2.0
    elasticity: float = -1.0
    random_seed: int = 42
    elite_count: int = 2


class OptimizeResponse(BaseModel):
    rows_used: int
    best_multiplier: float
    best_profit: float
    baseline_profit: float
    profit_gain: float
    output_csv: str
    summary_json: str


app = FastAPI(
    title='GA Pricing Optimization API',
    version='1.0.0',
    description='Optimize retail prices with a lightweight genetic algorithm service.',
)


def build_config(payload: OptimizePathRequest) -> GAConfig:
    return GAConfig(
        input_csv=Path(payload.input_csv),
        output_csv=Path(payload.output_csv),
        summary_json=Path(payload.summary_json),
        pop_size=payload.pop_size,
        generations=payload.generations,
        tournament_k=payload.tournament_k,
        cxpb=payload.cxpb,
        mutpb=payload.mutpb,
        mut_std=payload.mut_std,
        mult_min=payload.mult_min,
        mult_max=payload.mult_max,
        elasticity=payload.elasticity,
        random_seed=payload.random_seed,
        elite_count=payload.elite_count,
    )


@app.get('/health')
def health() -> dict:
    return {'status': 'ok', 'service': 'ga-pricing-optimization'}


@app.post('/optimize/path', response_model=OptimizeResponse)
def optimize_path(request: OptimizePathRequest) -> OptimizeResponse:
    try:
        config = build_config(request)
        df, _ = load_retail_data(config.input_csv)
        artifacts = PricingOptimizer(config).optimize(df)
        save_outputs(artifacts, config.output_csv, config.summary_json)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary = artifacts.summary
    return OptimizeResponse(
        rows_used=len(df),
        best_multiplier=summary.best_multiplier,
        best_profit=summary.best_profit,
        baseline_profit=summary.baseline_profit,
        profit_gain=summary.profit_gain,
        output_csv=str(config.output_csv),
        summary_json=str(config.summary_json),
    )


@app.post('/optimize/upload')
async def optimize_upload(
    file: UploadFile = File(...),
    elasticity: float = -1.0,
    pop_size: int = 200,
    generations: int = 80,
) -> dict:
    suffix = Path(file.filename or 'input.csv').suffix or '.csv'
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f'uploaded{suffix}'
        content = await file.read()
        tmp_path.write_bytes(content)

        output_csv = Path(tmp_dir) / 'retail_price_suggested_prices.csv'
        summary_json = Path(tmp_dir) / 'optimization_summary.json'

        config = GAConfig(
            input_csv=tmp_path,
            output_csv=output_csv,
            summary_json=summary_json,
            elasticity=elasticity,
            pop_size=pop_size,
            generations=generations,
        )

        try:
            df, mapping = load_retail_data(config.input_csv)
            artifacts = PricingOptimizer(config).optimize(df)
            save_outputs(artifacts, config.output_csv, config.summary_json)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            'rows_used': len(df),
            'column_mapping': mapping,
            'summary': asdict(artifacts.summary),
            'preview': artifacts.result_df.head(10).to_dict(orient='records'),
        }
