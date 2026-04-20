from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.config import GAConfig
from src.io_utils import load_retail_data
from src.pricing_optimizer import PricingOptimizer, save_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Optimize retail prices with a genetic algorithm.')
    parser.add_argument('--input-csv', default='data/input/retail_price.csv')
    parser.add_argument('--output-csv', default='data/output/retail_price_suggested_prices.csv')
    parser.add_argument('--summary-json', default='data/output/optimization_summary.json')
    parser.add_argument('--pop-size', type=int, default=200)
    parser.add_argument('--generations', type=int, default=80)
    parser.add_argument('--tournament-k', type=int, default=3)
    parser.add_argument('--cxpb', type=float, default=0.6)
    parser.add_argument('--mutpb', type=float, default=0.3)
    parser.add_argument('--mut-std', type=float, default=0.1)
    parser.add_argument('--mult-min', type=float, default=0.5)
    parser.add_argument('--mult-max', type=float, default=2.0)
    parser.add_argument('--elasticity', type=float, default=-1.0)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--elite-count', type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GAConfig(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        summary_json=Path(args.summary_json),
        pop_size=args.pop_size,
        generations=args.generations,
        tournament_k=args.tournament_k,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        mut_std=args.mut_std,
        mult_min=args.mult_min,
        mult_max=args.mult_max,
        elasticity=args.elasticity,
        random_seed=args.random_seed,
        elite_count=args.elite_count,
    )

    df, mapping = load_retail_data(config.input_csv)
    optimizer = PricingOptimizer(config)
    artifacts = optimizer.optimize(df)
    save_outputs(artifacts, config.output_csv, config.summary_json)

    response = {
        'column_mapping': mapping,
        'rows_used': len(df),
        'summary': asdict(artifacts.summary),
        'output_csv': str(config.output_csv),
        'summary_json': str(config.summary_json),
    }
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
