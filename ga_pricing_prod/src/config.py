from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class GAConfig:
    input_csv: Path = Path('data/input/retail_price.csv')
    output_csv: Path = Path('data/output/retail_price_suggested_prices.csv')
    summary_json: Path = Path('data/output/optimization_summary.json')
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

    def validate(self) -> None:
        if self.pop_size < 4:
            raise ValueError('pop_size must be >= 4')
        if self.generations < 1:
            raise ValueError('generations must be >= 1')
        if self.tournament_k < 2 or self.tournament_k > self.pop_size:
            raise ValueError('tournament_k must be in [2, pop_size]')
        if not (0.0 <= self.cxpb <= 1.0):
            raise ValueError('cxpb must be in [0, 1]')
        if not (0.0 <= self.mutpb <= 1.0):
            raise ValueError('mutpb must be in [0, 1]')
        if self.mut_std < 0.0:
            raise ValueError('mut_std must be >= 0')
        if self.mult_min <= 0 or self.mult_max <= 0 or self.mult_min >= self.mult_max:
            raise ValueError('Expected 0 < mult_min < mult_max')
        if self.elite_count < 0 or self.elite_count >= self.pop_size:
            raise ValueError('elite_count must be in [0, pop_size)')
