from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import GAConfig


@dataclass(slots=True)
class OptimizationSummary:
    best_multiplier: float
    best_profit: float
    baseline_profit: float
    profit_gain: float
    elasticity: float
    pop_size: int
    generations: int
    top_20: list[dict[str, float]]
    history: list[dict[str, float | int]]


@dataclass(slots=True)
class OptimizationArtifacts:
    result_df: pd.DataFrame
    summary: OptimizationSummary


def evaluate_multiplier(df: pd.DataFrame, multiplier: float, elasticity: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    p_old = df['unit_price'].to_numpy(dtype=float)
    q_old = df['qty'].to_numpy(dtype=float)
    freight = df['freight_price'].to_numpy(dtype=float)

    p_new = p_old * multiplier

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ratio = np.where(p_old == 0, 1.0, p_new / p_old)
        q_new = q_old * np.power(ratio, elasticity)

    q_new = np.where(np.isfinite(q_new), q_new, 0.0)
    q_new = np.maximum(q_new, 0.0)

    profit_per_row = (p_new - freight) * q_new
    profit_per_row = np.where(np.isfinite(profit_per_row), profit_per_row, 0.0)
    total_profit = float(np.sum(profit_per_row))

    return total_profit, p_new, q_new, profit_per_row


class PricingOptimizer:
    def __init__(self, config: GAConfig) -> None:
        config.validate()
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

    def _make_individual(self) -> float:
        return random.uniform(self.config.mult_min, self.config.mult_max)

    def _mutate(self, ind: float) -> float:
        if random.random() < self.config.mutpb:
            ind += random.gauss(0, self.config.mut_std)
        return max(self.config.mult_min, min(self.config.mult_max, ind))

    def _crossover(self, a: float, b: float) -> tuple[float, float]:
        if random.random() < self.config.cxpb:
            alpha = random.random()
            return alpha * a + (1 - alpha) * b, alpha * b + (1 - alpha) * a
        return a, b

    def _tournament_selection(self, population: list[float], fitnesses: list[float]) -> list[float]:
        selected: list[float] = []
        n = len(population)
        for _ in range(n):
            aspirants = random.sample(range(n), self.config.tournament_k)
            best_idx = max(aspirants, key=lambda i: fitnesses[i])
            selected.append(population[best_idx])
        return selected

    def optimize(self, df: pd.DataFrame) -> OptimizationArtifacts:
        baseline_profit, _, _, _ = evaluate_multiplier(df, 1.0, self.config.elasticity)

        population = [self._make_individual() for _ in range(self.config.pop_size)]
        history: list[dict[str, float | int]] = []

        for gen in range(1, self.config.generations + 1):
            fitnesses = [evaluate_multiplier(df, ind, self.config.elasticity)[0] for ind in population]
            best_idx = int(np.argmax(fitnesses))
            best_mult = float(population[best_idx])
            best_profit = float(fitnesses[best_idx])
            history.append({'generation': gen, 'best_multiplier': best_mult, 'best_profit': best_profit})

            selected = self._tournament_selection(population, fitnesses)
            next_pop: list[float] = []

            sorted_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
            for i in range(self.config.elite_count):
                next_pop.append(population[sorted_idx[i]])

            while len(next_pop) < self.config.pop_size:
                a = random.choice(selected)
                b = random.choice(selected)
                c1, c2 = self._crossover(a, b)
                next_pop.append(self._mutate(c1))
                if len(next_pop) < self.config.pop_size:
                    next_pop.append(self._mutate(c2))

            population = next_pop

        final_fitnesses = [evaluate_multiplier(df, ind, self.config.elasticity)[0] for ind in population]
        pairs = sorted(zip(population, final_fitnesses), key=lambda x: x[1], reverse=True)

        best_multiplier, best_profit = float(pairs[0][0]), float(pairs[0][1])
        _, p_new_arr, q_new_arr, profit_rows = evaluate_multiplier(df, best_multiplier, self.config.elasticity)

        result_df = df.copy()
        result_df['suggested_unit_price'] = p_new_arr
        result_df['suggested_qty'] = q_new_arr
        result_df['suggested_row_profit'] = profit_rows

        top_20 = [
            {'rank': idx + 1, 'multiplier': float(mult), 'profit': float(profit)}
            for idx, (mult, profit) in enumerate(pairs[:20])
        ]
        summary = OptimizationSummary(
            best_multiplier=best_multiplier,
            best_profit=best_profit,
            baseline_profit=float(baseline_profit),
            profit_gain=float(best_profit - baseline_profit),
            elasticity=float(self.config.elasticity),
            pop_size=int(self.config.pop_size),
            generations=int(self.config.generations),
            top_20=top_20,
            history=history,
        )
        return OptimizationArtifacts(result_df=result_df, summary=summary)


def save_outputs(artifacts: OptimizationArtifacts, output_csv: str | Path, summary_json: str | Path) -> None:
    output_csv = Path(output_csv)
    summary_json = Path(summary_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    artifacts.result_df.to_csv(output_csv, index=False)
    summary_json.write_text(json.dumps(asdict(artifacts.summary), ensure_ascii=False, indent=2), encoding='utf-8')
