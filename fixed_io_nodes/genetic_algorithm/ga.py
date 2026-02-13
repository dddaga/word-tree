"""
Genetic algorithm tuner: generate individual, crossover, mutate, select, run.
"""

import json
import multiprocessing as mp
import random
import signal
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List

import torch
from tqdm.auto import tqdm

from .config_resolver import resolve_config
from .fitness import evaluate_fitness
from .gene_expression import build_suppress_predicate


def _fitness_worker(run_dir, resolved_config, run_name, get_train_val_datasets, model_class, result_queue):
    """Run in subprocess so main can respond to Ctrl+C by terminating this process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    try:
        acc = evaluate_fitness(
            run_dir, resolved_config, get_train_val_datasets, model_class=model_class, run_name=run_name
        )
        result_queue.put(("ok", acc))
    except Exception as e:
        result_queue.put(("err", str(e)))


def _uniform_crossover(parent1: Dict[str, Any], parent2: Dict[str, Any], search_space: Dict[str, list]):
    """Uniform crossover: each gene from parent1 or parent2 with 50% chance."""
    offspring = {}
    for key in search_space:
        offspring[key] = parent1[key] if random.random() < 0.5 else parent2[key]
    return offspring


def _mutate(individual: Dict[str, Any], search_space: Dict[str, list], mutation_rate: float):
    """With probability mutation_rate per gene, replace with random value from search space."""
    out = dict(individual)
    for key, choices in search_space.items():
        if random.random() < mutation_rate:
            out[key] = random.choice(choices)
    return out


class GeneticTuner:
    def __init__(
        self,
        base_config: Dict[str, Any],
        search_space: Dict[str, list],
        model_class,
        generations: int = 10,
        population_size: int = 50,
        elite_frac: float = 0.5,
        crossover_rate: float = 0.3,
        mutation_rate: float = 0.2,
        top_k: int = 5,
    ):
        self.base_config = base_config
        gene_expression_raw = self.base_config.pop("gene_expression", {})
        self._is_suppressed = build_suppress_predicate(gene_expression_raw)
        self.search_space = search_space
        self.model_class = model_class
        self.generations = generations
        self.population_size = population_size
        self.elite_frac = elite_frac
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.top_k = top_k

    def generate_individual(self) -> Dict[str, Any]:
        """Random individual: one value per search-space key."""
        return {key: random.choice(choices) for key, choices in self.search_space.items()}

    def evaluate_fitness(
        self,
        individual: Dict[str, Any],
        run_dir: str,
        get_train_val_datasets,
        run_name: str = "ga",
        use_subprocess: bool = True,
    ) -> float:
        """Resolve individual to config, then run fitness in this process or a subprocess (subprocess allows Ctrl+C to stop)."""
        resolved = resolve_config(individual, self.base_config)
        if not use_subprocess:
            return evaluate_fitness(
                run_dir, resolved, get_train_val_datasets, model_class=self.model_class, run_name=run_name
            )
        result_queue = mp.Queue()
        p = mp.Process(
            target=_fitness_worker,
            args=(run_dir, resolved, run_name, get_train_val_datasets, self.model_class, result_queue),
            daemon=False,
        )
        p.start()
        try:
            while True:
                try:
                    status, value = result_queue.get(timeout=1.0)
                    p.join(timeout=2)
                    if status == "ok":
                        return float(value)
                    raise RuntimeError(value)
                except Empty:
                    if not p.is_alive():
                        raise RuntimeError("Fitness process exited without result")
        except KeyboardInterrupt:
            p.terminate()
            p.join(timeout=2)
            raise

    def select_top_k(self, population: List[Dict], fitness_scores: List[float], k: int) -> List[Dict]:
        """Return top-k individuals by fitness (descending)."""
        paired = list(zip(population, fitness_scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in paired[:k]]

    def run(
        self,
        run_dir: str,
        get_train_val_datasets,
        run_name: str = "ga",
    ) -> List[Dict[str, Any]]:
        """
        Initialize population, run generations (evaluate, select, crossover, mutate), return top_k.
        Optionally save best_configs.json under run_dir.
        """
        _MAX_RETRIES = 1000
        population = []
        for _ in range(self.population_size):
            for _ in range(_MAX_RETRIES):
                ind = self.generate_individual()
                if not self._is_suppressed(ind):
                    population.append(ind)
                    break
            else:
                raise RuntimeError(
                    "Could not generate valid individual after %d retries; "
                    "check gene_expression and search_space." % _MAX_RETRIES
                )
        elite_count = max(1, int(self.population_size * self.elite_frac))
        all_time_best = []
        interrupted = False

        # Run fitness in main process when CUDA requested so we use GPU; subprocess can cause device mismatch.
        use_subprocess = not (
            self.base_config.get("system", {}).get("device") == "cuda" and torch.cuda.is_available()
        )
        try:
            gen_range = tqdm(range(self.generations), desc="Generation", unit="gen")
            for gen in gen_range:
                fitness_scores = []
                for ind in tqdm(population, desc=f"Gen {gen + 1} eval", leave=False, unit="ind"):
                    f = self.evaluate_fitness(
                        ind, run_dir, get_train_val_datasets, run_name=run_name, use_subprocess=use_subprocess
                    )
                    fitness_scores.append(f)
                best_idx = max(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
                best_f = fitness_scores[best_idx]
                avg_f = sum(fitness_scores) / len(fitness_scores)
                best_ind = dict(population[best_idx])
                best_ind["fitness"] = best_f
                best_ind["generation"] = gen + 1
                all_time_best.append(best_ind)
                gen_range.set_postfix(best=f"{best_f:.4f}", avg=f"{avg_f:.4f}")

                if gen < self.generations - 1:
                    elites = self.select_top_k(population, fitness_scores, elite_count)
                    new_pop = list(elites)
                    while len(new_pop) < self.population_size:
                        for _ in range(_MAX_RETRIES):
                            p1, p2 = random.choices(elites, k=2)
                            if random.random() < self.crossover_rate:
                                child = _uniform_crossover(p1, p2, self.search_space)
                            else:
                                child = dict(p1)
                            child = _mutate(child, self.search_space, self.mutation_rate)
                            if not self._is_suppressed(child):
                                new_pop.append(child)
                                break
                        else:
                            raise RuntimeError(
                                "Could not generate valid offspring after %d retries; "
                                "check gene_expression and search_space." % _MAX_RETRIES
                            )
                    population = new_pop[: self.population_size]
        except KeyboardInterrupt:
            interrupted = True
            if all_time_best:
                tqdm.write("KeyboardInterrupt: saving partial results and exiting.")

        top = self.select_top_k(
            all_time_best,
            [b["fitness"] for b in all_time_best],
            min(self.top_k, len(all_time_best)) if all_time_best else 0,
        )
        if top:
            best_path = Path(run_dir) / "best_configs.json"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            with open(best_path, "w") as f:
                json.dump(top, f, indent=2)
        return top
