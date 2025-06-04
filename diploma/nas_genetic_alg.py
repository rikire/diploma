# nas_genetic_alg.py
import os
import copy
import random
import logging
from typing import List, Dict, Tuple, Union, Any, Optional

import numpy as np
import joblib

from initializer import (
    skeleton_conv_dense, skeleton_rnn, block_randomized, dag_parallel_v2, micro_arch
)
from mutator import ArchitectureMutator
from crossover import ArchitectureCrossover, CrossoverMethod
from selecter import Selector
from evaluator import (
    parallel_evaluate_fitness_comprehensive,
    pareto_fitness_evaluation,
    FitnessMetrics
)

# Type definition
Arch = Union[List[Dict[str, Any]], Dict[str, Any]]

def wrap_arch(arch):
    """Wrap list into parallel dict for crossover"""
    if isinstance(arch, list):
        return {'parallel': {'branch_0': arch}}, True
    return arch, False

def unwrap_arch(arch, was_wrapped):
    """Unwrap architecture from parallel dict if it was wrapped"""
    if was_wrapped and isinstance(arch, dict) and 'parallel' in arch:
        return next(iter(arch['parallel'].values()))
    return arch

def make_compatible_for_crossover(p1, p2):
    """Make architectures compatible for crossover"""
    is_list1 = isinstance(p1, list)
    is_list2 = isinstance(p2, list)
    if is_list1 and is_list2:
        return p1, p2, False, False
    p1w, w1 = wrap_arch(p1)
    p2w, w2 = wrap_arch(p2)
    return p1w, p2w, w1, w2

def cached_evaluate(arch, train_data, val_data, test_data, config, weights, n_jobs=None):
    if n_jobs is None:
        n_jobs = config.get('n_jobs', 1)
    fitnesses, metrics_list = parallel_evaluate_fitness_comprehensive(
        population=[arch],
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config,
        weights=weights,
        n_jobs=n_jobs
    )
    return fitnesses[0], metrics_list[0]

class NasGeneticAlg:
    """Neural Architecture Search using Genetic Algorithm"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ga_config = config.get('ga_config', {})
        
        # Setup parameters from config
        self.population_size = self.ga_config.get('population_size', 20)
        self.generations = self.ga_config.get('n_generations', 10)
        self.mutation_rate = self.ga_config.get('mutation_rate', 0.6)
        self.crossover_rate = self.ga_config.get('crossover_rate', 0.5)
        self.crossover_method = self.ga_config.get('crossover_method', 'single_point')
        self.elitism = self.ga_config.get('elite_size', 2)
        self.selection_strategy = self.ga_config.get('selection_strategy', 'tournament')
        self.use_pareto = self.ga_config.get('use_pareto', False)
        self.n_jobs = self.ga_config.get('n_jobs', 4)
        self.selection_size = self.ga_config.get('selection_size', self.population_size)

        # Initialize history
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'diversity': []
        }
        
        
        self.logger = logging.getLogger(__name__)
        
        # Setup caching
        self.cache_dir = "cache"
        self.memory = joblib.Memory(location=self.cache_dir, verbose=0)
        
        # Initialize operators
        self.mutator = ArchitectureMutator()
        self.crossover_op = ArchitectureCrossover()
        self.selector = Selector()

    def run(
        self,
        train_data,
        val_data,
        test_data=None,
        weights: Optional[Dict] = None
    ) -> Tuple[Arch, FitnessMetrics]:
        mutator = ArchitectureMutator()
        crossover_op = ArchitectureCrossover()
        selector = Selector()

        init_strategies = [
            skeleton_conv_dense, skeleton_rnn, block_randomized, dag_parallel_v2, micro_arch
        ]
        population: List[Arch] = [random.choice(init_strategies)() for _ in range(self.population_size)]

        self.logger.info(f"GA started: pop={self.population_size}, gens={self.generations}")

        for gen in range(self.generations):
            self.logger.info(f"\n=== Generation {gen} ===")
            # Оценка всей популяции за один вызов (использует n_jobs)
            fitnesses, metrics_list = parallel_evaluate_fitness_comprehensive(
                population=population,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                config=self.config,
                weights=weights,
                n_jobs=self.n_jobs
            )
            for idx, (fit, met) in enumerate(zip(fitnesses, metrics_list)):
                self.logger.info(f"Ind {idx}: fit={fit:.4f}, val_loss={met.validation_loss:.4f}, arch={population[idx]}")
            self.logger.info(f"Best fitness: {min(fitnesses):.4f}")

            if self.use_pareto:
                pareto_res = pareto_fitness_evaluation(metrics_list)
                # Extract indices of individuals with minimal Pareto rank
                pareto_ranks = [r for r, _ in pareto_res]
                if all(isinstance(r, float) and np.isinf(r) for r in pareto_ranks):
                    pareto_front = list(range(len(population)))  # fallback: all
                else:
                    min_rank = min(pareto_ranks)
                    pareto_front = [i for i, r in enumerate(pareto_ranks) if r == min_rank]
                selected = selector.pareto_selection(
                    population, fitnesses, self.selection_size, pareto_front=pareto_front
                )
            else:
                if self.selection_strategy == 'tournament':
                    selected = selector.tournament_selection(population, fitnesses, self.selection_size, minimize=True)
                elif self.selection_strategy == 'roulette':
                    selected = selector.roulette_wheel_selection(population, fitnesses, self.selection_size, minimize=True)
                elif self.selection_strategy == 'rank':
                    selected = selector.rank_selection(population, fitnesses, self.selection_size, minimize=True)
                else:
                    raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

            elite_indices = np.argsort(fitnesses)[:self.elitism]
            elites = [copy.deepcopy(population[i]) for i in elite_indices]

            offspring = []
            random.shuffle(selected)
            for i in range(0, len(selected) - 1, 2):
                p1, p2 = selected[i], selected[i + 1]
                if random.random() < self.crossover_rate:
                    p1c, p2c, w1, w2 = make_compatible_for_crossover(p1, p2)
                    c1, c2 = crossover_op.crossover(
                        p1c, p2c,
                        method=getattr(CrossoverMethod, self.crossover_method.upper(), CrossoverMethod.SINGLE_POINT)
                    )
                    c1 = unwrap_arch(c1, w1)
                    c2 = unwrap_arch(c2, w2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                offspring.extend([c1, c2])
            if len(selected) % 2 == 1:
                offspring.append(copy.deepcopy(selected[-1]))

            new_pop = []
            for arch in offspring:
                if random.random() < self.mutation_rate:
                    try:
                        new_pop.append(mutator.mutate(arch))
                    except Exception as e:
                        self.logger.warning(f"Mutation failed: {e}")
                        new_pop.append(arch)
                else:
                    new_pop.append(arch)

            population = elites + new_pop[:self.population_size - self.elitism]

        # Финальная оценка всей популяции за один вызов (использует n_jobs)
        final_fits, final_metrics = parallel_evaluate_fitness_comprehensive(
            population=population,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            config=self.config,
            weights=weights,
            n_jobs=self.n_jobs
        )
        best_idx = int(np.argmin(final_fits))
        best_arch, best_metrics = population[best_idx], final_metrics[best_idx]

        self.logger.info(f"\nGA done. Best fitness: {final_fits[best_idx]:.4f}")
        return best_arch, best_metrics
