# tests_selector.py

import os
import unittest
import random
import logging
import numpy as np
from typing import List, Dict, Any

from selecter import Selector, SelectorConfig, calculate_architecture_diversity, SelectorError
from initializer import generate_population

# Создаём директорию для логов, если её ещё нет
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "tests_selecter.log")

# Очищаем старые хендлеры у корневого логгера
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Настраиваем логирование: один файл + вывод в консоль
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

logger = logging.getLogger(__name__)


class TestSelector(unittest.TestCase):
    def setUp(self):
        self.selector = Selector()
        self.population = generate_population(size=20)
        self.fitnesses = [random.uniform(0.1, 0.9) for _ in range(len(self.population))]
        self.valid_methods = [
            'tournament', 'roulette', 'rank', 'elitist', 'diversity'
        ]
        
    def _log_selection(self, method: str, selected: List[Any]):
        """Логирует результаты селекции"""
        logger.info(f"--- {method} селекция: выбрано {len(selected)} особей")
        for i, arch in enumerate(selected[:3]):  # Логируем первые 3
            logger.debug(f"  Архитектура {i+1}: {type(arch).__name__}")

    def test_tournament_selection(self):
        """Тестирование турнирной селекции"""
        selected = self.selector.tournament_selection(
            self.population, self.fitnesses, 10, minimize=True
        )
        self._log_selection("Турнирная", selected)
        self.assertEqual(len(selected), 10)
        self.assertTrue(all(arch in self.population for arch in selected))

    def test_roulette_wheel_selection(self):
        """Тестирование рулеточной селекции"""
        selected = self.selector.roulette_wheel_selection(
            self.population, self.fitnesses, 8, minimize=True
        )
        self._log_selection("Рулеточная", selected)
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(arch in self.population for arch in selected))

    def test_rank_selection(self):
        """Тестирование ранговой селекции"""
        selected = self.selector.rank_selection(
            self.population, self.fitnesses, 12, minimize=True
        )
        self._log_selection("Ранговая", selected)
        self.assertEqual(len(selected), 12)
        self.assertTrue(all(arch in self.population for arch in selected))

    def test_elitist_selection(self):
        """Тестирование элитарной селекции"""
        selected = self.selector.elitist_selection(
            self.population, self.fitnesses, 15, minimize=True
        )
        self._log_selection("Элитарная", selected)
        self.assertEqual(len(selected), 15)
        
        # Проверяем что лучшие особи присутствуют
        best_indices = np.argsort(self.fitnesses)[:3]
        best_archs = [self.population[i] for i in best_indices]
        self.assertTrue(all(arch in selected for arch in best_archs))

    def test_diversity_preserving_selection(self):
        """Тестирование селекции с сохранением разнообразия"""
        selected = self.selector.diversity_preserving_selection(
            self.population, self.fitnesses, 10, minimize=True
        )
        self._log_selection("Сохранение разнообразия", selected)
        self.assertEqual(len(selected), 10)
        
        # Проверяем что есть разнообразие
        diversity = calculate_architecture_diversity(selected)
        logger.info(f"Разнообразие выборки: min={min(diversity):.3f}, max={max(diversity):.3f}, avg={np.mean(diversity):.3f}")
        self.assertGreater(np.mean(diversity), 0.3)

    def test_invalid_method(self):
        """Тестирование вызова несуществующего метода"""
        with self.assertRaises(SelectorError):
            self.selector.select(
                self.population, self.fitnesses, "invalid_method", 5
            )

    def test_small_population(self):
        """Тестирование с маленькой популяцией"""
        small_pop = self.population[:3]
        small_fitness = self.fitnesses[:3]
        
        with self.assertRaises(SelectorError):
            self.selector.select(small_pop, small_fitness, "tournament", 2)

    def test_invalid_fitness_size(self):
        """Тестирование с несовпадающими размерами популяции и фитнеса"""
        with self.assertRaises(SelectorError):
            self.selector.select(
                self.population, self.fitnesses[:-1], "tournament", 5
            )

    def test_invalid_fitness_values(self):
        """Тестирование с некорректными значениями фитнеса"""
        invalid_fitness = self.fitnesses.copy()
        invalid_fitness[0] = float('inf')
        
        with self.assertRaises(SelectorError):
            self.selector.select(
                self.population, invalid_fitness, "tournament", 5
            )

    def test_select_methods(self):
        """Комплексное тестирование всех методов через основной интерфейс"""
        for method in self.valid_methods:
            with self.subTest(method=method):
                selected = self.selector.select(
                    self.population, self.fitnesses, method, 10, True
                )
                self._log_selection(method, selected)
                self.assertEqual(len(selected), 10)
                self.assertTrue(all(arch in self.population for arch in selected))

    def test_diversity_calculation(self):
        """Тестирование расчета разнообразия архитектур"""
        diversity = calculate_architecture_diversity(self.population)
        self.assertEqual(len(diversity), len(self.population))
        self.assertTrue(all(d >= 0 for d in diversity))
        
        logger.info(f"Разнообразие популяции: min={min(diversity):.3f}, max={max(diversity):.3f}, avg={np.mean(diversity):.3f}")


if __name__ == "__main__":
    unittest.main()