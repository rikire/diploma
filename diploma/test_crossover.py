# tests_crossover.py

import os
import unittest
import random
import logging
from typing import Dict, List, Any, Union

from crossover import ArchitectureCrossover, CrossoverMethod, CrossoverError
from initializer import (
    skeleton_conv_dense,
    skeleton_rnn,
    block_randomized,
    dag_parallel_v2,
    micro_arch
)

# Создаём директорию для логов, если её ещё нет
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "tests_crossover.log")

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


class TestArchitectureCrossover(unittest.TestCase):
    def setUp(self):
        self.crossover = ArchitectureCrossover()
        # Создаём пару последовательных архитектур
        self.seq1 = skeleton_conv_dense()
        self.seq2 = skeleton_rnn()
        # Создаём пару параллельных архитектур
        self.par1 = dag_parallel_v2(2)
        self.par2 = block_randomized()

    def _log_architectures(self, name: str, arch_a: Any, arch_b: Any):
        """Логируем родительские архитектуры и результат"""
        logger.info(f"--- {name}: Parent A → {arch_a}")
        logger.info(f"--- {name}: Parent B → {arch_b}")

    def test_single_point_seq(self):
        """Проверяет одноточечное скрещивание для последовательных архитектур"""
        self._log_architectures("test_single_point_seq", self.seq1, self.seq2)
        child1, child2 = self.crossover.crossover(
            self.seq1, self.seq2, method=CrossoverMethod.SINGLE_POINT
        )
        logger.info(f"=== CHILDREN: {child1} , {child2}")
        self.assertIsInstance(child1, list)
        self.assertIsInstance(child2, list)
        self.assertGreaterEqual(len(child1), self.crossover.config.min_architecture_length)
        self.assertGreaterEqual(len(child2), self.crossover.config.min_architecture_length)
        is_valid1, _ = self.crossover.validator.validate_architecture(child1)
        is_valid2, _ = self.crossover.validator.validate_architecture(child2)
        self.assertTrue(is_valid1)
        self.assertTrue(is_valid2)

    def test_two_point_seq(self):
        """Проверяет двухточечное скрещивание для последовательных архитектур"""
        self._log_architectures("test_two_point_seq", self.seq1, self.seq2)
        child1, child2 = self.crossover.crossover(
            self.seq1, self.seq2, method=CrossoverMethod.TWO_POINT
        )
        logger.info(f"=== CHILDREN: {child1} , {child2}")
        self.assertIsInstance(child1, list)
        self.assertIsInstance(child2, list)
        is_valid1, _ = self.crossover.validator.validate_architecture(child1)
        is_valid2, _ = self.crossover.validator.validate_architecture(child2)
        self.assertTrue(is_valid1)
        self.assertTrue(is_valid2)

    def test_uniform_seq(self):
        """Проверяет равномерное скрещивание для последовательных архитектур"""
        self._log_architectures("test_uniform_seq", self.seq1, self.seq2)
        child1, child2 = self.crossover.crossover(
            self.seq1, self.seq2, method=CrossoverMethod.UNIFORM
        )
        logger.info(f"=== CHILDREN: {child1} , {child2}")
        self.assertIsInstance(child1, list)
        self.assertIsInstance(child2, list)
        is_valid1, _ = self.crossover.validator.validate_architecture(child1)
        is_valid2, _ = self.crossover.validator.validate_architecture(child2)
        self.assertTrue(is_valid1)
        self.assertTrue(is_valid2)

    def test_single_point_parallel(self):
        """Проверяет одноточечное скрещивание для параллельных архитектур"""
        self._log_architectures("test_single_point_parallel", self.par1, self.par2)
        child1, child2 = self.crossover.crossover(
            self.par1, self.par2, method=CrossoverMethod.SINGLE_POINT
        )
        logger.info(f"=== CHILDREN: {child1} , {child2}")
        self.assertIsInstance(child1, dict)
        self.assertIsInstance(child2, dict)
        self.assertIn("parallel", child1)
        self.assertIn("parallel", child2)
        is_valid1, _ = self.crossover.validator.validate_architecture(child1)
        is_valid2, _ = self.crossover.validator.validate_architecture(child2)
        self.assertTrue(is_valid1)
        self.assertTrue(is_valid2)

    def test_two_point_parallel(self):
        """Проверяет двухточечное скрещивание для параллельных архитектур"""
        self._log_architectures("test_two_point_parallel", self.par1, self.par2)
        child1, child2 = self.crossover.crossover(
            self.par1, self.par2, method=CrossoverMethod.TWO_POINT
        )
        logger.info(f"=== CHILDREN: {child1} , {child2}")
        is_valid1, _ = self.crossover.validator.validate_architecture(child1)
        is_valid2, _ = self.crossover.validator.validate_architecture(child2)
        self.assertTrue(is_valid1)
        self.assertTrue(is_valid2)

    def test_uniform_parallel(self):
        """Проверяет равномерное скрещивание для параллельных архитектур"""
        self._log_architectures("test_uniform_parallel", self.par1, self.par2)
        child1, child2 = self.crossover.crossover(
            self.par1, self.par2, method=CrossoverMethod.UNIFORM
        )
        logger.info(f"=== CHILDREN: {child1} , {child2}")
        is_valid1, _ = self.crossover.validator.validate_architecture(child1)
        is_valid2, _ = self.crossover.validator.validate_architecture(child2)
        self.assertTrue(is_valid1)
        self.assertTrue(is_valid2)

    def test_invalid_method(self):
        """Проверяет передачу некорректного метода скрещивания"""
        with self.assertRaises(CrossoverError):
            self.crossover.crossover(self.seq1, self.seq2, method="nonexistent_method")

    def test_mixed_seq_and_parallel(self):
        """Проверяет сценарий, когда один родитель последовательный, другой — параллельный"""
        self._log_architectures("test_mixed_seq_and_parallel", self.seq1, self.par1)
        child1, child2 = self.crossover.crossover(
            self.seq1, self.par1, method=CrossoverMethod.SINGLE_POINT
        )
        logger.info(f"=== CHILDREN: {child1} , {child2}")
        self.assertIn("parallel", child1)
        self.assertIn("parallel", child2)
        is_valid1, _ = self.crossover.validator.validate_architecture(child1)
        is_valid2, _ = self.crossover.validator.validate_architecture(child2)
        self.assertTrue(is_valid1)
        self.assertTrue(is_valid2)

    def test_repair_invalid(self):
        """Проверяет, что невалидные потомки пытаются исправиться, если repair_invalid=True"""
        p1 = [{"layer": "Dense", "units": 1, "activation": "relu"}]
        p2 = [{"layer": "Dense", "units": 1, "activation": "relu"}]
        logger.info(f"test_repair_invalid: Parent A → {p1}, Parent B → {p2}")
        child1, child2 = self.crossover.crossover(
            p1, p2, method=CrossoverMethod.SINGLE_POINT
        )
        logger.info(f"=== CHILDREN: {child1} , {child2}")
        self.assertGreaterEqual(len(child1), self.crossover.config.min_architecture_length)
        self.assertGreaterEqual(len(child2), self.crossover.config.min_architecture_length)

    def test_no_repair_invalid(self):
        """Проверяет, что при repair_invalid=False невалидный потомок вызывает ошибку"""
        self.crossover.config.repair_invalid = False
        p1 = [{"layer": "Dense", "units": 1, "activation": "relu"}]
        p2 = [{"layer": "Dense", "units": 1, "activation": "relu"}]
        logger.info(f"test_no_repair_invalid: Parent A → {p1}, Parent B → {p2}")
        with self.assertRaises(CrossoverError):
            self.crossover.crossover(p1, p2, method=CrossoverMethod.SINGLE_POINT)

    def test_random_consistency(self):
        """Несколько случайных скрещиваний не должны ломать валидность"""
        for _ in range(5):
            seq_a = skeleton_conv_dense()
            seq_b = skeleton_rnn()
            logger.info(f"test_random_consistency: Parent A → {seq_a}")
            logger.info(f"test_random_consistency: Parent B → {seq_b}")
            child1, child2 = self.crossover.crossover(
                seq_a, seq_b, method=CrossoverMethod.UNIFORM
            )
            logger.info(f"=== CHILDREN: {child1} , {child2}")
            valid1, _ = self.crossover.validator.validate_architecture(child1)
            valid2, _ = self.crossover.validator.validate_architecture(child2)
            self.assertTrue(valid1)
            self.assertTrue(valid2)


if __name__ == "__main__":
    unittest.main()
