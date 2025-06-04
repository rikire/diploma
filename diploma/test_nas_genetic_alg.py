# test_nas_genetic_alg.py
import unittest
import os
import logging
import numpy as np
import random
import sys

# Добавляем директорию с модулем nas_genetic_alg в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nas_genetic_alg import NasGeneticAlg

# Создаём директорию для логов, если её ещё нет
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "test_nas_genetic_alg.log")

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


def generate_mackey_glass(beta=0.2, gamma=0.1, n=10, tau=17, dt=0.1,
                          total_time=200, burn_in_time=50):
    total_steps = int((total_time + burn_in_time) / dt)
    burn_in_steps = int(burn_in_time / dt)
    tau_steps = int(tau / dt)
    x0 = ((beta / gamma) - 1) ** (1 / n)
    x = np.zeros(total_steps)
    x[:tau_steps] = x0 + 0.01 * np.random.randn(tau_steps)
    for t in range(tau_steps, total_steps):
        dx = dt * (beta * x[t - tau_steps] / (1 + x[t - tau_steps] ** n) -
                   gamma * x[t - 1])
        x[t] = x[t - 1] + dx
    return x[burn_in_steps:]

def make_dataset(series: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.float32)
    return X, y


class TestGeneticNAS(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        random.seed(123)
        self.series = generate_mackey_glass(total_time=200, burn_in_time=50, dt=0.5)
        self.X, self.y = make_dataset(self.series, window=10)
        n = len(self.X)
        self.train_data = (self.X[:int(0.6 * n)], self.y[:int(0.6 * n)])
        self.val_data = (self.X[int(0.6 * n):int(0.8 * n)], self.y[int(0.6 * n):int(0.8 * n)])
        self.test_data = (self.X[int(0.8 * n):], self.y[int(0.8 * n):])
        self.config = {
            'epochs': 15,
            'loss': 'mse',
            'optimizer': 'adam',
            'patience': 4,
            'batch_size': 32,
            'verbose': 0,
            'ga_config': {
                'population_size': 20,
                'n_generations': 5,
                'mutation_rate': 0.5,
                'crossover_rate': 0.7,
                'crossover_method': 'single_point',
                'elite_size': 2,
                'selection_strategy': 'tournament',
                'use_pareto': False,
                'n_jobs': 3
            }
        }
        self.weights = {
            'loss': 1.0,
            'size': 0.3,
            'training_time': 0.1,
            'complexity': 0.2,
            'stability': 0.05,
            'inference_time': 0.05
        }

    def test_ga_on_mackey_glass(self):
        logger.info("=== Тест: запуск ГА на синтетических данных Маккея-Гласса ===")
        try:
            ga = NasGeneticAlg(self.config)
            best_arch, best_metrics = ga.run(
                train_data=self.train_data,
                val_data=self.val_data,
                test_data=self.test_data,
                weights=self.weights,
            )
            logger.info("=== Best Architecture ===")
            logger.info(best_arch)
            logger.info("=== Best Metrics ===")
            logger.info(f"Validation Loss: {best_metrics.validation_loss:.4f}")
            logger.info(f"Train Time:      {best_metrics.train_time:.2f}s")
            logger.info(f"Inference Time:  {best_metrics.inference_time:.4f}s/sample")
            self.assertIsNotNone(best_arch)
            self.assertTrue(hasattr(best_metrics, "validation_loss"))
            logger.info("Тест ГА завершён успешно.")
        except Exception as e:
            logger.error(f"Ошибка при тестировании ГА: {e}")
            self.fail(f"GA test failed: {e}")

    def cached_evaluate(arch, train_data, val_data, test_data, config, weights, n_jobs=None):
        if n_jobs is None:
            n_jobs = config.get('n_jobs', 3)
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

if __name__ == "__main__":
    unittest.main()