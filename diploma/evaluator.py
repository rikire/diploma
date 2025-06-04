import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Отключить GPU для дочерних процессов

import gc
import time
import logging
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from initializer import generate_population
from builder import SmartModelBuilder, ArchitectureBuilderError


# Настройка логирования
logger = logging.getLogger(__name__)


class FitnessMetrics:
    """
    Класс для хранения различных метрик качества архитектуры.
    После оценки одного варианта архитектуры все поля заполняются,
    и поле `valid` выставляется в True, если оценка прошла без ошибок.
    """
    def __init__(self):
        self.validation_loss: float = float("inf")
        self.train_time: float = float("inf")
        self.model_size: int = 0
        self.inference_time: float = float("inf")
        self.stability: float = float("inf")
        self.complexity: float = float("inf")
        self.valid: bool = False


def calculate_architecture_complexity(arch: Union[List[Dict], Dict]) -> int:
    """
    Вычисляет «сложность» архитектуры как число блоков.
    Для параллельных архитектур суммируется длина каждой ветви.
    Для последовательных — просто длина списка слоёв.
    """
    if isinstance(arch, dict) and "parallel" in arch:
        total = 0
        for branch in arch["parallel"].values():
            total += len(branch)
        return total
    elif isinstance(arch, list):
        return len(arch)
    else:
        # Если формат неожидан (например, None), считаем как «1»
        return 1


def compute_composite_fitness(
    metrics: FitnessMetrics,
    weights: Dict[str, float]
) -> float:
    """
    Вычисляет композитную фитнесс-функцию на основе нескольких метрик.
    Чем меньше результат, тем «лучше» архитектура.
    Параметры:
      - metrics: объект FitnessMetrics, заполненный при оценке
      - weights: словарь, где ключи — названия метрик, значения — их вес
    Возвращает:
      - fitness: float, чем меньше — тем лучше
    """
    if not metrics.valid:
        # Если метрика невалидна (ошибка при сборке/обучении), возвращаем бесконечность
        return float("inf")

    # Основной компонент — validation_loss
    loss_component = metrics.validation_loss * weights.get("loss", 1.0)

    # Размер модели в логарифмической шкале
    size_component = np.log(metrics.model_size + 1) * weights.get("size", 0.0)

    # Время обучения (логарифмическая шкала)
    time_component = np.log(metrics.train_time + 1) * weights.get("training_time", 0.0)

    # Архитектурная сложность
    complexity_component = metrics.complexity * weights.get("complexity", 0.0)

    # Стабильность (чем меньше разброс потерь, тем лучше; напрямую берем std)
    stability_component = metrics.stability * weights.get("stability", 0.0)

    # Время инференса, переводим в миллисекунды
    inference_component = metrics.inference_time * 1000 * weights.get("inference_time", 0.0)

    fitness = (
        loss_component
        + size_component
        + time_component
        + complexity_component
        + stability_component
        + inference_component
    )
    return fitness


def get_parent_log_file() -> Optional[str]:
    """
    Retrieves the log file name used by the parent process, if any.
    """
    for handler in logging.root.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler.baseFilename
    return None


def configure_logging_for_worker(log_file: Optional[str] = None):
    """
    Configures logging for a worker process to ensure logs are written to both console and file.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )


def worker_evaluate_comprehensive(args: Tuple[int, Any, Any, Any, Dict[str, Any], Optional[str]]) -> FitnessMetrics:
    """
    Расширенная функция оценки одной архитектуры с множественными метриками.
    Принимает кортеж: (worker_id, arch, train_data, val_data, test_data, config, log_file)
    Возвращает объект FitnessMetrics.
    """
    worker_id, arch, train_data, val_data, test_data, config, log_file = args
    configure_logging_for_worker(log_file)  # Ensure logging is configured for this worker

    # Устанавливаем сиды для воспроизводимости внутри каждого процесса
    seed = 42 + worker_id
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Конфигурация для обучения
    optimizer = config.get("optimizer", "adam")
    loss_name = config.get("loss", "mse")
    patience = config.get("patience", 2)
    epochs = config.get("epochs", 5)
    verbose = config.get("verbose", 0)
    batch_size = config.get("batch_size", 32)



    metrics = FitnessMetrics()
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data if test_data is not None else (None, None)

    try:
        logger.info(f"[Worker {worker_id}] Начинаем билд и обучение для архитектуры: {arch}")

        # Засекаем время создания и компиляции модели
        build_start = time.time()
        builder = SmartModelBuilder()
        model = builder.build_model_from_architecture(arch, X_train.shape[1:])
        model.compile(
            optimizer=optimizer,
            loss=loss_name,
            metrics=[loss_name]
        )
        build_end = time.time()

        # Фиксируем размер модели (#параметров)
        metrics.model_size = model.count_params()

        # Архитектурная сложность
        metrics.complexity = calculate_architecture_complexity(arch)

        # Обучение с ранней остановкой
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor=f"val_{loss_name}",
            patience=patience,
            restore_best_weights=True,
        )

        train_start = time.time()
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[early_stop],
            verbose=verbose,
            batch_size=batch_size,
        )
        train_end = time.time()

        metrics.train_time = train_end - train_start

        # Лучшее значение валидационной ошибки
        loss_key = f"val_{loss_name}"
        if loss_key in history.history:
            metrics.validation_loss = min(history.history[loss_key])
        else:
            # Если нет ключа (редко бывает), берем последний loss
            metrics.validation_loss = history.history.get(loss_name, [float("inf")])[-1]

        # Стабильность: стандартное отклонение последних 20% эпох (если их достаточно)
        losses = history.history.get(loss_key, [])
        if len(losses) > 1:
            last_n = max(1, int(len(losses) * 0.2))
            recent = losses[-last_n:]
            metrics.stability = float(np.std(recent))
        else:
            metrics.stability = 0.0

        # Измеряем время инференса (если есть тестовые данные)
        if X_test is not None and len(X_test) > 0:
            inf_start = time.time()
            _ = model.predict(X_test, verbose=0, batch_size=batch_size)
            inf_end = time.time()
            metrics.inference_time = (inf_end - inf_start) / len(X_test)
        else:
            metrics.inference_time = 0.0

        metrics.valid = True
        logger.info(f"[Worker {worker_id}] Оценка завершена: val_loss={metrics.validation_loss:.4f}, "
                    f"model_size={metrics.model_size}, train_time={metrics.train_time:.2f}s")

    except ArchitectureBuilderError as abe:
        logger.error(f"[Worker {worker_id}] Ошибка при создании модели: {abe}")
        metrics.valid = False

    except Exception as e:
        logger.error(f"[Worker {worker_id}] Непредвиденная ошибка: {e}", exc_info=True)
        metrics.valid = False

    finally:
        # Гарантируем выгрузку модели и очистку памяти
        try:
            del model
        except Exception:
            pass
        tf.keras.backend.clear_session()
        gc.collect()

    return metrics


def parallel_evaluate_fitness_comprehensive(
    population: List[Union[List[Dict], Dict]],
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    config: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    n_jobs: Optional[int] = None
) -> Tuple[List[float], List[FitnessMetrics]]:
    """
    Комплексная параллельная оценка популяции архитектур.
    Параметры:
      - population: список архитектур (каждая — list или dict)
      - train_data: кортеж (X_train, y_train)
      - val_data: кортеж (X_val, y_val)
      - test_data: кортеж (X_test, y_test) или None
      - config: словарь с настройками обучения (epochs, loss, optimizer и т.д.)
      - weights: словарь весов для композитной функции
      - n_jobs: число параллельных процессов (по умолчанию = min(CPU_count, len(population)))
    Возвращает:
      - fitnesses: список float (значения фитнес-функции, чем меньше — тем лучше)
      - metrics_list: список объектов FitnessMetrics (по одному на архитектуру)
    """
    if config is None:
        config = {
            "epochs": 5,
            "loss": "mse",
            "optimizer": "adam",
            "patience": 2,
            "verbose": 0,
            "batch_size": 32,
        }

    if weights is None:
        weights = {
            "loss": 1.0,
            "size": 0.1,
            "training_time": 0.05,
            "complexity": 0.02,
            "stability": 0.1,
            "inference_time": 0.0,
        }

    n_jobs = n_jobs or min(os.cpu_count() or 1, len(population))
    logger.info(f"Запускаем параллельную оценку: {len(population)} архитектур, {n_jobs} процессов")

    # Формируем аргументы для каждого процесса
    args_list = []
    log_file = get_parent_log_file()
    for i, arch in enumerate(population):
        args_list.append((i, arch, train_data, val_data, test_data, config, log_file))

    with Pool(processes=n_jobs) as pool:
        metrics_list: List[FitnessMetrics] = pool.map(worker_evaluate_comprehensive, args_list)

    # Вычисляем композитные фитнес-значения
    fitnesses = [
        compute_composite_fitness(metrics_list[i], weights)
        for i in range(len(metrics_list))
    ]
    logger.info(f"Оценка завершена. Фитнесы: {fitnesses}")
    return fitnesses, metrics_list


# Альтернативная многокритериальная функция: Pareto-оптимизация
def pareto_fitness_evaluation(
    metrics_list: List[FitnessMetrics]
) -> List[Tuple[Union[int, float], float]]:
    """
    Вычисляет Pareto-фронт для многокритериальной оптимизации.
    Возвращает список кортежей (pareto_rank, crowding_distance) для каждой архитектуры.
    """
    # Фильтруем только валидные метрики
    valid_metrics = [m for m in metrics_list if m.valid]
    if not valid_metrics:
        # Если ни одна архитектура невалидна, возвращаем бесконечности
        return [(float("inf"), 0.0)] * len(metrics_list)

    # Собираем матрицу целей: минимизируем validation_loss, log(model_size+1), train_time, complexity
    objectives = np.array([
        [
            m.validation_loss,
            np.log(m.model_size + 1),
            m.train_time,
            m.complexity,
        ]
        for m in valid_metrics
    ], dtype=float)

    ranks = compute_pareto_ranks(objectives)
    crowding = compute_crowding_distance(objectives, ranks)

    results: List[Tuple[Union[int, float], float]] = []
    valid_idx = 0
    for m in metrics_list:
        if m.valid:
            results.append((int(ranks[valid_idx]), float(crowding[valid_idx])))
            valid_idx += 1
        else:
            results.append((float("inf"), 0.0))
    return results


def compute_pareto_ranks(objectives: np.ndarray) -> np.ndarray:
    """
    Выполняет Fast Non‐Dominated Sorting (NSGA‐II), возвращая ранг Pareto для каждой точки.
    Чем меньше ранг (равный 1, 2, ...), тем ближе решение к Pareto‐фронту.
    
    Параметры:
      - objectives: ndarray формы (N, M), где N — число решений, M — число критериев (минимизируем).
    
    Возвращает:
      - ranks: ndarray длины N, dtype=int, содержащий ранг для каждого решения (1 = первый фронт, 2 = второй и т.д.).
    """
    N = objectives.shape[0]
    # S[p] — список индексов решений, которые доминируются p
    S = [set() for _ in range(N)]
    # n[p] — число решений, которые доминируют над p
    n = np.zeros(N, dtype=int)
    # ranks[p] — итоговый ранг решения p
    ranks = np.zeros(N, dtype=int)

    # Функция доминирования: возвращает True, если a доминирует над b
    def dominates(a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a <= b) and np.any(a < b)

    # Шаг 1: инициализация S и n
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                S[p].add(q)
            elif dominates(objectives[q], objectives[p]):
                n[p] += 1

    # Шаг 2: формируем первый фронт (ранг = 1) — все p, у которых n[p] == 0
    current_front = []
    for p in range(N):
        if n[p] == 0:
            ranks[p] = 1
            current_front.append(p)

    # Шаг 3: итеративно строим следующие фронты
    front_rank = 1
    while current_front:
        next_front = []
        for p in current_front:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    ranks[q] = front_rank + 1
                    next_front.append(q)
        front_rank += 1
        current_front = next_front

    return ranks


def compute_crowding_distance(objectives: np.ndarray, ranks: np.ndarray) -> np.ndarray:
    """
    Вычисляет crowding distance для каждой точки в каждой ранговой группе.
    Нормализуем всё в [0,1], а потом для каждой ранговой группы:
      - выставляем бесконечность для крайних точек по каждому критерию
      - для внутренних точек суммируем расстояния между соседями
    """
    n, m = objectives.shape
    distances = np.zeros(n, dtype=float)

    # Нормализация по каждому столбцу
    eps = 1e-10
    min_vals = np.min(objectives, axis=0)
    max_vals = np.max(objectives, axis=0)
    normalized = (objectives - min_vals) / (max_vals - min_vals + eps)

    unique_ranks = np.unique(ranks)
    for rank in unique_ranks:
        idxs = np.where(ranks == rank)[0]
        if len(idxs) <= 2:
            # Если ранговая группа мала, ставим бесконечность (для выбора)
            distances[idxs] = float("inf")
            continue

        for dim in range(m):
            sorted_idxs = idxs[np.argsort(normalized[idxs, dim])]
            distances[sorted_idxs[0]] = float("inf")
            distances[sorted_idxs[-1]] = float("inf")
            for k in range(1, len(sorted_idxs) - 1):
                prev_i = sorted_idxs[k - 1]
                next_i = sorted_idxs[k + 1]
                distances[sorted_idxs[k]] += (
                    normalized[next_i, dim] - normalized[prev_i, dim]
                )

    return distances


if __name__ == "__main__":
    """
    Пример использования evaluator.py:
    - Генерируем популяцию архитектур при помощи initializer.generate_population()
    - Синтетически создаём train/val/test датасеты
    - Запускаем параллельную оценку fitness
    - Печатаем результаты (композитные фитнес-значения, Pareto-фронт)
    - Выбираем лучшую архитектуру и выводим её метрики
    """
    # Настройка логирования для скрипта
    logger.setLevel(logging.INFO)

    # Генерируем популяцию из 10 случайных архитектур
    pop_size = 10
    try:
        population = generate_population(size=pop_size)
        logger.info(f"Сгенерирована популяция из {pop_size} архитектур")
    except Exception as e:
        logger.error(f"Не удалось сгенерировать популяцию: {e}", exc_info=True)
        population = []

    # Генерация синтетического датасета
    n_samples = 1000
    timesteps = 100
    features = 3
    X = np.random.rand(n_samples, timesteps, features).astype(np.float32)
    y = np.random.rand(n_samples).astype(np.float32)

    # Разбиение на train/val/test
    train_end = int(0.6 * n_samples)
    val_end = int(0.8 * n_samples)

    train_data = (X[:train_end], y[:train_end])
    val_data = (X[train_end:val_end], y[train_end:val_end])
    test_data = (X[val_end:], y[val_end:])

    # Параметры обучения
    config = {
        "epochs": 20,
        "loss": "mse",
        "optimizer": "adam",
        "patience": 5,
        "verbose": 0,
        "batch_size": 32,
    }

    # Веса для композитной фитнес-функции
    weights = {
        "loss": 1.0,
        "size": 0.5,
        "training_time": 0.05,
        "complexity": 0.2,
        "stability": 0.1,
        "inference_time": 0.01,
    }

    # Запускаем параллельную оценку фитнеса
    logger.info("=== Запуск комплексной параллельной оценки фитнеса ===")
    fitnesses, metrics_list = parallel_evaluate_fitness_comprehensive(
        population=population,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        config=config,
        weights=weights,
        n_jobs=4,
    )

    logger.info(f"Композитные фитнес-значения: {fitnesses}")

    # Pareto-оптимизация
    logger.info("=== Многокритериальная Pareto-оценка ===")
    pareto_results = pareto_fitness_evaluation(metrics_list)
    logger.info(f"Pareto ранги и crowding distances: {pareto_results}")

    # Выводим детальную информацию о лучшей архитектуре
    if fitnesses:
        best_idx = int(np.argmin(fitnesses))
        best_metrics = metrics_list[best_idx]
        logger.info(f"=== Лучшая архитектура (индекс {best_idx}) ===")
        logger.info(f"Архитектура: {population[best_idx]}")
        logger.info(f"Валидационная ошибка: {best_metrics.validation_loss:.4f}")
        logger.info(f"Размер модели: {best_metrics.model_size} параметров")
        logger.info(f"Время обучения: {best_metrics.train_time:.2f} сек")
        logger.info(f"Стабильность (std последних эпох): {best_metrics.stability:.4f}")
        logger.info(f"Архитектурная сложность: {best_metrics.complexity}")
        logger.info(f"Время инференса на сэмпл: {best_metrics.inference_time:.6f} сек")
    else:
        logger.warning("Не удалось оценить ни одной архитектуры.")
