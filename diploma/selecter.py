# selecter.py

import numpy as np
import random
import logging
from typing import List, Tuple, Dict, Any, Callable, Optional, Union

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SelectorError(Exception):
    """Исключение для ошибок селекции"""
    pass


class SelectorConfig:
    """Конфигурация параметров селекции"""
    def __init__(
        self,
        tournament_size: int = 3,
        rank_pressure: float = 1.5,
        diversity_alpha: float = 0.7,
        elitism_ratio: float = 0.1,
        min_population_size: int = 5
    ):
        self.tournament_size = tournament_size
        self.rank_pressure = rank_pressure
        self.diversity_alpha = diversity_alpha
        self.elitism_ratio = elitism_ratio
        self.min_population_size = min_population_size


class Selector:
    """
    Класс, реализующий различные стратегии селекции для генетического алгоритма
    поиска оптимальной архитектуры нейронной сети.
    """
    
    def __init__(self, config: Optional[SelectorConfig] = None):
        self.config = config or SelectorConfig()
        
    def select(
        self,
        population: List[Any],
        fitnesses: List[float],
        method: str,
        num_selected: Optional[int] = None,
        minimize: bool = True,
        **kwargs: Any
    ) -> List[Any]:
        """
        Основная функция селекции особей из популяции
        
        Args:
            population: Список архитектур
            fitnesses: Список значений фитнес-функции
            method: Метод селекции ('tournament', 'roulette', 'rank', 'elitist', 'diversity')
            num_selected: Количество особей для отбора
            minimize: Минимизация фитнес-функции (True) или максимизация (False)
            **kwargs: Дополнительные параметры для конкретных методов
            
        Returns:
            List: Отобранные архитектуры
            
        Raises:
            SelectorError: При ошибках валидации или селекции
        """
        # Валидация входных данных
        self._validate_input(population, fitnesses)
        
        if num_selected is None:
            num_selected = len(population)
            
        logger.info(f"Запуск селекции методом {method}. Отбираем {num_selected} особей из {len(population)}")
        
        try:
            if method == 'tournament':
                return self.tournament_selection(
                    population, fitnesses, num_selected, minimize, **kwargs
                )
            elif method == 'roulette':
                return self.roulette_wheel_selection(
                    population, fitnesses, num_selected, minimize, **kwargs
                )
            elif method == 'rank':
                return self.rank_selection(
                    population, fitnesses, num_selected, minimize, **kwargs
                )
            elif method == 'elitist':
                return self.elitist_selection(
                    population, fitnesses, num_selected, minimize, **kwargs
                )
            elif method == 'diversity':
                return self.diversity_preserving_selection(
                    population, fitnesses, num_selected, minimize, **kwargs
                )
            elif method == 'pareto':
                return self.pareto_selection(
                    population, fitnesses, num_selected, minimize, **kwargs
                )
            else:
                raise SelectorError(f"Неизвестный метод селекции: {method}")
        except Exception as e:
            logger.error(f"Ошибка селекции: {str(e)}")
            raise SelectorError(f"Ошибка селекции: {str(e)}") from e

    def _validate_input(self, population: List[Any], fitnesses: List[float]) -> None:
        """Проверяет корректность входных данных"""
        if len(population) < self.config.min_population_size:
            raise SelectorError(
                f"Размер популяции ({len(population)}) меньше минимального ({self.config.min_population_size})"
            )
            
        if len(population) != len(fitnesses):
            raise SelectorError(
                f"Размеры популяции ({len(population)}) и фитнес-значений ({len(fitnesses)}) не совпадают"
            )
            
        if any(not np.isfinite(f) for f in fitnesses):
            raise SelectorError("Фитнес-значения содержат нечисловые или бесконечные значения")

    def tournament_selection(
        self,
        population: List[Any], 
        fitnesses: List[float], 
        num_selected: int,
        minimize: bool = True,
        **kwargs: Any
    ) -> List[Any]:
        """
        Турнирная селекция: случайно выбираем k особей и отбираем лучшую.
        
        Args:
            population: Список архитектур
            fitnesses: Список значений фитнес-функции
            num_selected: Количество особей для отбора
            minimize: Минимизация фитнес-функции
            **kwargs: Дополнительные параметры (k - размер турнира)
            
        Returns:
            List: Отобранные архитектуры
        """
        k = kwargs.get('k', self.config.tournament_size)
        if k < 2:
            raise SelectorError(f"Некорректный размер турнира: {k}. Должен быть >= 2")
            
        logger.info(f"Турнирная селекция с размером турнира {k}")
        
        selected = []
        for i in range(num_selected):
            # Выбираем k случайных индексов
            tournament_indices = random.sample(range(len(population)), k)
            
            # Выбираем лучшую особь из турнира
            if minimize:
                best_idx = min(tournament_indices, key=lambda idx: fitnesses[idx])
            else:
                best_idx = max(tournament_indices, key=lambda idx: fitnesses[idx])
                
            selected.append(population[best_idx])
            
            logger.debug(f"Турнир {i+1}: участники {tournament_indices}, победитель {best_idx} (фитнес={fitnesses[best_idx]:.4f})")
        
        logger.info(f"Отобрано {len(selected)} особей турнирной селекцией")
        return selected
    
    def roulette_wheel_selection(
        self,
        population: List[Any], 
        fitnesses: List[float], 
        num_selected: int,
        minimize: bool = True,
        **kwargs: Any
    ) -> List[Any]:
        """
        Селекция методом рулетки: вероятность выбора пропорциональна
        обратному значению фитнес-функции (для задачи минимизации).
        
        Args:
            population: Список архитектур
            fitnesses: Список значений фитнес-функции
            num_selected: Количество особей для отбора
            minimize: Минимизация фитнес-функции
            **kwargs: Дополнительные параметры
            
        Returns:
            List: Отобранные архитектуры
        """
        logger.info("Рулеточная селекция")
        
        # Преобразуем фитнес-значения для задачи минимизации/максимизации
        if minimize:
            max_fitness = max(fitnesses)
            weights = [max_fitness - f + 1e-10 for f in fitnesses]
        else:
            weights = [f + 1e-10 for f in fitnesses]
        
        # Нормализуем веса
        total_weight = sum(weights)
        if total_weight <= 0:
            logger.warning("Сумма весов <= 0, используем равномерное распределение")
            probabilities = [1/len(population)] * len(population)
        else:
            probabilities = [w / total_weight for w in weights]
        
        # Выбираем архитектуры с вероятностью пропорциональной их весам
        selected_indices = np.random.choice(
            len(population), 
            size=num_selected, 
            replace=True, 
            p=probabilities
        )
        
        selected = [population[i] for i in selected_indices]
        
        logger.info(f"Отобрано {len(selected)} особей рулеточной селекцией")
        logger.debug(f"Вероятности: {probabilities[:5]}...")
        logger.debug(f"Выбранные индексы: {selected_indices[:10]}...")
        
        return selected
    
    def rank_selection(
        self,
        population: List[Any], 
        fitnesses: List[float], 
        num_selected: int,
        minimize: bool = True,
        **kwargs: Any
    ) -> List[Any]:
        """
        Ранговая селекция: вероятность выбора основана на ранге особи в популяции.
        
        Args:
            population: Список архитектур
            fitnesses: Список значений фитнес-функции
            num_selected: Количество особей для отбора
            minimize: Минимизация фитнес-функции
            **kwargs: Дополнительные параметры (pressure - параметр давления)
            
        Returns:
            List: Отобранные архитектуры
        """
        pressure = kwargs.get('pressure', self.config.rank_pressure)
        if pressure < 1.0 or pressure > 2.0:
            raise SelectorError(f"Некорректное значение давления: {pressure}. Должно быть между 1.0 и 2.0")
            
        logger.info(f"Ранговая селекция с давлением {pressure}")
        
        # Сортируем индексы по фитнесу
        indices = list(range(len(population)))
        if minimize:
            indices.sort(key=lambda i: fitnesses[i])
        else:
            indices.sort(key=lambda i: fitnesses[i], reverse=True)
        
        # Вычисляем веса на основе рангов
        n = len(population)
        ranks = list(range(1, n + 1))
        
        # Формула для линейного ранжирования
        weights = [2 - pressure + 2 * (pressure - 1) * (n - r) / (n - 1) for r in ranks]
        
        # Нормализуем веса
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        
        # Отображаем ранги на отсортированные индексы
        rank_to_idx = {rank: idx for rank, idx in enumerate(indices, 1)}
        
        # Выбираем архитектуры с вероятностью пропорциональной их рангу
        selected_ranks = np.random.choice(
            ranks, 
            size=num_selected, 
            replace=True, 
            p=probabilities
        )
        
        selected = [population[rank_to_idx[rank]] for rank in selected_ranks]
        
        logger.info(f"Отобрано {len(selected)} особей ранговой селекцией")
        logger.debug(f"Вероятности: {probabilities[:5]}...")
        logger.debug(f"Выбранные ранги: {selected_ranks[:10]}...")
        
        return selected
    
    def elitist_selection(
        self,
        population: List[Any], 
        fitnesses: List[float], 
        num_selected: int,
        minimize: bool = True,
        **kwargs: Any
    ) -> List[Any]:
        """
        Элитарная селекция: сохраняет лучшие особи,
        затем использует другую стратегию для отбора остальных.
        
        Args:
            population: Список архитектур
            fitnesses: Список значений фитнес-функции
            num_selected: Количество особей для отбора
            minimize: Минимизация фитнес-функции
            **kwargs: Дополнительные параметры (elitism_ratio - доля элиты)
            
        Returns:
            List: Отобранные архитектуры
        """
        elitism_ratio = kwargs.get('elitism_ratio', self.config.elitism_ratio)
        if elitism_ratio < 0 or elitism_ratio > 1:
            raise SelectorError(f"Некорректное значение доли элиты: {elitism_ratio}. Должно быть между 0 и 1")
            
        num_elites = max(1, int(num_selected * elitism_ratio))
        num_rest = num_selected - num_elites
        
        logger.info(f"Элитарная селекция: {num_elites} элитных и {num_rest} остальных особей")
        
        # Сортируем индексы по фитнесу
        indices = list(range(len(population)))
        if minimize:
            indices.sort(key=lambda i: fitnesses[i])
        else:
            indices.sort(key=lambda i: fitnesses[i], reverse=True)
        
        # Выбираем элитных особей
        elites = [population[i] for i in indices[:num_elites]]
        
        # Используем турнирную селекцию для остальных
        rest = self.tournament_selection(
            population, fitnesses, num_rest, minimize
        )
        
        # Объединяем элиту и остальных
        selected = elites + rest
        
        logger.info(f"Отобрано {len(selected)} особей элитарной селекцией")
        logger.debug(f"Элитные индексы: {indices[:num_elites]}")
        logger.debug(f"Элитные фитнес-значения: {[fitnesses[i] for i in indices[:num_elites]]}")
        
        return selected
    
    def diversity_preserving_selection(
        self,
        population: List[Any], 
        fitnesses: List[float], 
        num_selected: int,
        minimize: bool = True,
        **kwargs: Any
    ) -> List[Any]:
        """
        Селекция с сохранением разнообразия: учитывает как фитнес, 
        так и разнообразие при отборе особей.
        
        Args:
            population: Список архитектур
            fitnesses: Список значений фитнес-функции
            num_selected: Количество особей для отбора
            minimize: Минимизация фитнес-функции
            **kwargs: Дополнительные параметры (alpha - вес фитнеса)
            
        Returns:
            List: Отобранные архитектуры
        """
        alpha = kwargs.get('alpha', self.config.diversity_alpha)
        if alpha < 0 or alpha > 1:
            raise SelectorError(f"Некорректное значение alpha: {alpha}. Должно быть между 0 и 1")
            
        logger.info(f"Селекция с сохранением разнообразия (alpha={alpha})")
        
        # Рассчитываем показатели разнообразия
        diversity_scores = calculate_architecture_diversity(population)
        
        # Нормализуем фитнес (для минимизации/максимизации)
        if minimize:
            max_fitness = max(fitnesses)
            norm_fitnesses = [(max_fitness - f) / (max_fitness - min(fitnesses) + 1e-10) for f in fitnesses]
        else:
            min_fitness = min(fitnesses)
            norm_fitnesses = [(f - min_fitness) / (max(fitnesses) - min_fitness + 1e-10) for f in fitnesses]
        
        # Нормализуем разнообразие (больше - лучше)
        min_div = min(diversity_scores)
        norm_diversity = [(d - min_div) / (max(diversity_scores) - min_div + 1e-10) for d in diversity_scores]
        
        # Комбинируем фитнес и разнообразие
        combined_scores = [alpha * nf + (1-alpha) * nd for nf, nd in zip(norm_fitnesses, norm_diversity)]
        
        # Используем пропорциональную селекцию на основе комбинированной оценки
        total_score = sum(combined_scores)
        probabilities = [score / total_score for score in combined_scores]
        
        # Выбираем архитектуры
        selected_indices = np.random.choice(
            len(population), 
            size=num_selected, 
            replace=True, 
            p=probabilities
        )
        
        selected = [population[i] for i in selected_indices]
        
        logger.info(f"Отобрано {len(selected)} особей с учетом разнообразия")
        logger.debug(f"Нормализованный фитнес: {norm_fitnesses[:5]}...")
        logger.debug(f"Нормализованное разнообразие: {norm_diversity[:5]}...")
        logger.debug(f"Комбинированные оценки: {combined_scores[:5]}...")
        
        return selected
    
    def pareto_selection(
        self,
        population: List[Any], 
        fitnesses: List[float], 
        num_selected: int,
        minimize: bool = True,
        **kwargs: Any
    ) -> List[Any]:
        """
        Селекция по Парето: отбор особей, находящихся на границе Парето.
        
        Args:
            population: Список архитектур
            fitnesses: Список значений фитнес-функции
            num_selected: Количество особей для отбора
            minimize: Минимизация фитнес-функции
            **kwargs: Дополнительные параметры (pareto_front - индексы особей на Парето-фронте)
            
        Returns:
            List: Отобранные архитектуры
        """
        pareto_front = kwargs.get('pareto_front', [])
        if not pareto_front:
            raise SelectorError("Необходимо предоставить индексы особей на Парето-фронте")
        
        logger.info(f"Селекция по Парето, найдено {len(pareto_front)} особей на фронте")
        
        # Выбираем особей, находящихся на Парето-фронте
        selected = [population[i] for i in pareto_front]
        
        # Если нужно, дополняем до num_selected случайными особями
        if len(selected) < num_selected:
            remaining_indices = list(set(range(len(population))) - set(pareto_front))
            additional_indices = random.sample(remaining_indices, num_selected - len(selected))
            selected += [population[i] for i in additional_indices]
        
        logger.info(f"Отобрано {len(selected)} особей по Парето-селекции")
        return selected


# Вспомогательные функции для расчета разнообразия
def calculate_architecture_diversity(population: List[Any]) -> List[float]:
    """
    Вычисляет меру разнообразия для каждой архитектуры в популяции
    на основе ее отличия от других архитектур.
    
    Args:
        population: Список архитектур нейронных сетей
        
    Returns:
        List[float]: Список значений разнообразия для каждой архитектуры
    """
    n = len(population)
    if n == 0:
        return []
    
    diversity_scores = [0.0] * n
    
    for i in range(n):
        arch_i = population[i]
        
        # Считаем среднее расстояние до других архитектур
        distances = []
        for j in range(n):
            if i != j:
                arch_j = population[j]
                dist = architecture_distance(arch_i, arch_j)
                distances.append(dist)
        
        # Среднее расстояние как мера разнообразия
        if distances:
            diversity_scores[i] = sum(distances) / len(distances)
    
    return diversity_scores

def architecture_distance(arch1: Any, arch2: Any) -> float:
    """
    Вычисляет расстояние между двумя архитектурами.
    
    Args:
        arch1, arch2: Архитектуры нейронных сетей
        
    Returns:
        float: Значение расстояния между архитектурами
    """
    # Обрабатываем параллельные архитектуры
    if isinstance(arch1, dict) and 'parallel' in arch1:
        if not (isinstance(arch2, dict) and 'parallel' in arch2):
            return 1.0  # Максимальное различие
        
        # Сравниваем каждую ветвь
        branches1 = list(arch1['parallel'].values())
        branches2 = list(arch2['parallel'].values())
        
        # Если разное количество ветвей
        if len(branches1) != len(branches2):
            return 0.5 + 0.5 * abs(len(branches1) - len(branches2)) / max(len(branches1), len(branches2))
        
        # Сравниваем попарно ветви
        branch_distances = []
        for b1, b2 in zip(branches1, branches2):
            # Каждая ветвь - это список слоев
            branch_distances.append(layers_distance(b1, b2))
        
        return sum(branch_distances) / len(branch_distances)
    
    # Обрабатываем последовательные архитектуры
    elif isinstance(arch1, list) and isinstance(arch2, list):
        return layers_distance(arch1, arch2)
    
    # Разные типы архитектур
    else:
        return 1.0  # Максимальное различие

def layers_distance(layers1: List[Dict], layers2: List[Dict]) -> float:
    """
    Вычисляет расстояние между двумя последовательностями слоев.
    
    Args:
        layers1, layers2: Списки слоев архитектур
        
    Returns:
        float: Значение расстояния между слоями
    """
    # Разная длина архитектур
    if len(layers1) != len(layers2):
        return 0.5 + 0.5 * abs(len(layers1) - len(layers2)) / max(len(layers1), len(layers2))
    
    # Считаем количество различных слоев
    layer_diff_count = 0
    for l1, l2 in zip(layers1, layers2):
        # Разные типы слоев
        if l1.get('layer') != l2.get('layer'):
            layer_diff_count += 1
            continue
        
        # Для одинаковых типов сравниваем параметры
        params_diff = 0
        common_keys = set(l1.keys()) & set(l2.keys())
        for key in common_keys:
            if l1[key] != l2[key]:
                params_diff += 1
        
        # Нормализуем различие параметров
        if common_keys:
            layer_diff_count += params_diff / len(common_keys)
    
    return layer_diff_count / len(layers1)