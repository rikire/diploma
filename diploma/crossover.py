# crossover.py

import random
import copy
import logging
from enum import Enum
from typing import Any, Dict, List, Tuple, Union, Optional

from initializer import ArchitectureValidator, ArchitectureConstraints
from builder import SmartModelBuilder, ArchitectureBuilderError

# Настройка логирования
logger = logging.getLogger(__name__)


class CrossoverMethod(Enum):
    """Перечисление методов скрещивания"""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"


class CrossoverError(Exception):
    """Исключение для ошибок скрещивания"""
    pass


class CrossoverConfig:
    """Конфигурация для операторов скрещивания"""
    def __init__(
        self,
        uniform_prob: float = 0.5,            # вероятность для равномерного скрещивания
        min_architecture_length: int = 2,     # минимальная длина последовательной архитектуры
        max_architecture_length: int = 20,    # максимальная длина последовательной архитектуры
        repair_invalid: bool = True,          # автоматически исправлять невалидные потомки
        allow_empty_branches: bool = False    # разрешать пустые ветви в параллельных архит.
    ):
        self.uniform_prob = uniform_prob
        self.min_architecture_length = min_architecture_length
        self.max_architecture_length = max_architecture_length
        self.repair_invalid = repair_invalid
        self.allow_empty_branches = allow_empty_branches


class ArchitectureCrossover:
    """
    Класс для выполнения операций скрещивания архитектур нейронных сетей.
    Поддерживает последовательные (list[dict]) и параллельные ({'parallel': {...}}) архитектуры.
    """
    def __init__(self, config: Optional[CrossoverConfig] = None):
        self.config = config or CrossoverConfig()
        self.validator = ArchitectureValidator(ArchitectureConstraints())
        self.builder = SmartModelBuilder()

    def crossover(
        self,
        parent1: Union[List[Dict], Dict],
        parent2: Union[List[Dict], Dict],
        method: Union[CrossoverMethod, str] = CrossoverMethod.SINGLE_POINT,
        **kwargs: Any
    ) -> Tuple[Union[List[Dict], Dict], Union[List[Dict], Dict]]:
        """
        Основная функция скрещивания двух родительских архитектур.

        Args:
            parent1: первая родительская архитектура (последовательная или параллельная)
            parent2: вторая родительская архитектура
            method: метод скрещивания (SINGLE_POINT, TWO_POINT, UNIFORM)
            **kwargs: дополнительные параметры (например, точки слияния)

        Returns:
            (child1, child2) – два потомка аналогичного формата

        Raises:
            CrossoverError: при ошибках валидации или скрещивания
        """
        if isinstance(method, str):
            try:
                method = CrossoverMethod(method)
            except ValueError:
                raise CrossoverError(f"Неизвестный метод скрещивания: {method}")

        logger.info(f"Запуск скрещивания методом {method.value}")
        # Отключаем логирование initializer во время скрещивания (чтобы не засорять вывод)
        logging.getLogger("initializer").setLevel(logging.WARNING)

        # Валидация родителей
        self._validate_parent(parent1, "parent1")
        self._validate_parent(parent2, "parent2")

        # Определяем, параллельные ли архитектуры
        p1_parallel = isinstance(parent1, dict) and "parallel" in parent1
        p2_parallel = isinstance(parent2, dict) and "parallel" in parent2

        if p1_parallel or p2_parallel:
            # Приводим к параллельному формату
            p1 = self._ensure_parallel(parent1)
            p2 = self._ensure_parallel(parent2)
            child1, child2 = self._crossover_parallel(p1, p2, method, **kwargs)
        else:
            # Последовательные архитектуры
            child1, child2 = self._crossover_sequential(parent1, parent2, method, **kwargs)

        # Постобработка и валидация потомков
        child1 = self._postprocess(child1, "child1", parent1, parent2)
        child2 = self._postprocess(child2, "child2", parent1, parent2)
        logger.info("Скрещивание завершено успешно")
        return child1, child2

    def _validate_parent(self, parent: Union[List[Dict], Dict], name: str) -> None:
        """Проверяет, что родительская архитектура непуста и не содержит неподдерживаемых ключей."""
        if not parent:
            raise CrossoverError(f"{name} пустая архитектура")

        # Проверка неподдерживаемых параметров в слоях
        if isinstance(parent, list):
            for layer in parent:
                self._check_layer_keys(layer, name)
        elif isinstance(parent, dict) and "parallel" in parent:
            for branch, layers in parent["parallel"].items():
                for layer in layers:
                    self._check_layer_keys(layer, f"{name}.{branch}")

        is_valid, err = self.validator.validate_architecture(parent)
        if not is_valid:
            logger.warning(f"{name} невалидна: {err}")
            if not self.config.repair_invalid:
                raise CrossoverError(f"{name} невалидна: {err}")

    def _check_layer_keys(self, layer: Dict, context: str) -> None:
        """
        Проверяет, нет ли в слое неподдерживаемых ключей:
         - ключ 'bidirectional' теперь допускается и обрабатывается билдером
         - 'kernel_regularizer="L1L2"' допускается, если билдер это поддерживает
        """
        if layer.get("bidirectional", False):
            logger.debug(f"{context}: встретился параметр 'bidirectional', будет обработан билдером")

        kr = layer.get("kernel_regularizer")
        if kr == "L1L2":
            logger.debug(f"{context}: встретился 'kernel_regularizer=\"L1L2\"', убедитесь, что билдер это поддерживает")

    def _ensure_parallel(self, arch: Union[List[Dict], Dict]) -> Dict:
        """
        Гарантирует формат {'parallel': {...}}.
        Если передана list → оборачивает в одну ветвь 'main_branch'.
        """
        if isinstance(arch, dict) and "parallel" in arch:
            return copy.deepcopy(arch)
        elif isinstance(arch, list):
            return {"parallel": {"main_branch": copy.deepcopy(arch)}}
        else:
            raise CrossoverError(f"Неподдерживаемый формат архитектуры: {type(arch)}")

    # -------------------------
    # Скрещивание последовательных архитектур
    # -------------------------
    def _crossover_sequential(
        self,
        p1: List[Dict],
        p2: List[Dict],
        method: CrossoverMethod,
        **kwargs: Any
    ) -> Tuple[List[Dict], List[Dict]]:
        if method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_seq(p1, p2, **kwargs)
        elif method == CrossoverMethod.TWO_POINT:
            return self._two_point_seq(p1, p2, **kwargs)
        elif method == CrossoverMethod.UNIFORM:
            return self._uniform_seq(p1, p2, **kwargs)
        else:
            raise CrossoverError(f"Метод {method.value} не поддерживается для последовательных архитектур")

    def _single_point_seq(
        self,
        p1: List[Dict],
        p2: List[Dict],
        **kwargs: Any
    ) -> Tuple[List[Dict], List[Dict]]:
        """Одноточечное скрещивание для последовательных архитектур"""
        len1, len2 = len(p1), len(p2)
        min_len = min(len1, len2)
        if min_len < 2:
            logger.warning(f"Одна из архитектур слишком коротка ({min_len}) для single_point")
            if not self.config.repair_invalid:
                raise CrossoverError("Невозможно выполнить single_point для архитектур длины < 2")
            # fallback: вернём копии, потом расширим
            return copy.deepcopy(p1), copy.deepcopy(p2)

        point = kwargs.get("point", random.randint(1, min_len - 1))
        logger.debug(f"Одноточечное скрещивание в точке {point}")
        child1 = copy.deepcopy(p1[:point] + p2[point:])
        child2 = copy.deepcopy(p2[:point] + p1[point:])
        return child1, child2

    def _two_point_seq(
        self,
        p1: List[Dict],
        p2: List[Dict],
        **kwargs: Any
    ) -> Tuple[List[Dict], List[Dict]]:
        """Двухточечное скрещивание для последовательных архитектур"""
        len1, len2 = len(p1), len(p2)
        min_len = min(len1, len2)
        if min_len < 3:
            logger.warning(f"Одна из архитектур слишком коротка ({min_len}) для two_point")
            return self._single_point_seq(p1, p2, **kwargs)

        # Выбор точек так, чтобы гарантировать сегмент длины >=1
        pt1 = kwargs.get("point1", random.randint(1, min_len - 2))
        pt2 = kwargs.get("point2", random.randint(pt1 + 1, min_len - 1))
        if pt2 <= pt1:
            pt2 = pt1 + 1
            if pt2 >= min_len:
                pt1 = 1
                pt2 = 2

        logger.debug(f"Двухточечное скрещивание в точках {pt1}, {pt2}")
        child1 = copy.deepcopy(p1[:pt1] + p2[pt1:pt2] + p1[pt2:])
        child2 = copy.deepcopy(p2[:pt1] + p1[pt1:pt2] + p2[pt2:])
        return child1, child2

    def _uniform_seq(
        self,
        p1: List[Dict],
        p2: List[Dict],
        **kwargs: Any
    ) -> Tuple[List[Dict], List[Dict]]:
        """Равномерное скрещивание для последовательных архитектур"""
        prob = kwargs.get("uniform_prob", self.config.uniform_prob)
        max_len = max(len(p1), len(p2))
        child1, child2 = [], []

        for i in range(max_len):
            layer1 = p1[i] if i < len(p1) else None
            layer2 = p2[i] if i < len(p2) else None

            if layer1 is not None and layer2 is not None:
                if random.random() < prob:
                    child1.append(copy.deepcopy(layer1))
                    child2.append(copy.deepcopy(layer2))
                else:
                    child1.append(copy.deepcopy(layer2))
                    child2.append(copy.deepcopy(layer1))
            elif layer1 is not None:
                if random.random() < 0.5:
                    child1.append(copy.deepcopy(layer1))
                else:
                    child2.append(copy.deepcopy(layer1))
            elif layer2 is not None:
                if random.random() < 0.5:
                    child1.append(copy.deepcopy(layer2))
                else:
                    child2.append(copy.deepcopy(layer2))
        return child1, child2

    # -------------------------
    # Скрещивание параллельных архитектур
    # -------------------------
    def _crossover_parallel(
        self,
        p1: Dict[str, Any],
        p2: Dict[str, Any],
        method: CrossoverMethod,
        **kwargs: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_par(p1, p2, **kwargs)
        elif method == CrossoverMethod.TWO_POINT:
            return self._two_point_par(p1, p2, **kwargs)
        elif method == CrossoverMethod.UNIFORM:
            return self._uniform_par(p1, p2, **kwargs)
        else:
            raise CrossoverError(f"Метод {method.value} не поддерживается для параллельных архитектур")

    def _single_point_par(
        self,
        p1: Dict[str, Any],
        p2: Dict[str, Any],
        **kwargs: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Одноточечное скрещивание параллельных архитектур:
        разбиваем список ветвей и обмениваем их части.
        Гарантируем: у каждого потомка есть минимум по одной ветви от каждого родителя.
        """
        b1 = list(p1["parallel"].items())
        b2 = list(p2["parallel"].items())
        if not b1 or not b2:
            logger.warning("Одна из параллельных архитектур пуста")
            return copy.deepcopy(p1), copy.deepcopy(p2)

        # Выбираем точку среза
        ratio = kwargs.get("ratio", 0.5)
        cut1 = max(1, int(len(b1) * ratio))
        cut2 = max(1, int(len(b2) * ratio))
        if cut1 >= len(b1):
            cut1 = len(b1) - 1
        if cut2 >= len(b2):
            cut2 = len(b2) - 1

        logger.debug(f"Одноточечное скрещивание ветвей: cut1={cut1}, cut2={cut2}")

        new_b1 = dict(b1[:cut1] + b2[cut2:])
        new_b2 = dict(b2[:cut2] + b1[cut1:])

        # Гарантируем, что потомки получили хотя бы по одной ветви от каждого родителя
        def ensure_branch(child: Dict[str, Any], parent: Dict[str, Any]):
            for branch in parent["parallel"].keys():
                if branch not in child["parallel"]:
                    child["parallel"][branch] = copy.deepcopy(parent["parallel"][branch])
                    break

        child1 = {"parallel": new_b1}
        child2 = {"parallel": new_b2}
        ensure_branch(child1, p1)
        ensure_branch(child1, p2)
        ensure_branch(child2, p1)
        ensure_branch(child2, p2)

        return child1, child2

    def _two_point_par(
        self,
        p1: Dict[str, Any],
        p2: Dict[str, Any],
        **kwargs: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Двухточечное скрещивание параллельных архитектур:
        выбираем несколько ветвей и обмениваемся ими.
        Гарантируем: у каждого потомка есть минимум по одной ветви от каждого родителя.
        """
        branches = list(set(p1["parallel"].keys()) | set(p2["parallel"].keys()))
        if len(branches) < 2:
            logger.warning("Недостаточно ветвей для двухточечного скрещивания")
            return self._single_point_par(p1, p2, **kwargs)

        num_to_exchange = kwargs.get("num", max(1, len(branches) // 3))
        exch = random.sample(branches, num_to_exchange)
        logger.debug(f"Двухточечное скрещивание ветвей: обменяем {exch}")

        child1 = {"parallel": copy.deepcopy(p1["parallel"])}
        child2 = {"parallel": copy.deepcopy(p2["parallel"])}

        for br in exch:
            if br in p1["parallel"]:
                child2["parallel"][br] = copy.deepcopy(p1["parallel"][br])
            if br in p2["parallel"]:
                child1["parallel"][br] = copy.deepcopy(p2["parallel"][br])

        # Гарантируем, что в потомках есть ветвь от каждого родителя
        def ensure_branch(child: Dict[str, Any], parent: Dict[str, Any]):
            for branch in parent["parallel"].keys():
                if branch not in child["parallel"]:
                    child["parallel"][branch] = copy.deepcopy(parent["parallel"][branch])
                    break

        ensure_branch(child1, p1)
        ensure_branch(child1, p2)
        ensure_branch(child2, p1)
        ensure_branch(child2, p2)

        return child1, child2

    def _uniform_par(
        self,
        p1: Dict[str, Any],
        p2: Dict[str, Any],
        **kwargs: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Равномерное скрещивание параллельных архитектур: по ветвям с вероятностью uniform_prob"""
        prob = kwargs.get("uniform_prob", self.config.uniform_prob)
        branches = list(set(p1["parallel"].keys()) | set(p2["parallel"].keys()))
        logger.debug(f"Равномерное скрещивание ветвей: {branches}")

        child1 = {"parallel": {}}
        child2 = {"parallel": {}}
        for br in branches:
            in_p1 = br in p1["parallel"]
            in_p2 = br in p2["parallel"]
            if in_p1 and in_p2:
                if random.random() < prob:
                    child1["parallel"][br] = copy.deepcopy(p1["parallel"][br])
                    child2["parallel"][br] = copy.deepcopy(p2["parallel"][br])
                else:
                    child1["parallel"][br] = copy.deepcopy(p2["parallel"][br])
                    child2["parallel"][br] = copy.deepcopy(p1["parallel"][br])
            elif in_p1:
                if random.random() < 0.5:
                    child1["parallel"][br] = copy.deepcopy(p1["parallel"][br])
                else:
                    child2["parallel"][br] = copy.deepcopy(p1["parallel"][br])
            else:  # in_p2
                if random.random() < 0.5:
                    child1["parallel"][br] = copy.deepcopy(p2["parallel"][br])
                else:
                    child2["parallel"][br] = copy.deepcopy(p2["parallel"][br])

        # Гарантируем, что в потомках есть ветвь от каждого родителя
        def ensure_branch(child: Dict[str, Any], parent: Dict[str, Any]):
            for branch in parent["parallel"].keys():
                if branch not in child["parallel"]:
                    child["parallel"][branch] = copy.deepcopy(parent["parallel"][branch])
                    break

        ensure_branch(child1, p1)
        ensure_branch(child1, p2)
        ensure_branch(child2, p1)
        ensure_branch(child2, p2)

        return child1, child2

    # -------------------------
    # Постобработка потомка
    # -------------------------
    def _postprocess(
        self,
        child: Union[List[Dict], Dict],
        name: str,
        original1: Union[List[Dict], Dict],
        original2: Union[List[Dict], Dict]
    ) -> Union[List[Dict], Dict]:
        """
        Проверяет длину (для последовательных) и валидность.
        При необходимости обрезает/дополняет и исправляет невалидные архитектуры.
        Также гарантирует, что потомок содержит хотя бы минимально одну ветвь от каждого родителя.
        """
        logger.debug(f"Постобработка потомка {name}")

        # Проверка на пустую последовательную архитектуру
        if isinstance(child, list) and len(child) == 0:
            raise CrossoverError(f"{name}: получилось 0 слоев после скрещивания")

        # Последовательная архитектура
        if isinstance(child, list):
            length = len(child)
            if length < self.config.min_architecture_length:
                if not self.config.repair_invalid:
                    raise CrossoverError(
                        f"{name}: длина ({length}) < min ({self.config.min_architecture_length})"
                    )
                logger.warning(f"{name}: длина ({length}) < min, дополняем")
                child = self._extend_seq(child)
                length = len(child)

            if length > self.config.max_architecture_length:
                if not self.config.repair_invalid:
                    raise CrossoverError(
                        f"{name}: длина ({length}) > max ({self.config.max_architecture_length})"
                    )
                logger.warning(f"{name}: длина ({length}) > max, обрезаем")
                child = child[: self.config.max_architecture_length]

        # Параллельная архитектура
        elif isinstance(child, dict) and "parallel" in child:
            # Удаляем ветви с пустым списком, если это запрещено
            if not self.config.allow_empty_branches:
                pruned = {k: v for k, v in child["parallel"].items() if v}
                if not pruned:
                    raise CrossoverError(f"{name}: все ветви оказались пустыми")
                child = {"parallel": pruned}

            # Гарантируем, что у потомка есть ветви от обоих родителей
            p1 = self._ensure_parallel(original1)
            p2 = self._ensure_parallel(original2)
            for parent in (p1, p2):
                found = any(br in child["parallel"] for br in parent["parallel"].keys())
                if not found:
                    # Добавляем первую ветвь от этого родителя
                    branch_to_restore = next(iter(parent["parallel"].keys()))
                    child["parallel"][branch_to_restore] = copy.deepcopy(parent["parallel"][branch_to_restore])

        else:
            raise CrossoverError(f"{name}: неподдерживаемый формат после скрещивания")

        # Валидация через валидатор
        is_valid, err = self.validator.validate_architecture(child)
        if not is_valid:
            logger.warning(f"{name}: невалидная архитектура после скрещивания: {err}")
            if self.config.repair_invalid:
                child = self._repair(child)
            else:
                raise CrossoverError(f"{name}: невалидная архитектура: {err}")
        return child

    def _extend_seq(self, arch: List[Dict]) -> List[Dict]:
        """Дополняет последовательную архитектуру до min_architecture_length, вставляя простые Dense"""
        while len(arch) < self.config.min_architecture_length:
            arch.append({"layer": "Dense", "units": 32, "activation": "relu"})
        return arch

    def _repair(self, arch: Union[List[Dict], Dict]) -> Union[List[Dict], Dict]:
        """
        Пытается исправить невалидную архитектуру простыми методами:
        - для последовательных: удаляет последние слои до момента валидности
        - для параллельных: удаляет невалидные ветви, затем гарантирует как в _postprocess
        """
        logger.debug("Попытка ремонта архитектуры")
        # Последовательная
        if isinstance(arch, list):
            arch_copy = copy.deepcopy(arch)
            while arch_copy:
                is_valid, _ = self.validator.validate_architecture(arch_copy)
                if is_valid:
                    return arch_copy
                arch_copy.pop()  # удаляем последний слой
            raise CrossoverError("Не удалось отремонтировать последовательную архитектуру")

        # Параллельная
        elif isinstance(arch, dict) and "parallel" in arch:
            arch_copy = {"parallel": {}}
            for name, branch in arch["parallel"].items():
                is_valid, _ = self.validator.validate_architecture(branch)
                if is_valid and branch:
                    arch_copy["parallel"][name] = branch
            if not arch_copy["parallel"]:
                raise CrossoverError("Не удалось отремонтировать параллельную архитектуру (нет валидных ветвей)")
            return arch_copy

        else:
            raise CrossoverError("Неподдерживаемый формат для ремонта архитектуры")
