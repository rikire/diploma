import random
import copy
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import os

# Импортируем валидатор и генератор из существующих модулей
from initializer import (
    ArchitectureValidator, ArchitectureConstraints,
    SmartArchitectureGenerator, generate_population
)


# Настройка логирования
logger = logging.getLogger(__name__)


@dataclass
class MutationConfig:
    """Конфигурация параметров мутации"""
    # Структурные мутации
    p_add_layer: float = 0.1
    p_remove_layer: float = 0.1
    p_swap_layers: float = 0.05
    p_shift_block: float = 0.05
    
    # Гиперпараметрические мутации
    p_activation: float = 0.3
    p_regularization: float = 0.2
    p_dropout: float = 0.25
    p_units_filters: float = 0.3
    p_conv_params: float = 0.2
    p_rnn_params: float = 0.2
    
    # Параллельные архитектуры
    p_branch_add: float = 0.1
    p_branch_remove: float = 0.1
    p_branch_merge: float = 0.05
    p_branch_split: float = 0.05
    
    # Ограничения
    min_layers: int = 2
    max_layers: int = 15
    max_shift_block_size: int = 3
    max_parallel_branches: int = 4  
    
    # Диапазоны значений для мутаций
    activation_pool: List[str] = None
    regularization_pool: List[str] = None
    
    def __post_init__(self):
        if self.activation_pool is None:
            self.activation_pool = ['relu', 'tanh', 'elu', 'swish', 'sigmoid', 'linear']
        if self.regularization_pool is None:
            self.regularization_pool = ['L1', 'L2', 'L1L2', None]


class MutationError(Exception):
    """Исключение для ошибок мутации"""
    pass


class ArchitectureMutator:
    """
    Класс для выполнения мутаций архитектур нейронных сетей
    """
    def __init__(self, config: Optional[MutationConfig] = None):
        self.config = config or MutationConfig()
        self.generator = SmartArchitectureGenerator()
        self.validator = ArchitectureValidator(ArchitectureConstraints())
        self.layer_pool = self._create_layer_pool()
        
    def _create_layer_pool(self) -> List[Dict[str, Any]]:
        """Создает пул доступных слоев для добавления"""
        pool = []
        
        # Различные типы слоев с разными параметрами
        conv_variants = [
            {'layer': 'Conv1D', 'filters': f, 'kernel_size': k, 'activation': 'relu'}
            for f in [32, 64, 128] for k in [3, 5, 7]
        ]
        
        rnn_variants = [
            {'layer': rnn_type, 'units': u, 'return_sequences': True}
            for rnn_type in ['GRU', 'LSTM'] for u in [32, 64, 128]
        ]
        
        dense_variants = [
            {'layer': 'Dense', 'units': u, 'activation': 'relu'}
            for u in [32, 64, 128, 256]
        ]
        
        pooling_variants = [
            {'layer': 'GlobalAvgPool1D'},
            {'layer': 'GlobalMaxPool1D'},
            {'layer': 'MaxPool1D'},
            {'layer': 'AvgPool1D'},
            {'layer': 'Flatten'}
        ]
        
        pool.extend(conv_variants)
        pool.extend(rnn_variants)
        pool.extend(dense_variants)
        pool.extend(pooling_variants)
        
        logger.debug(f"Создан пул слоев размером {len(pool)}")
        return pool
    
    def mutate_add_layer(self, arch: List[Dict], layer_pool: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Вставка случайного слоя из пула в случайную позицию
        
        Args:
            arch: Архитектура (список слоев)
            layer_pool: Пул доступных слоев для добавления
            
        Returns:
            Мутированная архитектура
            
        Raises:
            MutationError: Если добавление слоя приводит к невалидной архитектуре
        """
        if not arch:
            raise MutationError("Невозможно добавить слой в пустую архитектуру")
            
        if len(arch) >= self.config.max_layers:
            logger.warning(f"Архитектура уже содержит максимальное количество слоев ({self.config.max_layers})")
            return copy.deepcopy(arch)
        
        pool = layer_pool or self.layer_pool
        if not pool:
            raise MutationError("Пул слоев пуст")
        
        # Создаем копию архитектуры
        mutated = copy.deepcopy(arch)
        
        # Выбираем случайный слой из пула
        new_layer = copy.deepcopy(random.choice(pool))
        
        # Выбираем случайную позицию для вставки
        position = random.randint(0, len(mutated))
        
        # Интеллектуальная адаптация слоя к контексту
        new_layer = self._adapt_layer_to_context(new_layer, mutated, position)
        
        # Вставляем слой
        mutated.insert(position, new_layer)
        
        # Валидация результата
        is_valid, error = self.validator.validate_architecture(mutated)
        if not is_valid:
            logger.warning(f"Добавление слоя привело к невалидной архитектуре: {error}")
            return copy.deepcopy(arch)  # Возвращаем оригинал
        
        logger.info(f"ADD_LAYER: Добавлен {new_layer['layer']}" + 
           (f"({new_layer.get('units', new_layer.get('filters', ''))})" if new_layer.get('units') or new_layer.get('filters') else "") +
           f" в позицию {position}")
        logger.info(f"  Исходная: {arch}")
        logger.info(f"  Результат: {mutated}")
        
        return mutated
    
    def mutate_remove_layer(self, arch: List[Dict]) -> List[Dict]:
        """
        Удаление случайного слоя (если количество слоев > min_layers)
        
        Args:
            arch: Архитектура (список слоев)
            
        Returns:
            Мутированная архитектура
        """
        if len(arch) <= self.config.min_layers:
            logger.warning(f"Архитектура содержит минимальное количество слоев ({self.config.min_layers})")
            return copy.deepcopy(arch)
        
        # Создаем копию архитектуры
        mutated = copy.deepcopy(arch)
        
        # Выбираем случайный индекс для удаления
        remove_idx = random.randint(0, len(mutated) - 1)
        removed_layer = mutated.pop(remove_idx)
        
        # Валидация результата
        is_valid, error = self.validator.validate_architecture(mutated)
        if not is_valid:
            logger.warning(f"Удаление слоя привело к невалидной архитектуре: {error}")
            return copy.deepcopy(arch)  # Возвращаем оригинал
        
        logger.info(f"REMOVE_LAYER: Удален {removed_layer.get('layer', 'unknown')}" +
           (f"({removed_layer.get('units', removed_layer.get('filters', ''))})" if removed_layer.get('units') or removed_layer.get('filters') else "") +
           f" из позиции {remove_idx}")
        logger.info(f"  Исходная: {arch}")
        logger.info(f"  Результат: {mutated}")
        
        return mutated
    
    def mutate_swap_layers(self, arch: List[Dict]) -> List[Dict]:
        """
        Обмен местами двух случайных слоёв
        
        Args:
            arch: Архитектура (список слоев)
            
        Returns:
            Мутированная архитектура
        """
        if len(arch) < 2:
            logger.warning("Недостаточно слоев для обмена местами")
            return copy.deepcopy(arch)
        
        # Создаем копию архитектуры
        mutated = copy.deepcopy(arch)
        
        # Выбираем два разных индекса
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        
        # Меняем местами
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        # Валидация результата
        is_valid, error = self.validator.validate_architecture(mutated)
        if not is_valid:
            logger.warning(f"Обмен слоев привел к невалидной архитектуре: {error}")
            return copy.deepcopy(arch)  # Возвращаем оригинал
        
        layer1_info = f"{mutated[idx1].get('layer', 'unknown')}" + (f"({mutated[idx1].get('units', mutated[idx1].get('filters', ''))})" if mutated[idx1].get('units') or mutated[idx1].get('filters') else "")
        layer2_info = f"{mutated[idx2].get('layer', 'unknown')}" + (f"({mutated[idx2].get('units', mutated[idx2].get('filters', ''))})" if mutated[idx2].get('units') or mutated[idx2].get('filters') else "")
        logger.info(f"SWAP_LAYERS: Поменяны местами {layer1_info} <-> {layer2_info} (позиции {idx1} <-> {idx2})")
        logger.info(f"  Исходная: {arch}")
        logger.info(f"  Результат: {mutated}")

        return mutated
    
    def mutate_shift_block(self, arch: List[Dict]) -> List[Dict]:
        """
        Вырезка непрерывного блока слоёв и вставка в случайное место
        
        Args:
            arch: Архитектура (список слоев)
            
        Returns:
            Мутированная архитектура
        """
        if len(arch) < 3:  # Минимум 3 слоя для осмысленного сдвига блока
            logger.warning("Недостаточно слоев для сдвига блока")
            return copy.deepcopy(arch)
        
        # Создаем копию архитектуры
        mutated = copy.deepcopy(arch)
        
        # Определяем размер блока (1-3 слоя)
        max_block_size = min(self.config.max_shift_block_size, len(mutated) - 1)
        block_size = random.randint(1, max_block_size)
        
        # Выбираем начальную позицию блока
        start_idx = random.randint(0, len(mutated) - block_size)
        
        # Вырезаем блок
        block = mutated[start_idx:start_idx + block_size]
        del mutated[start_idx:start_idx + block_size]
        
        # Выбираем новую позицию для вставки
        new_pos = random.randint(0, len(mutated))
        
        # Вставляем блок
        for i, layer in enumerate(block):
            mutated.insert(new_pos + i, layer)
        
        # Валидация результата
        is_valid, error = self.validator.validate_architecture(mutated)
        if not is_valid:
            logger.warning(f"Сдвиг блока привел к невалидной архитектуре: {error}")
            return copy.deepcopy(arch)  # Возвращаем оригинал

        block_info = ' -> '.join([f"{l.get('layer', 'unknown')}" + (f"({l.get('units', l.get('filters', ''))})" if l.get('units') or l.get('filters') else "") for l in block])
        logger.info(f"SHIFT_BLOCK: Сдвинут блок [{block_info}] из позиции {start_idx} в позицию {new_pos}")
        logger.info(f"  Исходная: {arch}")
        logger.info(f"  Результат: {mutated}")        
        
        return mutated
    
    def mutate_activation(self, arch: List[Dict], p_activate_mutate: float = 0.3) -> List[Dict]:
        """
        Случайная смена функций активации по слоям
        
        Args:
            arch: Архитектура (список слоев)
            p_activate_mutate: Вероятность мутации для каждого слоя
            
        Returns:
            Мутированная архитектура
        """
        mutated = copy.deepcopy(arch)
        mutations_count = 0
        
        for layer in mutated:
            if random.random() < p_activate_mutate:
                if 'activation' in layer:
                    old_activation = layer['activation']
                    # Исключаем текущую активацию из выбора
                    available_activations = [a for a in self.config.activation_pool if a != old_activation]
                    if available_activations:
                        layer['activation'] = random.choice(available_activations)
                        mutations_count += 1
                        logger.debug(f"Изменена активация в слое {layer['layer']}: {old_activation} -> {layer['activation']}")
        
        logger.debug(f"Мутировано активаций: {mutations_count}")
        return mutated
    
    def mutate_regularization(self, arch: List[Dict], p_reg_mutate: float = 0.2) -> List[Dict]:
        """
        Изменение типа и коэффициента регуляризации в Dense-слоях
        
        Args:
            arch: Архитектура (список слоев)
            p_reg_mutate: Вероятность мутации для каждого Dense слоя
            
        Returns:
            Мутированная архитектура
        """
        mutated = copy.deepcopy(arch)
        mutations_count = 0
        
        for layer in mutated:
            if layer.get('layer') == 'Dense' and random.random() < p_reg_mutate:
                # Мутация типа регуляризации
                old_reg = layer.get('kernel_regularizer')
                layer['kernel_regularizer'] = random.choice(self.config.regularization_pool)
                
                # Если есть регуляризация, мутируем коэффициент
                if layer['kernel_regularizer']:
                    old_coef = layer.get('coef_regularizer', 0.01)
                    # Варьируем коэффициент в разумных пределах
                    new_coef = old_coef * random.uniform(0.1, 10.0)
                    layer['coef_regularizer'] = max(0.0001, min(new_coef, 0.1))  # Ограничиваем диапазон
                else:
                    layer.pop('coef_regularizer', None)  # Убираем коэффициент если нет регуляризации
                
                mutations_count += 1
                logger.debug(f"Изменена регуляризация в Dense слое: {old_reg} -> {layer['kernel_regularizer']}")
        
        logger.debug(f"Мутировано регуляризаций: {mutations_count}")
        return mutated
    
    def mutate_dropout(self, arch: List[Dict], p_dropout_mutate: float = 0.25) -> List[Dict]:
        """
        Изменение уровней dropout
        
        Args:
            arch: Архитектура (список слоев)
            p_dropout_mutate: Вероятность мутации для каждого слоя
            
        Returns:
            Мутированная архитектура
        """
        mutated = copy.deepcopy(arch)
        mutations_count = 0
        
        for layer in mutated:
            if random.random() < p_dropout_mutate:
                layer_type = layer.get('layer')
                
                # Для разных типов слоев разные типы dropout
                if layer_type in ['Dense', 'Conv1D']:
                    old_dropout = layer.get('dropout', 0.0)
                    # Новое значение в диапазоне [0.0, 0.5]
                    layer['dropout'] = random.uniform(0.0, 0.5)
                    mutations_count += 1
                    logger.debug(f"Изменен dropout в {layer_type}: {old_dropout:.3f} -> {layer['dropout']:.3f}")
                
                elif layer_type in ['GRU', 'LSTM', 'RNN']:
                    # Обычный dropout
                    if 'dropout' in layer or random.random() < 0.5:
                        old_dropout = layer.get('dropout', 0.0)
                        layer['dropout'] = random.uniform(0.0, 0.3)
                        logger.debug(f"Изменен dropout в {layer_type}: {old_dropout:.3f} -> {layer['dropout']:.3f}")
                    
                    # Recurrent dropout
                    if 'recurrent_dropout' in layer or random.random() < 0.5:
                        old_rec_dropout = layer.get('recurrent_dropout', 0.0)
                        layer['recurrent_dropout'] = random.uniform(0.0, 0.2)
                        logger.debug(f"Изменен recurrent_dropout в {layer_type}: {old_rec_dropout:.3f} -> {layer['recurrent_dropout']:.3f}")
                    
                    mutations_count += 1
        
        logger.debug(f"Мутировано dropout параметров: {mutations_count}")
        return mutated
    
    def mutate_units_filters(self, arch: List[Dict], p_size_mutate: float = 0.3) -> List[Dict]:
        """
        Изменение числа units или filters
        
        Args:
            arch: Архитектура (список слоев)
            p_size_mutate: Вероятность мутации для каждого слоя
            
        Returns:
            Мутированная архитектура
        """
        mutated = copy.deepcopy(arch)
        mutations_count = 0
        
        for layer in mutated:
            if random.random() < p_size_mutate:
                layer_type = layer.get('layer')
                
                if layer_type == 'Dense' and 'units' in layer:
                    old_units = layer['units']
                    # Мутируем в пределах разумного диапазона
                    multiplier = random.choice([0.5, 0.75, 1.25, 1.5, 2.0])
                    new_units = max(8, min(int(old_units * multiplier), 1024))  # Ограничиваем диапазон
                    layer['units'] = new_units
                    mutations_count += 1
                    logger.debug(f"Изменено units в Dense: {old_units} -> {new_units}")
                
                elif layer_type == 'Conv1D' and 'filters' in layer:
                    old_filters = layer['filters']
                    multiplier = random.choice([0.5, 0.75, 1.25, 1.5, 2.0])
                    new_filters = max(8, min(int(old_filters * multiplier), 512))
                    layer['filters'] = new_filters
                    mutations_count += 1
                    logger.debug(f"Изменено filters в Conv1D: {old_filters} -> {new_filters}")
                
                elif layer_type in ['GRU', 'LSTM', 'RNN'] and 'units' in layer:
                    old_units = layer['units']
                    multiplier = random.choice([0.5, 0.75, 1.25, 1.5, 2.0])
                    new_units = max(8, min(int(old_units * multiplier), 512))
                    layer['units'] = new_units
                    mutations_count += 1
                    logger.debug(f"Изменено units в {layer_type}: {old_units} -> {new_units}")
        
        logger.debug(f"Мутировано размеров: {mutations_count}")
        return mutated
    
    def mutate_conv_params(self, arch: List[Dict], p_conv_mutate: float = 0.2) -> List[Dict]:
        """
        Мутация параметров Conv1D (kernel_size, strides, padding, pooling, batch_norm)
        
        Args:
            arch: Архитектура (список слоев)
            p_conv_mutate: Вероятность мутации для каждого Conv1D слоя
            
        Returns:
            Мутированная архитектура
        """
        mutated = copy.deepcopy(arch)
        mutations_count = 0
        
        for layer in mutated:
            if layer.get('layer') == 'Conv1D' and random.random() < p_conv_mutate:
                mutations_made = []
                
                # Мутация kernel_size
                if 'kernel_size' in layer and random.random() < 0.4:
                    old_kernel = layer['kernel_size']
                    layer['kernel_size'] = random.choice([3, 5, 7, 9, 11])
                    mutations_made.append(f"kernel_size: {old_kernel} -> {layer['kernel_size']}")
                
                # Мутация strides
                if 'strides' in layer and random.random() < 0.3:
                    old_strides = layer['strides']
                    layer['strides'] = random.choice([1, 2])
                    mutations_made.append(f"strides: {old_strides} -> {layer['strides']}")
                
                # Мутация padding
                if 'padding' in layer and random.random() < 0.3:
                    old_padding = layer['padding']
                    layer['padding'] = random.choice(['same', 'valid'])
                    mutations_made.append(f"padding: {old_padding} -> {layer['padding']}")
                
                # Мутация pooling
                if random.random() < 0.3:
                    old_pooling = layer.get('pooling')
                    layer['pooling'] = random.choice([None, 'max2', 'max3', 'avg2', 'avg3'])
                    mutations_made.append(f"pooling: {old_pooling} -> {layer['pooling']}")
                
                # Мутация batch_norm
                if random.random() < 0.4:
                    old_bn = layer.get('batch_norm', False)
                    layer['batch_norm'] = not old_bn
                    mutations_made.append(f"batch_norm: {old_bn} -> {layer['batch_norm']}")
                
                if mutations_made:
                    mutations_count += 1
                    logger.debug(f"Мутированы параметры Conv1D: {', '.join(mutations_made)}")
        
        logger.debug(f"Мутировано Conv1D слоев: {mutations_count}")
        return mutated
    
    def mutate_rnn_params(self, arch: List[Dict], p_rnn_mutate: float = 0.2) -> List[Dict]:
        """
        Мутация рекуррентных параметров (return_sequences, bidirectional, recurrent_dropout)
        
        Args:
            arch: Архитектура (список слоев)
            p_rnn_mutate: Вероятность мутации для каждого RNN слоя
            
        Returns:
            Мутированная архитектура
        """
        mutated = copy.deepcopy(arch)
        mutations_count = 0
        
        for i, layer in enumerate(mutated):
            if layer.get('layer') in ['GRU', 'LSTM', 'RNN'] and random.random() < p_rnn_mutate:
                mutations_made = []
                
                # Мутация return_sequences (только если не последний RNN слой)
                next_is_rnn = (i < len(mutated) - 1 and 
                              mutated[i + 1].get('layer') in ['GRU', 'LSTM', 'RNN'])
                
                if 'return_sequences' in layer and random.random() < 0.4:
                    old_return_seq = layer['return_sequences']
                    # Если следующий слой тоже RNN, то должно быть True
                    if next_is_rnn:
                        layer['return_sequences'] = True
                    else:
                        layer['return_sequences'] = random.choice([True, False])
                    mutations_made.append(f"return_sequences: {old_return_seq} -> {layer['return_sequences']}")
                
                # Мутация bidirectional
                if random.random() < 0.2:
                    old_bidir = layer.get('bidirectional', False)
                    layer['bidirectional'] = not old_bidir
                    mutations_made.append(f"bidirectional: {old_bidir} -> {layer['bidirectional']}")
                
                # Мутация recurrent_dropout уже обрабатывается в mutate_dropout
                
                if mutations_made:
                    mutations_count += 1
                    logger.debug(f"Мутированы параметры {layer['layer']}: {', '.join(mutations_made)}")
        
        logger.debug(f"Мутировано RNN слоев: {mutations_count}")
        return mutated
    
    def mutate_hyperparams(self, arch: List[Dict], 
                          p_activation: float = 0.3,
                          p_regularization: float = 0.2,
                          p_dropout: float = 0.25,
                          p_units_filters: float = 0.3,
                          p_conv_params: float = 0.2,
                          p_rnn_params: float = 0.2) -> List[Dict]:
        """
        Последовательно применяет все гиперпараметрические мутации с заданными вероятностями
        
        Args:
            arch: Архитектура (список слоев)
            p_*: Вероятности для каждого типа мутации
            
        Returns:
            Мутированная архитектура
        """
        logger.debug("Начало гиперпараметрической мутации")
        
        mutated = copy.deepcopy(arch)
        
        # Применяем мутации последовательно
        mutated = self.mutate_activation(mutated, p_activation)
        mutated = self.mutate_regularization(mutated, p_regularization)  
        mutated = self.mutate_dropout(mutated, p_dropout)
        mutated = self.mutate_units_filters(mutated, p_units_filters)
        mutated = self.mutate_conv_params(mutated, p_conv_params)
        mutated = self.mutate_rnn_params(mutated, p_rnn_params)
        
        logger.debug("Завершена гиперпараметрическая мутация")
        return mutated
    
    def mutate_list(self, arch: List[Dict],
                   p_add: float = 0.1,
                   p_remove: float = 0.1, 
                   p_swap: float = 0.05,
                   p_shift: float = 0.05,
                   p_hyperparam: float = 0.5,
                   layer_pool: Optional[List[Dict]] = None,
                   min_depth: int = 2,
                   max_depth: int = 15,
                   hyperparam_config: Optional[Dict] = None) -> List[Dict]:
        """
        Высокоуровневая мутация списка. Сначала структурная мутация, затем гиперпараметрическая
        
        Args:
            arch: Архитектура (список слоев)
            p_*: Вероятности для каждого типа мутации
            layer_pool: Пул слоев для добавления
            min_depth, max_depth: Ограничения на глубину архитектуры
            hyperparam_config: Конфигурация гиперпараметрических мутаций
            
        Returns:
            Мутированная архитектура
        """
        logger.info(f"MUTATE_LIST: Начало мутации")
        logger.info(f"  Исходная архитектура: {arch}")
        
        # Обновляем ограничения
        original_min = self.config.min_layers
        original_max = self.config.max_layers
        self.config.min_layers = min_depth
        self.config.max_layers = max_depth
        
        try:
            mutated = copy.deepcopy(arch)
            mutations_applied = []
            
            # Структурные мутации (применяем только одну за раз)
            structural_mutations = []
            if len(mutated) < max_depth and random.random() < p_add:
                structural_mutations.append(('add', p_add))
            if len(mutated) > min_depth and random.random() < p_remove:
                structural_mutations.append(('remove', p_remove))
            if random.random() < p_swap:
                structural_mutations.append(('swap', p_swap))
            if random.random() < p_shift:
                structural_mutations.append(('shift', p_shift))
            
            # Применяем одну случайную структурную мутацию
            if structural_mutations:
                mutation_type, _ = random.choice(structural_mutations)
                
                if mutation_type == 'add':
                    mutated = self.mutate_add_layer(mutated, layer_pool)
                    mutations_applied.append('add_layer')
                elif mutation_type == 'remove':
                    mutated = self.mutate_remove_layer(mutated)
                    mutations_applied.append('remove_layer')
                elif mutation_type == 'swap':
                    mutated = self.mutate_swap_layers(mutated)
                    mutations_applied.append('swap_layers')
                elif mutation_type == 'shift':
                    mutated = self.mutate_shift_block(mutated)
                    mutations_applied.append('shift_block')
            
            # Гиперпараметрические мутации
            if random.random() < p_hyperparam:
                if hyperparam_config:
                    mutated = self.mutate_hyperparams(mutated, **hyperparam_config)
                else:
                    mutated = self.mutate_hyperparams(mutated)
                mutations_applied.append('hyperparams')
            
            logger.info(f"MUTATE_LIST: Применены мутации: {', '.join(mutations_applied) if mutations_applied else 'none'}")
            logger.info(f"  Результат: {mutated}")

            return mutated
            
        finally:
            # Восстанавливаем оригинальные ограничения
            self.config.min_layers = original_min
            self.config.max_layers = original_max
    
    def mutate_parallel(self, parent: Dict[str, Any],
                       p_branch_add: float = 0.1,
                       p_branch_remove: float = 0.1,
                       p_branch_merge: float = 0.05,
                       p_branch_split: float = 0.05,
                       p_hyperparam: float = 0.5,
                       hyperparam_config: Optional[Dict] = None,
                       list_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Мутация параллельной архитектуры (DAG)
        
        Args:
            parent: Параллельная архитектура вида {'parallel': {'branch1': [...], 'branch2': [...]}}
            p_branch_add: Вероятность добавления новой ветви
            p_branch_remove: Вероятность удаления ветви
            p_branch_merge: Вероятность слияния двух ветвей
            p_branch_split: Вероятность разделения ветви
            p_hyperparam: Вероятность гиперпараметрической мутации каждой ветви
            hyperparam_config: Конфигурация гиперпараметрических мутаций
            list_kwargs: Параметры для mutate_list
            
        Returns:
            Мутированная параллельная архитектура
            
        Raises:
            MutationError: Если мутация привела к невалидной архитектуре
        """
        if 'parallel' not in parent:
            raise MutationError("Архитектура не является параллельной")
        
        logger.info(f"MUTATE_PARALLEL: Начало мутации")
        logger.info(f"  Исходная архитектура: {parent}")
        
        mutated = copy.deepcopy(parent)
        branches = mutated['parallel']
        mutations_applied = []
        
        # Параметры по умолчанию для mutate_list
        default_list_kwargs = {
            'p_add': 0.1, 'p_remove': 0.1, 'p_swap': 0.05, 'p_shift': 0.05,
            'p_hyperparam': 0.3, 'min_depth': 2, 'max_depth': 10
        }
        if list_kwargs:
            default_list_kwargs.update(list_kwargs)
        
        try:
            # Структурные мутации ветвей (применяем одну случайную)
            structural_mutations = []
            
            if len(branches) < self.config.max_parallel_branches and random.random() < p_branch_add:
                structural_mutations.append('add')
            if len(branches) > 2 and random.random() < p_branch_remove:  # Минимум 2 ветви
                structural_mutations.append('remove')
            if len(branches) >= 2 and random.random() < p_branch_merge:
                structural_mutations.append('merge')
            if len(branches) >= 1 and random.random() < p_branch_split:
                structural_mutations.append('split')
            
            # Применяем одну структурную мутацию
            if structural_mutations:
                mutation_type = random.choice(structural_mutations)
                
                if mutation_type == 'add':
                    branches = self._add_branch(branches)
                    mutations_applied.append('add_branch')
                elif mutation_type == 'remove':
                    branches = self._remove_branch(branches)
                    mutations_applied.append('remove_branch')
                elif mutation_type == 'merge':
                    branches = self._merge_branches(branches)
                    mutations_applied.append('merge_branches')
                elif mutation_type == 'split':
                    branches = self._split_branch(branches)
                    mutations_applied.append('split_branch')
                
                mutated['parallel'] = branches
            
            # Мутация содержимого каждой ветви
            branch_mutations = 0
            for branch_name, branch_layers in branches.items():
                if random.random() < p_hyperparam:
                    try:
                        # Структурная мутация ветви
                        mutated_branch = self.mutate_list(branch_layers, **default_list_kwargs)
                        
                        # Дополнительная гиперпараметрическая мутация
                        if hyperparam_config and random.random() < 0.5:
                            mutated_branch = self.mutate_hyperparams(mutated_branch, **hyperparam_config)
                        
                        branches[branch_name] = mutated_branch
                        branch_mutations += 1
                        logger.debug(f"Мутирована ветвь {branch_name}")
                        
                    except Exception as e:
                        logger.warning(f"Ошибка мутации ветви {branch_name}: {e}")
                        # Оставляем ветвь без изменений при ошибке
            
            if branch_mutations > 0:
                mutations_applied.append(f'mutate_{branch_mutations}_branches')
            
            # Валидация результата
            is_valid, error = self.validator.validate_architecture(mutated)
            if not is_valid:
                logger.warning(f"Мутация параллельной архитектуры привела к невалидному результату: {error}")
                return copy.deepcopy(parent)  # Возвращаем оригинал
            
            logger.info(f"MUTATE_PARALLEL: Применены мутации: {', '.join(mutations_applied) if mutations_applied else 'none'}")
            logger.info(f"  Результат: {mutated}")

            return mutated
            
        except Exception as e:
            logger.error(f"Критическая ошибка при мутации параллельной архитектуры: {e}")
            return copy.deepcopy(parent)
    
    def _add_branch(self, branches: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Добавляет новую ветвь к параллельной архитектуре"""
        new_branches = copy.deepcopy(branches)
        
        # Генерируем новое имя ветви
        branch_count = len(new_branches)
        new_branch_name = f"branch_{branch_count + 1}_generated"
        
        # Создаем новую ветвь разными способами
        creation_methods = [
            lambda: self._create_simple_branch(),
            lambda: self._duplicate_random_branch(new_branches),
            lambda: self._create_complementary_branch(new_branches)
        ]
        
        method = random.choice(creation_methods)
        new_branch = method()
        new_branches[new_branch_name] = new_branch
        
        branch_info = ' -> '.join([f"{l.get('layer', 'unknown')}" + (f"({l.get('units', l.get('filters', ''))})" if l.get('units') or l.get('filters') else "") for l in new_branch])
        logger.info(f"ADD_BRANCH: Добавлена ветвь {new_branch_name}: [{branch_info}]")

        return new_branches
    
    def _remove_branch(self, branches: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Удаляет случайную ветвь из параллельной архитектуры"""
        if len(branches) <= 2:
            logger.warning("Нельзя удалить ветвь: минимум 2 ветви должно остаться")
            return branches
        
        new_branches = copy.deepcopy(branches)
        branch_to_remove = random.choice(list(new_branches.keys()))
        del new_branches[branch_to_remove]
        
        removed_branch_info = ' -> '.join([f"{l.get('layer', 'unknown')}" + (f"({l.get('units', l.get('filters', ''))})" if l.get('units') or l.get('filters') else "") for l in new_branches[branch_to_remove]] if branch_to_remove in branches else [])
        logger.info(f"REMOVE_BRANCH: Удалена ветвь {branch_to_remove}: [{removed_branch_info}]")

        return new_branches
    
    def _merge_branches(self, branches: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Объединяет две случайные ветви в одну"""
        if len(branches) < 2:
            logger.warning("Недостаточно ветвей для слияния")
            return branches
        
        new_branches = copy.deepcopy(branches)
        branch_names = list(new_branches.keys())
        branch1, branch2 = random.sample(branch_names, 2)
        
        # Создаем объединенную ветвь
        merged_branch = new_branches[branch1] + new_branches[branch2]
        
        # Добавляем промежуточный слой глобального пулинга если нужно
        if self._needs_pooling_layer(new_branches[branch1], new_branches[branch2]):
            pooling_layer = {'layer': 'GlobalAvgPool1D'}
            merged_branch.insert(len(new_branches[branch1]), pooling_layer)
        
        # Удаляем исходные ветви и добавляем объединенную
        del new_branches[branch1]
        del new_branches[branch2]
        new_branches[f"merged_{branch1}_{branch2}"] = merged_branch
        
        merged_info = ' -> '.join([f"{l.get('layer', 'unknown')}" + (f"({l.get('units', l.get('filters', ''))})" if l.get('units') or l.get('filters') else "") for l in merged_branch])
        logger.info(f"MERGE_BRANCHES: Объединены ветви {branch1} + {branch2} -> merged_{branch1}_{branch2}")
        logger.info(f"  Результат: [{merged_info}]")
        return new_branches
    
    def _split_branch(self, branches: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Разделяет случайную ветвь на две"""
        new_branches = copy.deepcopy(branches)
        branch_names = list(new_branches.keys())
        branch_to_split = random.choice(branch_names)
        branch_layers = new_branches[branch_to_split]
        
        if len(branch_layers) < 4:  # Минимум 4 слоя для разделения
            logger.debug(f"Ветвь {branch_to_split} слишком мала для разделения")
            return branches
        
        # Находим точку разделения
        split_point = random.randint(2, len(branch_layers) - 2)
        
        # Создаем две новые ветви
        branch1_layers = branch_layers[:split_point]
        branch2_layers = branch_layers[split_point:]
        
        # Удаляем исходную ветвь и добавляем две новые
        del new_branches[branch_to_split]
        new_branches[f"{branch_to_split}_part1"] = branch1_layers
        new_branches[f"{branch_to_split}_part2"] = branch2_layers
        
        part1_info = ' -> '.join([f"{l.get('layer', 'unknown')}" + (f"({l.get('units', l.get('filters', ''))})" if l.get('units') or l.get('filters') else "") for l in branch1_layers])
        part2_info = ' -> '.join([f"{l.get('layer', 'unknown')}" + (f"({l.get('units', l.get('filters', ''))})" if l.get('units') or l.get('filters') else "") for l in branch2_layers])
        logger.info(f"SPLIT_BRANCH: Разделена ветвь {branch_to_split} на позиции {split_point}")
        logger.info(f"  Часть 1: [{part1_info}]")
        logger.info(f"  Часть 2: [{part2_info}]")

        return new_branches
    
    def _create_simple_branch(self) -> List[Dict]:
        """Создает простую ветвь из 2-4 слоев"""
        branch_types = ['conv', 'rnn', 'dense']
        branch_type = random.choice(branch_types)
        
        if branch_type == 'conv':
            return [
                self.generator.random_conv1d_block(layer_depth=0),
                {'layer': 'GlobalAvgPool1D'}
            ]
        elif branch_type == 'rnn':
            return [
                self.generator.random_rnn_block(is_last_rnn=True)
            ]
        else:  # dense
            return [
                {'layer': 'Flatten'},
                self.generator.random_dense_layer(context="hidden")
            ]
    
    def _duplicate_random_branch(self, branches: Dict[str, List[Dict]]) -> List[Dict]:
        """Дублирует случайную существующую ветвь с небольшими изменениями"""
        if not branches:
            return self._create_simple_branch()
        
        source_branch = random.choice(list(branches.values()))
        duplicated = copy.deepcopy(source_branch)
        
        # Применяем небольшие мутации к дубликату
        duplicated = self.mutate_hyperparams(duplicated, 
                                           p_activation=0.2, p_units_filters=0.3, 
                                           p_dropout=0.2, p_regularization=0.1,
                                           p_conv_params=0.2, p_rnn_params=0.2)
        
        return duplicated
    
    def _create_complementary_branch(self, branches: Dict[str, List[Dict]]) -> List[Dict]:
        """Создает ветвь, дополняющую существующие (разного типа)"""
        if not branches:
            return self._create_simple_branch()
        
        # Анализируем типы существующих ветвей
        existing_types = set()
        for branch in branches.values():
            for layer in branch:
                layer_type = layer.get('layer', '')
                if layer_type in ['Conv1D']:
                    existing_types.add('conv')
                elif layer_type in ['GRU', 'LSTM', 'RNN']:
                    existing_types.add('rnn')
                elif layer_type == 'Dense':
                    existing_types.add('dense')
        
        # Создаем ветвь недостающего типа
        all_types = {'conv', 'rnn', 'dense'}
        missing_types = all_types - existing_types
        
        if missing_types:
            complement_type = random.choice(list(missing_types))
        else:
            complement_type = random.choice(list(all_types))
        
        if complement_type == 'conv':
            return [
                self.generator.random_conv1d_block(layer_depth=0),
                self.generator.random_conv1d_block(layer_depth=1),
                {'layer': 'GlobalAvgPool1D'}
            ]
        elif complement_type == 'rnn':
            return [
                self.generator.random_rnn_block(is_last_rnn=False),
                self.generator.random_rnn_block(is_last_rnn=True)
            ]
        else:  # dense
            return [
                {'layer': 'Flatten'},
                self.generator.random_dense_layer(context="hidden"),
                self.generator.random_dense_layer(context="hidden")
            ]
    
    def _needs_pooling_layer(self, branch1: List[Dict], branch2: List[Dict]) -> bool:
        """Определяет, нужен ли промежуточный слой пулинга при слиянии ветвей"""
        # Простая эвристика: если первая ветвь заканчивается Conv1D, а вторая начинается с Dense
        if not branch1 or not branch2:
            return False
        
        last_layer_branch1 = branch1[-1].get('layer', '')
        first_layer_branch2 = branch2[0].get('layer', '')
        
        return (last_layer_branch1 == 'Conv1D' and first_layer_branch2 == 'Dense')
    
    def _adapt_layer_to_context(self, layer: Dict[str, Any], 
                               arch: List[Dict], position: int) -> Dict[str, Any]:
        """
        Адаптирует добавляемый слой к контексту архитектуры
        
        Args:
            layer: Слой для адаптации
            arch: Текущая архитектура
            position: Позиция вставки
            
        Returns:
            Адаптированный слой
        """
        adapted = copy.deepcopy(layer)
        layer_type = adapted.get('layer')
        
        # Контекстная адаптация для RNN слоев
        if layer_type in ['GRU', 'LSTM', 'RNN']:
            # Если после этого слоя будет еще один RNN, нужно return_sequences=True
            next_is_rnn = (position < len(arch) and 
                          arch[position].get('layer') in ['GRU', 'LSTM', 'RNN'])
            adapted['return_sequences'] = next_is_rnn
        
        # Контекстная адаптация для Dense слоев
        elif layer_type == 'Dense':
            # Если это последний слой, делаем его выходным
            if position == len(arch):
                adapted['units'] = 1
                adapted['activation'] = 'linear'
                adapted['dropout'] = 0.0
        
        # Адаптация Conv1D для коротких последовательностей
        elif layer_type == 'Conv1D':
            # Проверяем наличие предыдущих сверточных слоев
            conv_count = sum(1 for l in arch[:position] if l.get('layer') == 'Conv1D')
            if conv_count >= 2:
                # Уменьшаем kernel_size для глубоких сверток
                adapted['kernel_size'] = min(adapted.get('kernel_size', 3), 3)
        
        return adapted
    
    def mutate(self, parent: Union[List[Dict], Dict[str, Any]], 
              p_hyperparam: float = 0.5, 
              hyperparam_config: Optional[Dict] = None,
              **kwargs) -> Union[List[Dict], Dict[str, Any]]:
        """
        Универсальная функция мутации - вызывает соответствующий метод в зависимости от типа архитектуры
        
        Args:
            parent: Архитектура для мутации (список слоев или параллельная архитектура)
            p_hyperparam: Вероятность гиперпараметрической мутации
            hyperparam_config: Конфигурация гиперпараметрических мутаций
            **kwargs: Дополнительные параметры для конкретных типов мутации
            
        Returns:
            Мутированная архитектура
            
        Raises:
            MutationError: Если тип архитектуры не поддерживается
        """
        logger.debug(f"Начало универсальной мутации (тип: {type(parent).__name__})")
        
        try:
            if isinstance(parent, dict) and 'parallel' in parent:
                # Параллельная архитектура
                parallel_kwargs = {
                    'p_branch_add': kwargs.get('p_branch_add', 0.1),
                    'p_branch_remove': kwargs.get('p_branch_remove', 0.1),
                    'p_branch_merge': kwargs.get('p_branch_merge', 0.05),
                    'p_branch_split': kwargs.get('p_branch_split', 0.05),
                    'p_hyperparam': p_hyperparam,
                    'hyperparam_config': hyperparam_config,
                    'list_kwargs': kwargs.get('list_kwargs', {})
                }
                return self.mutate_parallel(parent, **parallel_kwargs)
                
            elif isinstance(parent, list):
                # Последовательная архитектура
                list_kwargs = {
                    'p_add': kwargs.get('p_add', 0.1),
                    'p_remove': kwargs.get('p_remove', 0.1),
                    'p_swap': kwargs.get('p_swap', 0.05),
                    'p_shift': kwargs.get('p_shift', 0.05),
                    'p_hyperparam': p_hyperparam,
                    'layer_pool': kwargs.get('layer_pool'),
                    'min_depth': kwargs.get('min_depth', 2),
                    'max_depth': kwargs.get('max_depth', 15),
                    'hyperparam_config': hyperparam_config
                }
                return self.mutate_list(parent, **list_kwargs)
                
            else:
                raise MutationError(f"Неподдерживаемый тип архитектуры: {type(parent)}")
                
        except Exception as e:
            logger.error(f"Критическая ошибка в универсальной мутации: {e}")
            return copy.deepcopy(parent)

