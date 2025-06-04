import random
import math
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

# Настройка логирования
logger = logging.getLogger(__name__)

@dataclass
class ArchitectureConstraints:
    """Ограничения для генерации архитектур"""
    max_layers: int = 10
    max_filters: int = 512
    max_units: int = 512
    max_kernel_size: int = 15
    min_sequence_length: int = 10
    max_parallel_branches: int = 4
    
@dataclass 
class LayerConfig:
    """Базовый класс для конфигурации слоя"""
    layer_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Валидация параметров слоя"""
        return True

class ArchitectureValidator:
    """Валидатор архитектур для предотвращения ошибок"""
    
    def __init__(self, constraints: ArchitectureConstraints):
        self.constraints = constraints
        
    def validate_architecture(self, arch: Union[List, Dict]) -> tuple[bool, str]:
        """
        Валидирует архитектуру на корректность
        
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            if isinstance(arch, dict) and 'parallel' in arch:
                return self._validate_parallel_architecture(arch)
            elif isinstance(arch, list):
                return self._validate_sequential_architecture(arch)
            else:
                return False, "Неизвестный тип архитектуры"
        except Exception as e:
            logger.error(f"Ошибка валидации архитектуры: {e}")
            return False, str(e)
    
    def _validate_sequential_architecture(self, arch: List) -> tuple[bool, str]:
        """Валидация последовательной архитектуры"""
        if len(arch) > self.constraints.max_layers:
            return False, f"Слишком много слоев: {len(arch)} > {self.constraints.max_layers}"
            
        # Проверяем совместимость слоев
        prev_output_shape = None
        for i, layer in enumerate(arch):
            is_valid, error = self._validate_layer(layer, prev_output_shape)
            if not is_valid:
                return False, f"Слой {i}: {error}"
            prev_output_shape = self._get_output_shape(layer, prev_output_shape)
            
        return True, ""
    
    def _validate_parallel_architecture(self, arch: Dict) -> tuple[bool, str]:
        """Валидация параллельной архитектуры"""
        if 'parallel' not in arch:
            return False, "Отсутствует ключ 'parallel'"
            
        branches = arch['parallel']
        if len(branches) > self.constraints.max_parallel_branches:
            return False, f"Слишком много параллельных ветвей: {len(branches)}"
            
        # Валидируем каждую ветвь
        for branch_name, branch_layers in branches.items():
            is_valid, error = self._validate_sequential_architecture(branch_layers)
            if not is_valid:
                return False, f"Ветвь {branch_name}: {error}"
                
        return True, ""
    
    def _validate_layer(self, layer: Dict, input_shape: Optional[tuple]) -> tuple[bool, str]:
        """Валидация отдельного слоя"""
        if 'layer' not in layer:
            return False, "Отсутствует тип слоя"
            
        layer_type = layer['layer']
        
        # Валидация Conv1D
        if layer_type == 'Conv1D':
            if layer.get('filters', 0) > self.constraints.max_filters:
                return False, f"Слишком много фильтров: {layer['filters']}"
            if layer.get('kernel_size', 0) > self.constraints.max_kernel_size:
                return False, f"Слишком большое ядро: {layer['kernel_size']}"
                
        # Валидация RNN/GRU
        elif layer_type in ['GRU', 'RNN', 'LSTM']:
            if layer.get('units', 0) > self.constraints.max_units:
                return False, f"Слишком много единиц: {layer['units']}"
                
        # Валидация Dense
        elif layer_type == 'Dense':
            if layer.get('units', 0) > self.constraints.max_units:
                return False, f"Слишком много единиц: {layer['units']}"
                
        return True, ""
    
    def _get_output_shape(self, layer: Dict, input_shape: Optional[tuple]) -> Optional[tuple]:
        """Примерный расчет выходной формы слоя"""
        # Упрощенная логика для основных типов слоев
        if input_shape is None:
            return None
            
        layer_type = layer['layer']
        
        if layer_type == 'Conv1D':
            # Упрощенный расчет
            return input_shape  # В реальности зависит от padding, strides
        elif layer_type in ['GRU', 'RNN', 'LSTM']:
            if layer.get('return_sequences', False):
                return input_shape
            else:
                return (layer.get('units', 64),)  # Только последний выход
        elif layer_type == 'Dense':
            return (layer.get('units', 64),)
        elif layer_type in ['Flatten', 'GlobalAvgPool1D']:
            return (64,)  # Примерное значение
            
        return input_shape

class SmartArchitectureGenerator:
    """Улучшенный генератор архитектур с интеллектуальными эвристиками"""
    
    def __init__(self, constraints: ArchitectureConstraints = None):
        self.constraints = constraints or ArchitectureConstraints()
        self.validator = ArchitectureValidator(self.constraints)
        
    def random_dense_layer(self, 
                          units_choices: Optional[List[int]] = None,
                          context: str = "default") -> Dict[str, Any]:
        """
        Создает конфигурацию полносвязного слоя с учетом контекста
        
        Args:
            units_choices: Возможные размеры слоя
            context: Контекст использования ("output", "hidden", "feature_extraction")
        """
        if units_choices is None:
            if context == "output":
                units_choices = [1]  # Для регрессии
            elif context == "feature_extraction":
                units_choices = [128, 256, 512]
            else:
                units_choices = [32, 64, 128, 256]
        
        layer = {
            'layer': 'Dense',
            'units': random.choice(units_choices),
            'activation': self._choose_activation(context),
            'kernel_regularizer': random.choice(['L1', 'L2', 'L1L2', None])
        }
        
        # Адаптивный коэффициент регуляризации
        if layer['kernel_regularizer']:
            if layer['units'] > 256:
                layer['coef_regularizer'] = random.choice([0.001, 0.01])  # Больше регуляризации
            else:
                layer['coef_regularizer'] = random.choice([0.0001, 0.001])
        
        # Адаптивный dropout
        if context == "output":
            layer['dropout'] = 0.0  # Без dropout на выходном слое
        else:
            layer['dropout'] = random.choice([0.0, 0.1, 0.2, 0.3])
            
        logger.debug(f"Создан Dense слой: {layer}")
        return layer
    
    def random_conv1d_block(self, 
                           input_length: Optional[int] = None,
                           layer_depth: int = 0) -> Dict[str, Any]:
        """
        Создает сверточный блок с учетом размера входа и глубины
        
        Args:
            input_length: Длина входной последовательности
            layer_depth: Глубина текущего слоя (для адаптации параметров)
        """
        # Адаптивный выбор количества фильтров
        if layer_depth == 0:
            filters_choices = [32, 64]  # Меньше фильтров в первом слое
        elif layer_depth < 3:
            filters_choices = [64, 128, 256]
        else:
            filters_choices = [128, 256, 512]  # Больше в глубоких слоях
            
        # Адаптивный размер ядра
        if input_length and input_length < 20:
            kernel_choices = [3, 5]  # Меньшие ядра для коротких последовательностей
        else:
            kernel_choices = [3, 5, 7, 9]
            
        block = {
            'layer': 'Conv1D',
            'filters': random.choice(filters_choices),
            'kernel_size': random.choice(kernel_choices),
            'strides': random.choice([1, 2]),
            'padding': random.choice(['valid', 'same']),
            'activation': random.choice(['relu', 'elu', 'swish']),  # Современные активации
            'batch_norm': random.choice([True, False]),
            'dropout': random.choice([0.0, 0.1, 0.2]),
        }
        
        # Интеллектуальный выбор пулинга
        if block['strides'] == 1:  # Если нет stride, можно добавить pooling
            block['pooling'] = random.choice([None, 'max2', 'max3'])
        else:
            block['pooling'] = None  # Избегаем двойного сокращения размерности
            
        logger.debug(f"Создан Conv1D блок (глубина {layer_depth}): {block}")
        return block
    
    def random_rnn_block(self, 
                        rnn_type: Optional[str] = None,
                        is_last_rnn: bool = False) -> Dict[str, Any]:
        """
        Создает RNN блок с улучшенными эвристиками
        
        Args:
            rnn_type: Тип RNN ('GRU', 'LSTM', 'RNN')
            is_last_rnn: Является ли последним RNN в последовательности
        """
        rnn_types = ['GRU', 'LSTM', 'RNN']
        # GRU и LSTM предпочтительнее простого RNN
        rnn_type = rnn_type or random.choices(
            rnn_types, 
            weights=[0.4, 0.4, 0.2]  # Предпочтение GRU и LSTM
        )[0]
        
        block = {
            'layer': rnn_type,
            'units': random.choice([32, 64, 128, 256]),
            'activation': 'tanh' if rnn_type != 'LSTM' else 'tanh',
            'return_sequences': not is_last_rnn,  # Умное управление последовательностями
            'dropout': random.choice([0.0, 0.1, 0.2]),
            'recurrent_dropout': random.choice([0.0, 0.1])
        }
        
        # Дополнительные параметры для LSTM
        if rnn_type == 'LSTM':
            block['recurrent_activation'] = 'sigmoid'
            block['use_bias'] = True
            
        logger.debug(f"Создан {rnn_type} блок (последний: {is_last_rnn}): {block}")
        return block
    
    def _choose_activation(self, context: str) -> str:
        """Умный выбор функции активации в зависимости от контекста"""
        if context == "output":
            return 'linear'  # Для регрессии
        elif context == "feature_extraction":
            return random.choice(['relu', 'elu', 'swish'])
        else:
            return random.choice(['relu', 'tanh', 'elu'])

# Улучшенные стратегии генерации архитектур

def skeleton_conv_dense(min_conv: int = 1, 
                       max_conv: int = 3, 
                       min_dense: int = 1, 
                       max_dense: int = 2,
                       input_length: Optional[int] = None) -> List[Dict]:
    """
    Генерирует CNN->Dense архитектуру с улучшенными эвристиками
    
    Args:
        min_conv, max_conv: Диапазон количества сверточных слоев  
        min_dense, max_dense: Диапазон количества полносвязных слоев
        input_length: Длина входной последовательности для адаптации
    """
    generator = SmartArchitectureGenerator()
    layers = []
    
    # Сверточная часть
    num_conv = random.randint(min_conv, max_conv)
    logger.info(f"Генерируется Conv-Dense архитектура: {num_conv} conv слоев")
    
    for i in range(num_conv):
        conv_layer = generator.random_conv1d_block(input_length, i)
        layers.append(conv_layer)
        
    # Глобальное усреднение перед Dense слоями
    layers.append({'layer': 'GlobalAvgPool1D'})
    
    # Полносвязная часть
    num_dense = random.randint(min_dense, max_dense)
    for i in range(num_dense):
        context = "output" if i == num_dense - 1 else "hidden"
        dense_layer = generator.random_dense_layer(context=context)
        layers.append(dense_layer)
    
    # Валидация
    validator = ArchitectureValidator(ArchitectureConstraints())
    is_valid, error = validator.validate_architecture(layers)
    if not is_valid:
        logger.warning(f"Сгенерированная архитектура невалидна: {error}")
        
    return layers

def skeleton_rnn(min_rnn: int = 1, 
                max_rnn: int = 3, 
                rnn_type: str = 'GRU',
                bidirectional_prob: float = 0.3) -> List[Dict]:
    """
    Генерирует RNN архитектуру с возможностью bidirectional слоев
    
    Args:
        min_rnn, max_rnn: Диапазон количества RNN слоев
        rnn_type: Тип RNN ('GRU', 'LSTM', 'RNN', 'mixed')
        bidirectional_prob: Вероятность создания bidirectional слоя
    """
    generator = SmartArchitectureGenerator()
    layers = []
    
    num_rnn = random.randint(min_rnn, max_rnn)
    logger.info(f"Генерируется RNN архитектура: {num_rnn} {rnn_type} слоев")
    
    for i in range(num_rnn):
        # Выбор типа RNN
        if rnn_type == 'mixed':
            current_rnn_type = random.choice(['GRU', 'LSTM', 'RNN'])
        else:
            current_rnn_type = rnn_type
            
        is_last = (i == num_rnn - 1)
        rnn_layer = generator.random_rnn_block(current_rnn_type, is_last)
        
        # Возможность создания bidirectional слоя
        if random.random() < bidirectional_prob and not is_last:
            rnn_layer['bidirectional'] = True
            logger.debug(f"Создан bidirectional {current_rnn_type} слой")
            
        layers.append(rnn_layer)
    
    # Финальный Dense слой
    layers.append(generator.random_dense_layer(context="output"))
    
    return layers

def advanced_hybrid_architecture(complexity_level: str = "medium") -> List[Dict]:
    """
    Создает продвинутую гибридную архитектуру
    
    Args:
        complexity_level: "simple", "medium", "complex"
    """
    generator = SmartArchitectureGenerator()
    layers = []
    
    if complexity_level == "simple":
        # Простая комбинация: Conv -> RNN -> Dense
        layers.extend([
            generator.random_conv1d_block(layer_depth=0),
            generator.random_rnn_block(is_last_rnn=True),
            generator.random_dense_layer(context="hidden"),
            generator.random_dense_layer(context="output")
        ])
    elif complexity_level == "medium":
        # Средняя сложность: несколько Conv -> несколько RNN -> Dense
        for i in range(random.randint(1, 2)):
            layers.append(generator.random_conv1d_block(layer_depth=i))
        
        layers.append({'layer': 'GlobalAvgPool1D'})
        
        for i in range(random.randint(1, 2)):
            is_last_rnn = (i == 1)  # Предполагаем максимум 2 RNN
            layers.append(generator.random_rnn_block(is_last_rnn=is_last_rnn))
            
        layers.extend([
            generator.random_dense_layer(context="hidden"),
            generator.random_dense_layer(context="output")
        ])
    else:  # complex
        # Сложная архитектура с residual connections (упрощенная версия)
        for i in range(random.randint(2, 4)):
            layers.append(generator.random_conv1d_block(layer_depth=i))
            
        layers.append({'layer': 'GlobalAvgPool1D'})
        
        for i in range(random.randint(2, 3)):
            is_last_rnn = (i == 2)
            layers.append(generator.random_rnn_block(is_last_rnn=is_last_rnn))
            
        # Несколько Dense слоев с dropout
        for i in range(random.randint(1, 3)):
            context = "output" if i == 2 else "hidden"  # Последний как output
            layers.append(generator.random_dense_layer(context=context))
    
    logger.info(f"Создана {complexity_level} гибридная архитектура из {len(layers)} слоев")
    return layers

def dag_parallel_v2(num_branches: int = 2,
                   branch_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Улучшенная генерация параллельных архитектур
    
    Args:
        num_branches: Количество параллельных ветвей
        branch_types: Типы ветвей ('conv', 'rnn', 'hybrid')
    """
    if branch_types is None:
        branch_types = ['conv', 'rnn', 'hybrid']
    
    generator = SmartArchitectureGenerator()
    branches = {}
    
    logger.info(f"Генерируется параллельная архитектура с {num_branches} ветвями")
    
    for i in range(num_branches):
        branch_type = random.choice(branch_types)
        branch_name = f"branch_{i+1}_{branch_type}"
        
        if branch_type == 'conv':
            # Сверточная ветвь
            branch = []
            for j in range(random.randint(1, 3)):
                branch.append(generator.random_conv1d_block(layer_depth=j))
            branch.append({'layer': 'GlobalAvgPool1D'})
            
        elif branch_type == 'rnn':
            # RNN ветвь
            branch = []
            num_rnn = random.randint(1, 2)
            for j in range(num_rnn):
                is_last = (j == num_rnn - 1)
                branch.append(generator.random_rnn_block(is_last_rnn=is_last))
                
        else:  # hybrid
            # Гибридная ветвь
            branch = advanced_hybrid_architecture("simple")
            
        branches[branch_name] = branch
        logger.debug(f"Создана ветвь {branch_name} с {len(branch)} слоями")
    
    architecture = {'parallel': branches}
    
    # Валидация
    validator = ArchitectureValidator(ArchitectureConstraints())
    is_valid, error = validator.validate_architecture(architecture)
    if not is_valid:
        logger.warning(f"Параллельная архитектура невалидна: {error}")
    
    return architecture

# Остальные функции для совместимости
def block_randomized(min_blocks: int = 2, max_blocks: int = 5) -> List[Dict]:
    """Случайная генерация блоков (улучшенная версия)"""
    generator = SmartArchitectureGenerator()
    layers = []
    
    num_blocks = random.randint(min_blocks, max_blocks)
    logger.info(f"Генерируется случайная архитектура из {num_blocks} блоков")
    
    for i in range(num_blocks):
        block_type = random.choices(
            ['conv', 'rnn', 'dense'], 
            weights=[0.4, 0.3, 0.3]  # Предпочтение conv и rnn
        )[0]
        
        if block_type == 'conv':
            layers.append(generator.random_conv1d_block(layer_depth=i))
        elif block_type == 'rnn':
            is_last_rnn = (i == num_blocks - 1)
            layers.append(generator.random_rnn_block(is_last_rnn=is_last_rnn))
        else:  # dense
            context = "output" if i == num_blocks - 1 else "hidden"
            layers.append(generator.random_dense_layer(context=context))
    
    return layers

def micro_arch(complexity: str = "simple") -> List[Dict]:
    """Микро-архитектуры для быстрого старта"""
    return advanced_hybrid_architecture(complexity)

# Вспомогательные функции
def flatten_block() -> Dict[str, str]:
    """Возвращает конфигурацию Flatten слоя"""
    return {'layer': 'Flatten'}

def global_pool_block() -> Dict[str, str]:
    """Возвращает конфигурацию GlobalAveragePooling1D слоя"""
    return {'layer': 'GlobalAvgPool1D'}

# Функция для создания популяции
def generate_population(size: int = 20, 
                       strategies: Optional[List[str]] = None) -> List[Union[List, Dict]]:
    """
    Генерирует популяцию архитектур разных типов
    
    Args:
        size: Размер популяции
        strategies: Список стратегий для использования
    """
    if strategies is None:
        strategies = [
            'skeleton_conv_dense', 'skeleton_rnn', 'advanced_hybrid',
            'dag_parallel_v2', 'block_randomized'
        ]
    
    population = []
    logger.info(f"Генерируется популяция размером {size}")
    
    for i in range(size):
        strategy = random.choice(strategies)
        
        try:
            if strategy == 'skeleton_conv_dense':
                arch = skeleton_conv_dense()
            elif strategy == 'skeleton_rnn':
                arch = skeleton_rnn(rnn_type='mixed')
            elif strategy == 'advanced_hybrid':
                complexity = random.choice(['simple', 'medium', 'complex'])
                arch = advanced_hybrid_architecture(complexity)
            elif strategy == 'dag_parallel_v2':
                arch = dag_parallel_v2(random.randint(2, 3))
            else:  # block_randomized
                arch = block_randomized()
                
            population.append(arch)
            logger.debug(f"Архитектура {i+1}/{size} создана стратегией {strategy}")
            
        except Exception as e:
            logger.error(f"Ошибка создания архитектуры {i+1}: {e}")
            # Создаем простую резервную архитектуру
            fallback = skeleton_conv_dense(min_conv=1, max_conv=1, min_dense=1, max_dense=1)
            population.append(fallback)
    
    logger.info(f"Популяция создана: {len(population)} архитектур")
    return population

if __name__ == '__main__':
    # Тестирование улучшенного генератора
    logger.info("Тестирование улучшенного генератора архитектур")
    
    # Тест отдельных функций
    generator = SmartArchitectureGenerator()
    
    print("=== Тест Dense слоя ===")
    print(generator.random_dense_layer(context="output"))
    
    print("\n=== Тест Conv1D блока ===")
    print(generator.random_conv1d_block(input_length=50))
    
    print("\n=== Тест RNN блока ===")
    print(generator.random_rnn_block(rnn_type="LSTM"))
    
    # Тест стратегий
    print("\n=== Тест улучшенных стратегий ===")
    strategies = {
        'skeleton_conv_dense': lambda: skeleton_conv_dense(input_length=100),
        'skeleton_rnn': lambda: skeleton_rnn(rnn_type='mixed'),
        'advanced_hybrid': lambda: advanced_hybrid_architecture('medium'),
        'dag_parallel_v2': lambda: dag_parallel_v2(num_branches=3),
        'block_randomized': lambda: block_randomized()
    }
    
    for name, func in strategies.items():
        print(f"\n--- {name} ---")
        try:
            arch = func()
            print(f"Архитектура: {arch}")
            
            # Валидация
            validator = ArchitectureValidator(ArchitectureConstraints())
            is_valid, error = validator.validate_architecture(arch)
            print(f"Валидна: {is_valid}, Ошибка: {error}")
        except Exception as e:
            print(f"Ошибка: {e}")
    
    # Тест создания популяции
    print("\n=== Тест создания популяции ===")
    population = generate_population(size=5)
    print(f"Создано {len(population)} архитектур")