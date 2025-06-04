import logging
import os
import copy
from typing import Dict, List, Any

# Импортируем модули
from mutator import ArchitectureMutator, MutationConfig, MutationError
from initializer import generate_population

# Создаём директорию для логов, если её ещё нет
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "tests_mutator.log")

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

def format_architecture_for_test(arch, name="Архитектура"):
    """Форматирует архитектуру для вывода в тестах"""
    if isinstance(arch, dict) and 'parallel' in arch:
        branches = arch['parallel']
        branch_info = []
        for branch_name, layers in branches.items():
            layer_names = [f"{l.get('layer', 'Unknown')}" + 
                          (f"({l.get('units', l.get('filters', ''))})" if l.get('units') or l.get('filters') else "")
                          for l in layers]
            branch_info.append(f"{branch_name}: [{' -> '.join(layer_names)}]")
        
        logger.info(f"  {name} - Параллельная ({len(branches)} ветвей):")
        for branch in branch_info:
            logger.info(f"    {branch}")
    
    elif isinstance(arch, list):
        layer_names = [f"{l.get('layer', 'Unknown')}" + 
                      (f"({l.get('units', l.get('filters', ''))})" if l.get('units') or l.get('filters') else "")
                      for l in arch]
        logger.info(f"  {name} - Последовательная ({len(arch)} слоев): [{' -> '.join(layer_names)}]")
    
    else:
        logger.info(f"  {name} - Неизвестный тип: {type(arch)}")

def test_mutations():
    """Тестирование различных типов мутаций"""
    logger.info("=== Запуск тестов мутаций ===")
    
    # Создаем тестовые архитектуры
    population = generate_population(size=5)
    mutator = ArchitectureMutator()
    

    logger.info("Детали созданной популяции:")
    for i, arch in enumerate(population):
        format_architecture_for_test(arch, f"Архитектура {i+1}")
            
    # Тестируем каждый тип мутации
    test_results = {}
    
    for i, arch in enumerate(population[:3]):  # Тестируем первые 3 архитектуры
        logger.info(f"--- Тестирование архитектуры {i+1} ---")
        arch_type = 'Параллельная' if isinstance(arch, dict) else 'Последовательная'
        logger.info(f"Тип: {arch_type}")
        
        if isinstance(arch, list):
            logger.info(f"Слоев: {len(arch)}")
            test_results[f"arch_{i+1}"] = test_sequential_mutations(arch, mutator)
        else:
            logger.info(f"Ветвей: {len(arch.get('parallel', {}))}")
            test_results[f"arch_{i+1}"] = test_parallel_mutations(arch, mutator)
    
    # Итоговая статистика
    logger.info("=== Итоговая статистика тестов ===")
    for arch_name, results in test_results.items():
        logger.info(f"{arch_name}:")
        for mutation_type, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"  {status} {mutation_type}")
    
    return test_results


def test_sequential_mutations(arch: List[Dict], mutator: ArchitectureMutator) -> Dict[str, bool]:
    """Тестирование мутаций для последовательной архитектуры"""
    results = {}
    
    logger.info("=== Тестирование последовательной архитектуры ===")
    format_architecture_for_test(arch, "Исходная архитектура")
    
    original_length = len(arch)
    
    # Тест добавления слоя
    try:
        mutated = mutator.mutate_add_layer(arch)
        results['add_layer'] = len(mutated) > original_length
        logger.info(f"  add_layer: {original_length} -> {len(mutated)} слоев")
    except Exception as e:
        results['add_layer'] = False
        logger.error(f"  add_layer: Ошибка - {e}")
    
    # Тест удаления слоя
    try:
        mutated = mutator.mutate_remove_layer(arch)
        results['remove_layer'] = len(mutated) <= original_length
        logger.info(f"  remove_layer: {original_length} -> {len(mutated)} слоев")
    except Exception as e:
        results['remove_layer'] = False
        logger.error(f"  remove_layer: Ошибка - {e}")
    
    # Тест обмена слоев
    try:
        mutated = mutator.mutate_swap_layers(arch)
        results['swap_layers'] = len(mutated) == original_length
        logger.info(f"  swap_layers: Длина сохранена - {len(mutated) == original_length}")
    except Exception as e:
        results['swap_layers'] = False
        logger.error(f"  swap_layers: Ошибка - {e}")
    
    # Тест сдвига блока
    try:
        mutated = mutator.mutate_shift_block(arch)
        results['shift_block'] = len(mutated) == original_length
        logger.info(f"  shift_block: Длина сохранена - {len(mutated) == original_length}")
    except Exception as e:
        results['shift_block'] = False
        logger.error(f"  shift_block: Ошибка - {e}")
    
    # Тест гиперпараметрических мутаций
    hyperparam_tests = {
        'activation': lambda: mutator.mutate_activation(arch, 0.5),
        'regularization': lambda: mutator.mutate_regularization(arch, 0.5),
        'dropout': lambda: mutator.mutate_dropout(arch, 0.5),
        'units_filters': lambda: mutator.mutate_units_filters(arch, 0.5),
        'conv_params': lambda: mutator.mutate_conv_params(arch, 0.5),
        'rnn_params': lambda: mutator.mutate_rnn_params(arch, 0.5)
    }
    
    for test_name, test_func in hyperparam_tests.items():
        try:
            mutated = test_func()
            results[test_name] = len(mutated) == original_length
            logger.info(f"  {test_name}: ✓")
        except Exception as e:
            results[test_name] = False
            logger.error(f"  {test_name}: Ошибка - {e}")
    
    # Тест комплексной мутации
    try:
        mutated = mutator.mutate_list(arch, p_hyperparam=0.8)
        results['mutate_list'] = isinstance(mutated, list)
        logger.info(f"  mutate_list: {original_length} -> {len(mutated)} слоев")
    except Exception as e:
        results['mutate_list'] = False
        logger.error(f"  mutate_list: Ошибка - {e}")
    
    return results


def test_parallel_mutations(arch: Dict[str, Any], mutator: ArchitectureMutator) -> Dict[str, bool]:
    """Тестирование мутаций для параллельной архитектуры"""
    results = {}

    logger.info("=== Тестирование параллельной архитектуры ===")
    format_architecture_for_test(arch, "Исходная архитектура")

    original_branches = len(arch.get('parallel', {}))
    
    # Тест мутации параллельной архитектуры
    try:
        mutated = mutator.mutate_parallel(arch, p_branch_add=0.3, p_branch_remove=0.2, p_hyperparam=0.5)
        results['mutate_parallel'] = 'parallel' in mutated
        new_branches = len(mutated.get('parallel', {}))
        logger.info(f"  mutate_parallel: {original_branches} -> {new_branches} ветвей")
    except Exception as e:
        results['mutate_parallel'] = False
        logger.error(f"  mutate_parallel: Ошибка - {e}")
    
    # Тест универсальной мутации
    try:
        mutated = mutator.mutate(arch, p_hyperparam=0.5)
        results['universal_mutate'] = 'parallel' in mutated
        logger.info(f"  universal_mutate: ✓")
    except Exception as e:
        results['universal_mutate'] = False
        logger.error(f"  universal_mutate: Ошибка - {e}")
    
    return results


def test_edge_cases():
    """Тестирование граничных случаев"""
    logger.info("=== Тестирование граничных случаев ===")
    
    mutator = ArchitectureMutator()
    
    # Тест с пустой архитектурой
    logger.info("Тест с пустой архитектурой:")
    try:
        empty_arch = []
        mutated = mutator.mutate_add_layer(empty_arch)
        logger.info(f"  Результат: {len(mutated)} слоев")
        if mutated:
            format_architecture_for_test(mutated, "Результат")
    except Exception as e:
        logger.error(f"  Ошибка: {e}")
    
    # Тест с минимальной архитектурой
    logger.info("Тест с минимальной архитектурой:")
    try:
        minimal_arch = [{'layer': 'Dense', 'units': 1, 'activation': 'linear'}]
        mutated = mutator.mutate_remove_layer(minimal_arch)
        format_architecture_for_test(minimal_arch, "Исходная минимальная")
        logger.info(f"  Исходная: {len(minimal_arch)}, Результат: {len(mutated)} слоев")
        format_architecture_for_test(mutated, "Результат")    
    except Exception as e:
        logger.error(f"  Ошибка: {e}")
    
    # Тест с максимальной архитектурой
    logger.info("Тест с максимальной архитектурой:")
    try:
        max_arch = [{'layer': 'Dense', 'units': 32} for _ in range(15)]  # Максимум слоев
        format_architecture_for_test(max_arch, "Исходная максимальная")
        mutated = mutator.mutate_add_layer(max_arch)
        logger.info(f"  Исходная: {len(max_arch)}, Результат: {len(mutated)} слоев")
        format_architecture_for_test(mutated, "Результат")
    except Exception as e:
        logger.error(f"  Ошибка: {e}")
    
    # Тест с некорректной архитектурой
    logger.info("Тест с некорректной архитектурой:")
    try:
        invalid_arch = [{'invalid': 'layer'}]
        format_architecture_for_test(invalid_arch, "Исходная некорректная")
        mutated = mutator.mutate(invalid_arch)
        logger.info(f"  Результат: {len(mutated)} слоев")
        if mutated:
            format_architecture_for_test(mutated, "Результат")

    except Exception as e:
        logger.error(f"  Ошибка: {e}")


def test_mutation_config():
    """Тестирование конфигурации мутаций"""
    logger.info("=== Тестирование конфигурации мутаций ===")
    
    # Тест стандартной конфигурации
    config = MutationConfig()
    logger.info(f"Стандартная конфигурация создана успешно")
    logger.info(f"  Активации: {len(config.activation_pool)}")
    logger.info(f"  Регуляризации: {len(config.regularization_pool)}")
    logger.info(f"  Минимум слоев: {config.min_layers}")
    logger.info(f"  Максимум слоев: {config.max_layers}")
    
    # Тест кастомной конфигурации
    custom_config = MutationConfig(
        p_add_layer=0.2,
        p_remove_layer=0.15,
        min_layers=3,
        max_layers=20,
        activation_pool=['relu', 'tanh']
    )
    logger.info(f"Кастомная конфигурация создана успешно")
    logger.info(f"  Вероятность добавления слоя: {custom_config.p_add_layer}")
    logger.info(f"  Активации: {custom_config.activation_pool}")
    
    # Тест мутатора с кастомной конфигурацией
    mutator = ArchitectureMutator(custom_config)
    logger.info("Мутатор с кастомной конфигурацией создан успешно")


def test_layer_pool():
    """Тестирование пула слоев"""
    logger.info("=== Тестирование пула слоев ===")
    
    mutator = ArchitectureMutator()
    layer_pool = mutator._create_layer_pool()
    
    logger.info("Примеры слоев из пула (первые 5):")
    for i, layer in enumerate(layer_pool[:5]):
        layer_str = f"{layer.get('layer', 'Unknown')}"
        if layer.get('units'):
            layer_str += f"(units={layer['units']})"
        elif layer.get('filters'):
            layer_str += f"(filters={layer['filters']})"
        if layer.get('activation'):
            layer_str += f", activation={layer['activation']}"
        logger.info(f"  {i+1}. {layer_str}")

    
    # Подсчитываем типы слоев
    layer_types = {}
    for layer in layer_pool:
        layer_type = layer.get('layer', 'unknown')
        layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
    
    logger.info("Распределение типов слоев:")
    for layer_type, count in layer_types.items():
        logger.info(f"  {layer_type}: {count}")


def test_detailed_mutations():
    """Детальное тестирование с выводом архитектур до и после мутаций"""
    logger.info("=== Детальное тестирование мутаций ===")
    
    # Создаем простую тестовую архитектуру
    test_arch = [
        {'layer': 'Conv1D', 'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
        {'layer': 'GRU', 'units': 64, 'return_sequences': True},
        {'layer': 'Dense', 'units': 128, 'activation': 'relu'},
        {'layer': 'Dense', 'units': 1, 'activation': 'linear'}
    ]
    
    logger.info("Тестовая архитектура для детального анализа:")
    format_architecture_for_test(test_arch, "Исходная")
    
    mutator = ArchitectureMutator()
    
    # Тестируем различные мутации
    mutations_to_test = [
        ('Добавление слоя', lambda: mutator.mutate_add_layer(test_arch)),
        ('Удаление слоя', lambda: mutator.mutate_remove_layer(test_arch)),
        ('Обмен слоев', lambda: mutator.mutate_swap_layers(test_arch)),
        ('Мутация активаций', lambda: mutator.mutate_activation(test_arch, 1.0)),
        ('Мутация размеров', lambda: mutator.mutate_units_filters(test_arch, 1.0)),
        ('Комплексная мутация', lambda: mutator.mutate_list(test_arch, p_hyperparam=0.8))
    ]
    
    for mutation_name, mutation_func in mutations_to_test:
        logger.info(f"\n--- {mutation_name} ---")
        try:
            mutated = mutation_func()
            format_architecture_for_test(mutated, "Результат мутации")
        except Exception as e:
            logger.error(f"Ошибка при выполнении {mutation_name}: {e}")


def run_all_tests():
    """Запуск всех тестов"""
    logger.info("Запуск тестирования модуля мутаций...")
    
    try:
        # Тест конфигурации
        test_mutation_config()
        
        # Тест пула слоев
        test_layer_pool()
        
        # Основные тесты
        test_results = test_mutations()

        # Детальные тесты        
        test_detailed_mutations()

        # Тесты граничных случаев
        test_edge_cases()
        
        logger.info("=== Тестирование завершено успешно ===")
        
        # Подсчет успешных тестов
        total_tests = 0
        passed_tests = 0
        for arch_results in test_results.values():
            for test_name, result in arch_results.items():
                total_tests += 1
                if result:
                    passed_tests += 1
        
        logger.info(f"Пройдено тестов: {passed_tests}/{total_tests}")
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        logger.info(f"Процент успешных тестов: {success_rate:.1f}%")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Критическая ошибка при тестировании: {e}")
        raise


if __name__ == '__main__':
    run_all_tests()