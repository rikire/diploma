# Neural Architecture Search with Genetic Algorithm

Система автоматического поиска архитектур нейронных сетей с использованием генетического алгоритма.

## Оглавление
- [Установка](#установка)
- [Структура проекта](#структура-проекта)
- [Конфигурация](#конфигурация)
- [Запуск](#запуск)
- [Интерактивный режим](#интерактивный-режим)
- [Оптимизация гиперпараметров](#оптимизация-гиперпараметров)
- [Визуализация результатов](#визуализация-результатов)
- [Структура логов](#структура-логов)
- [Советы по использованию](#советы-по-использованию)

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd diploma/new_version/diploma
``` 

2. Установите необходимые зависимости:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn optuna
```

## Структура проекта

```
diploma/
├── new_version/diploma/
│   ├── main.py           # Основной скрипт запуска
│   ├── config.json       # Конфигурационный файл
│   ├── nas_genetic_alg.py # Реализация генетического алгоритма
│   ├── builder.py        # Построение моделей
│   ├── evaluator.py      # Оценка моделей
│   ├── mutator.py        # Операторы мутации
│   ├── crossover.py      # Операторы скрещивания
│   ├── selecter.py       # Операторы селекции
│   └── optimizer.py      # Оптимизация гиперпараметров
```

## Конфигурация

Конфигурационный файл (config.json) содержит все настройки для:
- Генетического алгоритма
- Архитектуры нейросетей
- Обучения моделей
- Оптимизации гиперпараметров
- Аппаратных настроек

Пример конфигурации:
```json
{
    "ga_config": {
        "population_size": 30,
        "n_generations": 20,
        "mutation_rate": 0.2,
        "crossover_rate": 0.8,
        "tournament_size": 3,
        "elite_size": 2
    },
    "architecture_config": {
        "min_layers": 2,
        "max_layers": 10,
        "layer_types": ["Dense", "LSTM", "GRU", "Conv1D"],
        "activation_functions": ["relu", "tanh", "sigmoid"],
        "max_units": 256,
        "min_units": 16,
        "max_parallel_branches": 3,
        "use_skip_connections": true
    },
    "training_params": {
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.2,
        "callbacks": {
            "early_stopping": {
                "monitor": "val_loss",
                "patience": 10
            }
        }
    }
}
```

## Запуск

Основные параметры запуска:

```bash
python main.py \
  --config config.json \
  --dataset datasets/mackey_glass_dataset.csv \
  --experiment-name "experiment_1" \
  --optimize \
  --plot \
  --save-checkpoints \
  --seed 42
```

Параметры командной строки:
- `--config`: путь к конфигурационному файлу (обязательный)
- `--dataset`: путь к CSV файлу с данными (обязательный)
- `--log-dir`: директория для логов (по умолчанию 'logs')
- `--experiment-name`: имя эксперимента
- `--optimize`: включить оптимизацию гиперпараметров
- `--plot`: генерировать графики и визуализации
- `--interactive`: запустить в интерактивном режиме
- `--save-checkpoints`: сохранять чекпоинты моделей
- `--seed`: seed для воспроизводимости результатов
- `--debug`: включить подробное логирование

## Интерактивный режим

При запуске с флагом `--interactive` доступны следующие опции:
1. Изменение параметров ГА
2. Изменение параметров обучения
3. Просмотр текущей конфигурации
4. Сохранение конфигурации в файл

## Оптимизация гиперпараметров

При использовании флага `--optimize` система:
1. Находит лучшую архитектуру с помощью ГА
2. Оптимизирует гиперпараметры найденной модели:
   - Learning rate
   - Batch size
   - Optimizer
   - Dropout rate

## Визуализация результатов

При использовании флага `--plot` генерируются:
1. График обучения (loss и метрики)
2. График предсказаний vs реальных значений
3. Распределение ошибок
4. Прогресс генетического алгоритма
5. Визуализация архитектуры модели
6. Корреляционная матрица признаков
7. Детальное описание архитектуры

## Структура логов

```
logs/
├── experiment_name/
│   ├── models/
│   │   └── best_model.h5
│   ├── plots/
│   │   ├── training_history.png
│   │   ├── prediction_vs_actual.png
│   │   ├── error_distribution.png
│   │   ├── genetic_progress.png
│   │   ├── architecture.png
│   │   └── correlation_matrix.png
│   ├── final_config.json
│   └── run.log
```

## Советы по использованию

1. Начните с малого:
   - Небольшая популяция (10-20 особей)
   - Малое число поколений (5-10)
   - Простые архитектуры (2-5 слоев)

2. Используйте интерактивный режим для экспериментов с параметрами

3. Включайте оптимизацию гиперпараметров только после нахождения хорошей архитектуры

4. Сохраняйте конфигурации успешных экспериментов

5. Используйте seed для воспроизводимости результатов

6. Анализируйте графики для понимания процесса:
   - Genetic_progress.png покажет сходимость ГА
   - Training_history.png поможет выявить переобучение
   - Prediction_vs_actual.png и error_distribution.png покажут качество предсказаний

7. При работе с большими датасетами:
   - Увеличьте размер популяции
   - Используйте GPU
   - Включите mixed precision
   - Сохраняйте чекпоинты

8. Для улучшения результатов:
   - Экспериментируйте с параметрами ГА
   - Попробуйте различные типы слоев
   - Настройте параметры обучения
   - Используйте оптимизацию гиперпараметров

## Примеры использования

### Базовый запуск:
```bash
python main.py --config config.json --dataset data.csv
```

### Полный эксперимент:
```bash
python main.py \
  --config config.json \
  --dataset data.csv \
  --experiment-name "full_experiment" \
  --optimize \
  --plot \
  --save-checkpoints \
  --seed 42
```

### Интерактивный режим:
```bash
python main.py \
  --config config.json \
  --dataset data.csv \
  --interactive \
  --plot
```

### Отладка:
```bash
python main.py \
  --config config.json \
  --dataset data.csv \
  --debug \
  --experiment-name "debug_run"
```

## Пример работы с временным рядом Mackey-Glass

### Описание эксперимента

Датасет Mackey-Glass представляет собой хаотический временной ряд, часто используемый для тестирования алгоритмов прогнозирования. Наша задача - построить оптимальную гибридную архитектуру нейронной сети для предсказания следующего значения временного ряда.

### Подготовка

1. Убедитесь, что датасет находится в правильной директории:
```bash
ls datasets/mackey_glass_dataset.csv
```

2. Создайте конфигурационный файл `mackey_glass_config.json`:
```json
{
    "ga_config": {
        "population_size": 20,
        "n_generations": 15,
        "mutation_rate": 0.3,
        "crossover_rate": 0.7,
        "tournament_size": 3,
        "elite_size": 2
    },
    "architecture_config": {
        "min_layers": 2,
        "max_layers": 8,
        "layer_types": ["LSTM", "GRU", "Conv1D", "Dense"],
        "activation_functions": ["relu", "tanh"],
        "max_units": 128,
        "min_units": 16,
        "max_parallel_branches": 2,
        "use_skip_connections": true
    },
    "training_params": {
        "epochs": 100,
        "batch_size": 32,
        "validation_split": 0.2,
        "callbacks": {
            "early_stopping": {
                "monitor": "val_loss",
                "patience": 15,
                "restore_best_weights": true
            },
            "reduce_lr": {
                "monitor": "val_loss",
                "factor": 0.5,
                "patience": 5,
                "min_lr": 1e-6
            }
        }
    },
    "optimizer_config": {
        "n_trials": 20,
        "timeout": 3600,
        "param_ranges": {
            "learning_rate": [1e-4, 1e-2],
            "batch_size": [16, 128],
            "optimizer_name": ["adam", "adamw", "rmsprop"],
            "dropout_rate": [0.1, 0.5]
        }
    }
}
```

### Запуск эксперимента

1. Базовый поиск архитектуры:
```bash
python main.py \
  --config mackey_glass_config.json \
  --dataset datasets/mackey_glass_dataset.csv \
  --experiment-name "mackey_glass_basic" \
  --plot
```

2. Полный эксперимент с оптимизацией:
```bash
python main.py \
  --config mackey_glass_config.json \
  --dataset datasets/mackey_glass_dataset.csv \
  --experiment-name "mackey_glass_full" \
  --optimize \
  --plot \
  --save-checkpoints \
  --seed 42
```

### Анализ результатов

После завершения эксперимента, в директории `logs/mackey_glass_full/` будут доступны:

1. Графики в `plots/`:
   - `training_history.png` - процесс обучения лучшей модели
   - `prediction_vs_actual.png` - качество предсказаний
   - `error_distribution.png` - распределение ошибок
   - `genetic_progress.png` - эволюция популяции
   - `architecture.png` - визуализация найденной архитектуры

2. Метрики в `run.log`:
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - R² (коэффициент детерминации)
   - MAPE (Mean Absolute Percentage Error)

3. Модель в `models/best_model.h5`

### Ожидаемые результаты

При правильной настройке система должна найти архитектуру, способную предсказывать временной ряд Mackey-Glass с:
- RMSE < 0.01
- R² > 0.99
- MAE < 0.008

Типичная найденная архитектура обычно включает:
- Комбинацию LSTM/GRU слоев для обработки временной зависимости
- Conv1D слои для извлечения локальных паттернов
- Dense слои для финального отображения
- Skip-connections для улучшения потока градиентов

### Советы по улучшению результатов

1. Увеличьте размер популяции и число поколений:
```bash
python main.py \
  --config mackey_glass_config.json \
  --dataset datasets/mackey_glass_dataset.csv \
  --experiment-name "mackey_glass_large" \
  --optimize \
  --plot \
  --save-checkpoints \
  --seed 42
```
И измените в конфигурации:
```json
{
    "ga_config": {
        "population_size": 40,
        "n_generations": 30
    }
}
```

2. Попробуйте разные комбинации слоев в `architecture_config`

3. Настройте параметры обучения для лучшей сходимости:
   - Увеличьте `patience` в early stopping
   - Уменьшите `factor` в reduce_lr
   - Попробуйте различные оптимизаторы

4. Используйте интерактивный режим для экспериментов:
```bash
python /mnt/c/study/diploma/new_version/diploma/main.py \
  --config  /mnt/c/study/diploma/new_version/diploma/config.json \
  --dataset /mnt/c/study/diploma/datasets/mackey_glass_dataset.csv \
  --experiment-name "mackey_glass_interactive" \
  --interactive \
  --plot
```
