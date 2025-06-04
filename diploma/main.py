#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import History, ModelCheckpoint, EarlyStopping
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import plot_model
import multiprocessing

# Отключить GPU для избежания ошибок CUDA при multiprocessing
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Ensure proper multiprocessing start method
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

# Configure GPU in the main process
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

configure_gpu()

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from nas_genetic_alg import NasGeneticAlg
from builder import SmartModelBuilder
from optimizer import HyperParameterOptimizer

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Enable memory growth for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)  # Нужно будет создать этот модуль

def setup_logging(log_dir, log_file):
    """Setup logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file)
    
    # Reset the logging configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging with our desired settings
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def load_dataset(dataset_path):
    """Load and preprocess dataset"""
    df = pd.read_csv(dataset_path)
    
    # Предполагаем, что последний столбец - целевая переменная
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Разделение на train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred)
    }
    
    # Add MAPE if no zero values
    if not np.any(y_true == 0):
        metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    
    return metrics

def plot_training_history(history: History, save_dir: str):
    """Plot and save training metrics"""
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    for metric in ['loss', 'val_loss']:
        plt.plot(history.history[metric], label=metric)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Metrics plot
    plt.subplot(1, 2, 2)
    metrics = [m for m in history.history.keys() 
              if not m.startswith('val_') and m != 'loss']
    for metric in metrics:
        plt.plot(history.history[metric], label=f'train_{metric}')
        if f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
    plt.title('Model Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_vs_actual(model, X, y, save_dir: str):
    """Plot predicted vs actual values"""
    y_pred = model.predict(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs Actual')
    
    # Add R² score
    r2 = r2_score(y, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True)
    plot_path = os.path.join(save_dir, 'prediction_vs_actual.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(model, X, y, save_dir: str):
    """Plot error distribution"""
    y_pred = model.predict(X)
    errors = y_pred - y
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    
    # Add statistics
    plt.text(0.05, 0.95,
             f'Mean: {np.mean(errors):.4f}\n'
             f'Std: {np.std(errors):.4f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plot_path = os.path.join(save_dir, 'error_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def optimize_hyperparameters(model, data, config):
    """Optimize model hyperparameters"""
    optimizer = HyperParameterOptimizer(config.get('optimizer_config', {}))
    return optimizer.optimize(model, data)

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Architecture Search with Genetic Algorithm')
    
    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON configuration file')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to CSV dataset file')
    
    # Optional arguments
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Directory to save logs (default: ./logs/YYYYMMDD/)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Log file name (default: ga_run_YYYYMMDD_HHMMSS.log)')
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize hyperparameters of the best model')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots and visualizations')
    
    # Advanced options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Test split ratio (default: 0.2)')
    parser.add_argument('--save-checkpoints', action='store_true',
                        help='Save model checkpoints during training')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for the experiment (used in logging)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()

def modify_ga_parameters(config):
    """Interactive modification of GA parameters"""
    print("\nCurrent GA parameters:")
    for key, value in config['ga_config'].items():
        print(f"{key}: {value}")
    
    print("\nWhich parameter would you like to modify?")
    print("Available parameters: " + ", ".join(config['ga_config'].keys()))
    param = input("Enter parameter name (or 'done' to finish): ")
    
    while param != 'done':
        if param in config['ga_config']:
            try:
                value = input(f"Enter new value for {param}: ")
                # Convert to appropriate type
                old_value = config['ga_config'][param]
                if isinstance(old_value, int):
                    config['ga_config'][param] = int(value)
                elif isinstance(old_value, float):
                    config['ga_config'][param] = float(value)
                elif isinstance(old_value, bool):
                    config['ga_config'][param] = value.lower() == 'true'
                else:
                    config['ga_config'][param] = value
                print(f"Updated {param} to {config['ga_config'][param]}")
            except ValueError:
                print("Invalid value. No changes made.")
        else:
            print("Invalid parameter name.")
        
        param = input("\nEnter parameter name (or 'done' to finish): ")
    
    return config

def modify_training_parameters(config):
    """Interactive modification of training parameters"""
    print("\nCurrent training parameters:")
    for key, value in config['training_params'].items():
        if not isinstance(value, dict):
            print(f"{key}: {value}")
    
    print("\nWhich parameter would you like to modify?")
    print("Available parameters: " + ", ".join(
        [k for k, v in config['training_params'].items() if not isinstance(v, dict)]
    ))
    param = input("Enter parameter name (or 'done' to finish): ")
    
    while param != 'done':
        if param in config['training_params'] and not isinstance(config['training_params'][param], dict):
            try:
                value = input(f"Enter new value for {param}: ")
                # Convert to appropriate type
                old_value = config['training_params'][param]
                if isinstance(old_value, int):
                    config['training_params'][param] = int(value)
                elif isinstance(old_value, float):
                    config['training_params'][param] = float(value)
                elif isinstance(old_value, bool):
                    config['training_params'][param] = value.lower() == 'true'
                else:
                    config['training_params'][param] = value
                print(f"Updated {param} to {config['training_params'][param]}")
            except ValueError:
                print("Invalid value. No changes made.")
        else:
            print("Invalid parameter name.")
        
        param = input("\nEnter parameter name (or 'done' to finish): ")
    
    return config

def interactive_mode(config):
    """Interactive dialog with user"""
    while True:
        print("\nNeural Architecture Search Interactive Mode")
        print("==========================================")
        print("1. Proceed with current configuration")
        print("2. Modify GA parameters")
        print("3. Modify training parameters")
        print("4. Show current configuration")
        print("5. Save configuration to file")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            return config
        elif choice == '2':
            config = modify_ga_parameters(config)
        elif choice == '3':
            config = modify_training_parameters(config)
        elif choice == '4':
            print("\nCurrent configuration:")
            print(json.dumps(config, indent=2))
        elif choice == '5':
            filename = input("Enter filename to save configuration: ")
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration saved to {filename}")
        elif choice == '6':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")

def plot_genetic_progress(history: dict, save_dir: str):
    """Plot genetic algorithm progress metrics"""
    metrics = ['best_fitness', 'mean_fitness', 'diversity']
    plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(metrics, 1):
        if metric in history:
            plt.subplot(1, 3, i)
            plt.plot(history[metric], marker='o', markersize=2)
            plt.title(f'Evolution of {metric.replace("_", " ").title()}')
            plt.xlabel('Generation')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'genetic_progress.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_architecture_visualization(model, save_dir: str):
    """Create and save visualization of model architecture"""
    try:
        plt.figure(figsize=(15, 10))
        plot_model(
            model,
            to_file=os.path.join(save_dir, 'architecture.png'),
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=200
        )
        plt.close()
    except Exception as e:
        logging.warning(f"Could not create architecture visualization: {str(e)}")

def plot_correlation_matrix(X, y, save_dir: str):
    """Plot correlation matrix of features and target"""
    try:
        data = np.column_stack([X, y])
        cols = [f'Feature_{i+1}' for i in range(X.shape[1])] + ['Target']
        corr = pd.DataFrame(data, columns=cols).corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        plot_path = os.path.join(save_dir, 'correlation_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logging.warning(f"Could not create correlation matrix: {str(e)}")

def save_model_architecture_summary(model, save_dir: str):
    """Save model architecture summary to a text file"""
    summary_path = os.path.join(save_dir, 'model_summary.txt')
    
    # Save summary to file
    from contextlib import redirect_stdout
    with open(summary_path, 'w') as f:
        with redirect_stdout(f):
            model.summary()
            
        # Add additional information
        f.write("\nLayer Details:\n")
        f.write("-" * 80 + "\n")
        for layer in model.layers:
            f.write(f"\nLayer: {layer.name}\n")
            f.write(f"Type: {layer.__class__.__name__}\n")
            f.write(f"Config: {layer.get_config()}\n")
            f.write(f"Input shape: {layer.input_shape}\n")
            f.write(f"Output shape: {layer.output_shape}\n")
            f.write("-" * 80 + "\n")

def main():
    args = parse_args()
    
    # Setup default logging paths if not provided
    if not args.log_dir:
        args.log_dir = os.path.join(os.getcwd(), 'logs', datetime.now().strftime("%Y%m%d"))
    
    if not args.log_file:
        args.log_file = f'ga_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Ensure log directory exists
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.log_dir, args.log_file)
    logger.info("Starting Neural Architecture Search")
    logger.info(f"Logs will be written to: {os.path.join(args.log_dir, args.log_file)}")
    
    # Загрузка конфигурации
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Интерактивный режим
    if args.interactive:
        config = interactive_mode(config)
        logger.info("Configuration updated through interactive mode")
    
    # Настройка TensorFlow
    if config.get('hardware', {}).get('use_gpu', True):
        logger.info("Configuring GPU settings...")
        if config.get('hardware', {}).get('mixed_precision', True):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled")
    
    # Загрузка и подготовка данных
    logger.info(f"Loading dataset from {args.dataset}")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(args.dataset)
    
    # Вывод информации о датасете
    logger.info(f"Dataset shapes:")
    logger.info(f"Training: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Validation: X={X_val.shape}, y={y_val.shape}")
    logger.info(f"Test: X={X_test.shape}, y={y_test.shape}")
    
    # Инициализация и запуск генетического алгоритма
    ga = NasGeneticAlg(config)
    best_architecture, best_metrics = ga.run(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val)
    )
    
    # Построение лучшей модели
    logger.info("Building best model with found architecture")
    builder = SmartModelBuilder()
    best_model = builder.build_model_from_architecture(
        best_architecture,
        input_shape=(X_train.shape[1],)
    )
    
    # Компиляция модели
    logger.info("Compiling model")
    compile_params = config['compile_params']
    best_model.compile(**compile_params)
    
    # Вывод summary лучшей модели
    best_model.summary()
    
    # Оптимизация гиперпараметров если требуется
    if args.optimize:
        logger.info("Optimizing hyperparameters")
        best_model = optimize_hyperparameters(
            best_model,
            (X_train, y_train, X_val, y_val),
            config
        )
    
    # Обучение финальной модели
    logger.info("Training final model")
    
    # Подготовка параметров обучения
    training_params = config.get('training_params', {}).copy()
    
    # Создание колбэков из конфигурации
    callbacks = []
    if 'callbacks' in training_params:
        callback_configs = training_params.pop('callbacks')
        
        # Early Stopping
        if 'early_stopping' in callback_configs:
            es_config = callback_configs['early_stopping']
            callbacks.append(EarlyStopping(**es_config))
        
        # ReduceLROnPlateau
        if 'reduce_lr' in callback_configs:
            lr_config = callback_configs['reduce_lr']
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(**lr_config))
        
        # ModelCheckpoint если требуется
        if args.save_checkpoints:
            checkpoint_path = os.path.join(model_dir, 'checkpoints', 'model_{epoch:02d}.h5')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(
                checkpoint_path,
                save_best_only=True,
                monitor='val_loss'
            ))
    
    # Добавляем колбэки в параметры обучения
    if callbacks:
        training_params['callbacks'] = callbacks
    
    # Запуск обучения
    history = best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        **training_params
    )
    
    # Сохранение модели
    model_dir = os.path.join(args.log_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_model.h5')
    best_model.save(model_path)
    logger.info(f"Best model saved to {model_path}")
    
    # Оценка на тестовом наборе
    test_metrics = best_model.evaluate(X_test, y_test, return_dict=True)
    logger.info("Test metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Расчет дополнительных метрик
    y_pred_test = best_model.predict(X_test)
    detailed_metrics = calculate_metrics(y_test, y_pred_test)
    logger.info("\nDetailed metrics:")
    for metric, value in detailed_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Построение графиков если требуется
    if args.plot:
        plots_dir = os.path.join(args.log_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        logger.info("Generating plots and visualizations...")
        
        # График обучения
        plot_training_history(history, plots_dir)
        logger.info(f"Training history plot saved")
        
        # График предсказания
        plot_prediction_vs_actual(best_model, X_test, y_test, plots_dir)
        logger.info(f"Prediction vs Actual plot saved")
        
        # График распределения ошибок
        plot_error_distribution(best_model, X_test, y_test, plots_dir)
        logger.info(f"Error distribution plot saved")
        
        # График прогресса генетического алгоритма
        if hasattr(ga, 'history'):
            plot_genetic_progress(ga.history, plots_dir)
            logger.info(f"Genetic algorithm progress plot saved")
        
        # Визуализация архитектуры
        plot_architecture_visualization(best_model, plots_dir)
        logger.info(f"Model architecture visualization saved")
        
        # Корреляционная матрица
        plot_correlation_matrix(X_train, y_train, plots_dir)
        logger.info(f"Feature correlation matrix saved")
        
        # Сохранение детального описания архитектуры
        save_model_architecture_summary(best_model, plots_dir)
        logger.info(f"Detailed model architecture summary saved")
    
    # Сохранение конфигурации
    config_path = os.path.join(args.log_dir, 'final_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Final configuration saved to {config_path}")
    
    logger.info("\nNeural Architecture Search completed successfully!")
    logger.info(f"All results saved in: {args.log_dir}")
    logger.info("Summary of best model:")
    best_model.summary()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)