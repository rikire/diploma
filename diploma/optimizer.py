import optuna
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import logging
from typing import Tuple, Dict, Any
import tensorflow as tf

class HyperParameterOptimizer:
    """Оптимизатор гиперпараметров на основе Optuna"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_trials = config.get('n_trials', 20)
        self.timeout = config.get('timeout', 3600)  # 1 hour default
        self.logger = logging.getLogger(__name__)
        
        # Диапазоны поиска по умолчанию
        self.param_ranges = {
            'learning_rate': (1e-4, 1e-2),
            'batch_size': (16, 256),
            'optimizer_name': ['adam', 'adamw', 'rmsprop'],
            'dropout_rate': (0.1, 0.5)
        }
        
        # Обновляем диапазоны из конфига если они есть
        if 'param_ranges' in config:
            self.param_ranges.update(config['param_ranges'])
    
    def optimize(self, model: tf.keras.Model, 
                data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> tf.keras.Model:
        """
        Оптимизация гиперпараметров модели
        
        Args:
            model: Базовая модель для оптимизации
            data: Кортеж (X_train, y_train, X_val, y_val)
        
        Returns:
            Оптимизированная модель
        """
        X_train, y_train, X_val, y_val = data
        
        def objective(trial):
            # Копируем архитектуру модели
            current_model = tf.keras.models.clone_model(model)
            
            # Подбираем гиперпараметры
            lr = trial.suggest_float('learning_rate', 
                                   *self.param_ranges['learning_rate'], 
                                   log=True)
            
            batch_size = trial.suggest_int('batch_size', 
                                         *self.param_ranges['batch_size'],
                                         log=True)
            
            optimizer_name = trial.suggest_categorical('optimizer_name',
                                                     self.param_ranges['optimizer_name'])
            
            dropout_rate = trial.suggest_float('dropout_rate',
                                             *self.param_ranges['dropout_rate'])
            
            # Настраиваем оптимизатор
            if optimizer_name == 'adam':
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            elif optimizer_name == 'adamw':
                optimizer = tf.keras.optimizers.AdamW(learning_rate=lr)
            else:
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
            
            # Обновляем dropout rate во всех слоях
            for layer in current_model.layers:
                if isinstance(layer, tf.keras.layers.Dropout):
                    layer.rate = dropout_rate
            
            # Компилируем модель
            current_model.compile(
                optimizer=optimizer,
                loss=model.loss,
                metrics=model.metrics
            )
            
            # Callbacks для обучения
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
            
            # Обучаем модель
            history = current_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,  # Максимальное число эпох
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            return min(history.history['val_loss'])
        
        # Создаем исследование Optuna
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Запускаем оптимизацию
        self.logger.info("Starting hyperparameter optimization...")
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Получаем лучшие параметры
        best_params = study.best_params
        self.logger.info(f"Best hyperparameters: {best_params}")
        
        # Создаем финальную модель с лучшими параметрами
        final_model = tf.keras.models.clone_model(model)
        
        # Настраиваем оптимизатор
        if best_params['optimizer_name'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
        elif best_params['optimizer_name'] == 'adamw':
            optimizer = tf.keras.optimizers.AdamW(learning_rate=best_params['learning_rate'])
        else:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=best_params['learning_rate'])
        
        # Обновляем dropout
        for layer in final_model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                layer.rate = best_params['dropout_rate']
        
        # Компилируем финальную модель
        final_model.compile(
            optimizer=optimizer,
            loss=model.loss,
            metrics=model.metrics
        )
        
        # Обучаем финальную модель
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        final_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Больше эпох для финального обучения
            batch_size=best_params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return final_model
