import logging
import warnings
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, Dropout,
    GRU, LSTM, SimpleRNN, GlobalAveragePooling1D, GlobalMaxPooling1D,
    Flatten, Dense, Concatenate, MaxPooling1D, AveragePooling1D,
    Lambda, Add, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1, l2, l1_l2
import tensorflow.keras.backend as K

# Импортируем из initializer валидатор и генератор популяции
from initializer import ArchitectureValidator, ArchitectureConstraints, generate_population


logger = logging.getLogger(__name__)

@dataclass
class BuilderConfig:
    """Конфигурация для построения моделей"""
    min_sequence_length: int = 5
    max_sequence_length: int = 10000
    default_activation: str = 'relu'
    default_final_activation: str = 'linear'
    enable_batch_norm: bool = True
    enable_skip_connections: bool = True
    verbose_building: bool = False


class ArchitectureBuilderError(Exception):
    """Кастомное исключение для ошибок построения архитектуры"""
    pass


class SmartModelBuilder:
    """
    Улучшенный построитель моделей с интеллектуальной обработкой архитектур
    """
    
    def __init__(self, config: Optional[BuilderConfig] = None):
        self.config = config or BuilderConfig()
        self.layer_counter = 0
    
    def build_model_from_architecture(self,
                                      arch: Union[List[Dict], Dict],
                                      input_shape: Tuple[int, ...]) -> Model:
        """
        Главная функция построения модели из описания архитектуры
        
        Args:
            arch: Описание архитектуры (список слоев или словарь с параллельными ветвями)
            input_shape: Форма входного тензора (без batch dimension)
            
        Returns:
            tf.keras.Model: Построенная модель
            
        Raises:
            ArchitectureBuilderError: При невозможности построить архитектуру
        """
        logger.info(f"Начинаем построение модели. Input shape: {input_shape}")
        
        # 1) Базовая проверка input_shape
        self._validate_input_shape(input_shape)
        
        # 2) Проверка структуры арх-ры (не пустой список или валидный dict с ключом 'parallel')
        self._validate_architecture_structure(arch)
        
        # 3) Валидируем саму архитектуру через класса из initializer
        validator = ArchitectureValidator(ArchitectureConstraints())
        is_valid, err_msg = validator.validate_architecture(arch)
        if not is_valid:
            raise ArchitectureBuilderError(f"Архитектура не прошла валидацию: {err_msg}")
        
        # 4) Создаём input-слой
        input_layer = Input(shape=input_shape, name='input')
        self._log_tensor_shape(input_layer, "input")
        
        # 5) Строим блоки: либо параллельно, либо последовательно
        try:
            if isinstance(arch, dict) and 'parallel' in arch:
                output = self._build_parallel_architecture(input_layer, arch)
            else:
                output = self._build_sequential_architecture(input_layer, arch)
        except ArchitectureBuilderError:
            raise
        except Exception as e:
            # Ловим всё что осталось и завершаем
            tb = traceback.format_exc()
            logger.error(f"Непредвиденная ошибка при сборке блоков:\n{tb}")
            raise ArchitectureBuilderError(f"Ошибка при сборке блоков: {str(e)}")
        
        # 6) Финализация: если нет финального Dense(1), добавляем
        model = self._finalize_model(input_layer, output, arch)
        
        logger.info(f"Модель успешно построена. Параметры: {model.count_params():,}")
        return model
    
    def _validate_input_shape(self, input_shape: Tuple[int, ...]) -> None:
        """Валидация формы входного тензора"""
        if not input_shape or len(input_shape) < 1:
            raise ArchitectureBuilderError("Input shape должен содержать минимум одно измерение")
        
        if len(input_shape) == 2:  # (timesteps, features)
            timesteps, features = input_shape
            if timesteps and (timesteps < self.config.min_sequence_length or 
                              timesteps > self.config.max_sequence_length):
                logger.warning(f"Необычная длина последовательности: {timesteps}")
        logger.debug(f"Input shape валиден: {input_shape}")
    
    def _validate_architecture_structure(self, arch: Union[List[Dict], Dict]) -> None:
        """Проверка, что архитектура имеет ожидаемый формат (не пустые ветки и пр.)"""
        if isinstance(arch, dict):
            if 'parallel' not in arch:
                raise ArchitectureBuilderError("Параллельная архитектура должна содержать ключ 'parallel'")
            if not arch['parallel']:
                raise ArchitectureBuilderError("Параллельная архитектура не может быть пустой")
            # Проверим, что у каждой ветки есть непустой список
            for branch_name, branch_layers in arch['parallel'].items():
                if not isinstance(branch_layers, list) or len(branch_layers) == 0:
                    raise ArchitectureBuilderError(f"Ветвь '{branch_name}' пуста или не список")
        elif isinstance(arch, list):
            if len(arch) == 0:
                raise ArchitectureBuilderError("Последовательная архитектура не может быть пустой")
        else:
            raise ArchitectureBuilderError(f"Неподдерживаемый тип архитектуры: {type(arch)}")
    
    def _build_sequential_architecture(self,
                                       input_tensor: tf.Tensor,
                                       layers: List[Dict]) -> tf.Tensor:
        """
        Построение последовательной архитектуры
        
        Args:
            input_tensor: Входной тензор
            layers: Список описаний слоёв
            
        Returns:
            tf.Tensor: Выходной тензор
        """
        logger.info(f"Построение последовательной архитектуры из {len(layers)} слоёв")
        
        current_tensor = input_tensor
        skip_connections: Dict[int, tf.Tensor] = {}
        
        for i, layer_config in enumerate(layers):
            layer_type = layer_config.get('layer', '<unknown>')
            logger.debug(f"Обработка слоя {i+1}/{len(layers)}: {layer_type}")
            
            # сохраняем в skip_connections каждые 3 слоя
            if self.config.enable_skip_connections and (i % 3 == 0):
                skip_connections[i] = current_tensor
            
            try:
                current_tensor = self._build_layer(current_tensor, layer_config, f"seq_{i}")
            except ArchitectureBuilderError:
                raise
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Ошибка в слое {i} (тип {layer_type}): {str(e)}\n{tb}")
                raise ArchitectureBuilderError(f"Ошибка в слое {i} (тип {layer_type}): {str(e)}")
            
            # попытка skip-connection
            if (self.config.enable_skip_connections and (i - 3) in skip_connections):
                prev = skip_connections[i-3]
                current_tensor = self._try_add_skip_connection(current_tensor, prev, f"skip_{i}")
            
            self._log_tensor_shape(current_tensor, f"after_layer_{i}")
        
        return current_tensor
    
    def _build_parallel_architecture(self,
                                     input_tensor: tf.Tensor,
                                     arch: Dict) -> tf.Tensor:
        """
        Построение параллельной архитектуры с интеллектуальным объединением
        
        Args:
            input_tensor: Входной тензор
            arch: Словарь {'parallel': {branch_name: [слои], ...}}
            
        Returns:
            tf.Tensor: Объединённый выходной тензор
        """
        parallel_branches: Dict[str, List[Dict]] = arch['parallel']
        logger.info(f"Построение параллельной архитектуры с {len(parallel_branches)} ветвями")
        
        branch_outputs = []
        branch_info = []
        
        for branch_name, branch_layers in parallel_branches.items():
            if not isinstance(branch_layers, list) or len(branch_layers) == 0:
                raise ArchitectureBuilderError(f"Ветвь '{branch_name}' пуста или не список")
            
            logger.debug(f"Начало сборки ветви '{branch_name}' ({len(branch_layers)} слоёв)")
            try:
                out = self._build_sequential_architecture(input_tensor, branch_layers)
            except ArchitectureBuilderError:
                raise
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Ошибка в ветви '{branch_name}': {str(e)}\n{tb}")
                raise ArchitectureBuilderError(f"Ошибка в ветви '{branch_name}': {str(e)}")
            
            branch_outputs.append(out)
            branch_info.append({
                'name': branch_name,
                'output_tensor': out
            })
            logger.debug(f"Ветвь '{branch_name}' построена. Форма выхода: {out.shape}")
        
        # Умная нормализация и объединение
        merged = self._smart_merge_branches(branch_info)
        return merged
    
    def _smart_merge_branches(self, branch_info: List[Dict]) -> tf.Tensor:
        """
        Интеллектуальное объединение параллельных ветвей:
        - Если временные (time) размерности не совпадают, пытаемся свести через GlobalPooling
        - Если feature‐размерности не совпадают, вставляем адаптивный Dense
        - Далее, если две ветви — используем concat+add+mul, иначе — concat+attention
        """
        if len(branch_info) == 1:
            return branch_info[0]['output_tensor']
        
        logger.info("Выполняется _smart_merge_branches")
        
        # Шаг 1: приводим к «фиксированному» rank=2 (batch, features) через pooling, если есть rank>2.
        normalized: List[tf.Tensor] = []
        # Найдём «минимальную» feature‐размерность, к которой будем приводить
        target_feat_dim: Optional[int] = None
        
        # Сначала просто преобразуем все тензоры в 2D (batch, features)
        for info in branch_info:
            t: tf.Tensor = info['output_tensor']
            shape = t.shape  # (batch, time, features) или (batch, features)
            logger.debug(f"Нормализация ветви '{info['name']}': форма {shape}")
            
            # Если rank>2, сначала pooling по time‐размерности
            if len(shape) == 3:
                # Попробуем GlobalAveragePooling1D
                t = GlobalAveragePooling1D(name=f"gavgpool_{info['name']}")(t)
                logger.debug(f"После GAP форма: {t.shape}")
            elif len(shape) < 2:
                raise ArchitectureBuilderError(f"Неподдерживаемая форма ветви '{info['name']}': {shape}")
            
            # Теперь rank=2: (batch, features). Определяем target_feat_dim
            feat_dim = t.shape[-1]
            if target_feat_dim is None or (feat_dim is not None and feat_dim < target_feat_dim):
                target_feat_dim = int(feat_dim or target_feat_dim)
            
            normalized.append(t)
        
        if target_feat_dim is None:
            raise ArchitectureBuilderError("Не удалось определить целевую размерность для объединения ветвей")
        
        # Шаг 2: адаптируем каждую ветвь к target_feat_dim
        adapted: List[tf.Tensor] = []
        for idx, t in enumerate(normalized):
            feat_dim = int(t.shape[-1])
            if feat_dim != target_feat_dim:
                t = Dense(target_feat_dim, activation=self.config.default_activation,
                          name=f"adapt_dim_{branch_info[idx]['name']}")(t)
                logger.debug(f"Ветвь '{branch_info[idx]['name']}' адаптирована к {target_feat_dim}")
            adapted.append(t)
        
        # Шаг 3: собственно объединение
        if len(adapted) == 2:
            return self._merge_two_branches(adapted[0], adapted[1])
        else:
            return self._merge_multiple_branches(adapted)
    
    def _merge_two_branches(self, b1: tf.Tensor, b2: tf.Tensor) -> tf.Tensor:
        """Объединение двух ветвей через concat, add, multiply + проекция"""
        # concat по features
        concat = Concatenate(name='branch_concat')([b1, b2])
        # add и mul
        add = Add(name='branch_add')([b1, b2])
        mul = Multiply(name='branch_mul')([b1, b2])
        
        # Объединяем
        merged_all = Concatenate(name='all_merge_strategies')([concat, add, mul])
        
        # Проекция в более «сжатую» размерность: например, берем оставшуюся / 2
        dim = int(merged_all.shape[-1]) // 2
        dim = max(32, min(dim, 256))
        projected = Dense(dim, activation=self.config.default_activation,
                          name='merge_projection')(merged_all)
        return projected
    
    def _merge_multiple_branches(self, branches: List[tf.Tensor]) -> tf.Tensor:
        """Объединение >2 ветвей через concat + простой attention"""
        concat = Concatenate(name='multi_branch_concat')(branches)
        # attention‐веса: выход softmax по числу ветвей
        num = len(branches)
        attention = Dense(num, activation='softmax', name='branch_attention')(concat)
        
        weighted: List[tf.Tensor] = []
        for i, br in enumerate(branches):
            # забираем i‐й коэффициент
            w = Lambda(lambda x, idx=i: x[:, idx:idx+1], name=f"att_weight_{i}")(attention)
            wb = Multiply(name=f"weighted_branch_{i}")([br, w])
            weighted.append(wb)
        
        if len(weighted) > 1:
            added = Add(name='attended_sum')(weighted)
        else:
            added = weighted[0]
        
        final = Dense(128, activation=self.config.default_activation,
                      name='final_merge_dense')(added)
        return final
    
    def _build_layer(self,
                     input_tensor: tf.Tensor,
                     layer_config: Dict,
                     layer_prefix: str) -> tf.Tensor:
        """
        Построение отдельного слоя с учётом типов: Conv1D, RNN, Dense, Pooling, Flatten
        
        Args:
            input_tensor: входной тензор
            layer_config: словарь с конфигурацией ({'layer': 'Conv1D', ...})
            layer_prefix: префикс в имени (например, 'seq_3')
        
        Returns:
            tf.Tensor: выходной тензор
        """
        layer_type = layer_config.get('layer')
        if not layer_type:
            raise ArchitectureBuilderError("В конфигурации слоя отсутствует ключ 'layer'")
        
        self.layer_counter += 1
        layer_name = f"{layer_prefix}_{layer_type}_{self.layer_counter}"
        logger.debug(f"Построение слоя {layer_name}: {layer_config}")
        
        if layer_type == 'Conv1D':
            return self._build_conv1d_layer(input_tensor, layer_config, layer_name)
        elif layer_type in ['GRU', 'LSTM', 'RNN']:
            return self._build_rnn_layer(input_tensor, layer_config, layer_name)
        elif layer_type == 'Dense':
            return self._build_dense_layer(input_tensor, layer_config, layer_name)
        elif layer_type in ['GlobalAvgPool1D', 'GlobalMaxPool1D']:
            return self._build_global_pooling_layer(input_tensor, layer_config, layer_name)
        elif layer_type == 'Flatten':
            return self._build_flatten_layer(input_tensor, layer_name)
        elif layer_type in ['MaxPool1D', 'AvgPool1D']:
            return self._build_pooling_layer(input_tensor, layer_config, layer_name)
        else:
            raise ArchitectureBuilderError(f"Неподдерживаемый тип слоя: {layer_type}")
    
    def _build_conv1d_layer(self,
                            input_tensor: tf.Tensor,
                            cfg: Dict,
                            layer_name: str) -> tf.Tensor:
        """Построение Conv1D слоя с batch_norm, активацией, dropout и pooling"""
        # Убедимся, что у нас rank=3 (batch, time, features)
        current = self._ensure_3d_for_conv(input_tensor, layer_name)
        
        filters = cfg.get('filters', 32)
        kernel_size = cfg.get('kernel_size', 3)
        strides = cfg.get('strides', 1)
        padding = cfg.get('padding', 'same')
        activation = cfg.get('activation', self.config.default_activation)
        
        # Проверка: если padding='valid' и длина последовательности < kernel_size, переключаем
        seq_len = current.shape[1]
        if seq_len is not None and padding == 'valid' and seq_len < kernel_size:
            logger.warning(f"В Conv1D {layer_name}: kernel_size ({kernel_size}) > seq_len ({seq_len}), меняем padding на 'same'")
            padding = 'same'
        
        # Сама сверточная операция
        conv = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            use_bias=not cfg.get('batch_norm', False),
            name=f"{layer_name}_conv"
        )(current)
        
        # BatchNorm
        if cfg.get('batch_norm', False) and self.config.enable_batch_norm:
            conv = BatchNormalization(name=f"{layer_name}_bn")(conv)
        
        # Активация
        conv = Activation(activation, name=f"{layer_name}_act")(conv)
        
        # Dropout
        dr = cfg.get('dropout', 0.0)
        if dr and dr > 0.0:
            conv = Dropout(dr, name=f"{layer_name}_dropout")(conv)
        
        # Pooling (если указан)
        pooling = cfg.get('pooling')
        if pooling:
            # Заменили несуществующий _apply_pooling на вызов _build_pooling_layer
            conv = self._build_pooling_layer(conv, {'pooling': pooling}, layer_name)
        
        return conv
    
    def _build_rnn_layer(self,
                         input_tensor: tf.Tensor,
                         cfg: Dict,
                         layer_name: str) -> tf.Tensor:
        """Построение RNN-слоя (GRU/LSTM/SimpleRNN) с учётом bidirectional"""
        # Приводим rank=3 (batch, time, features)
        current = self._ensure_3d_for_rnn(input_tensor, layer_name)
        
        layer_type = cfg['layer']
        units = cfg.get('units', 64)
        activation = cfg.get('activation', 'tanh')
        return_sequences = cfg.get('return_sequences', False)
        dr = cfg.get('dropout', 0.0)
        rec_dr = cfg.get('recurrent_dropout', 0.0)
        bidirectional = cfg.get('bidirectional', False)
        
        if layer_type == 'GRU':
            rnn_cls = GRU
        elif layer_type == 'LSTM':
            rnn_cls = LSTM
        else:
            rnn_cls = SimpleRNN
        
        rnn_kwargs: Dict[str, Any] = {
            'units': units,
            'activation': activation,
            'return_sequences': return_sequences,
            'dropout': dr,
            'recurrent_dropout': rec_dr,
            'name': f"{layer_name}_rnn"
        }
        if layer_type == 'LSTM':
            rnn_kwargs['recurrent_activation'] = cfg.get('recurrent_activation', 'sigmoid')
        
        rnn_layer = rnn_cls(**rnn_kwargs)
        if bidirectional:
            from tensorflow.keras.layers import Bidirectional
            rnn_layer = Bidirectional(rnn_layer, name=f"{layer_name}_bidir")
        
        out = rnn_layer(current)
        return out
    
    def _build_dense_layer(self,
                           input_tensor: tf.Tensor,
                           cfg: Dict,
                           layer_name: str) -> tf.Tensor:
        """Построение Dense-слоя (Flatten + Dense + Dropout + регуляризация)"""
        current = self._flatten_if_needed(input_tensor, layer_name)
        
        units = cfg.get('units', 64)
        activation = cfg.get('activation', self.config.default_activation)
        reg = self._build_regularizer(cfg)
        
        dense = Dense(
            units=units,
            activation=activation,
            kernel_regularizer=reg,
            name=f"{layer_name}_dense"
        )(current)
        
        dr = cfg.get('dropout', 0.0)
        if dr and dr > 0.0:
            dense = Dropout(dr, name=f"{layer_name}_dropout")(dense)
        
        return dense
    
    def _build_global_pooling_layer(self,
                                    input_tensor: tf.Tensor,
                                    cfg: Dict,
                                    layer_name: str) -> tf.Tensor:
        """Построение GlobalAveragePooling1D или GlobalMaxPooling1D"""
        current = self._ensure_3d_for_conv(input_tensor, layer_name)
        layer_type = cfg.get('layer', 'GlobalAvgPool1D')
        
        if layer_type == 'GlobalAvgPool1D':
            return GlobalAveragePooling1D(name=f"{layer_name}_gap")(current)
        else:
            return GlobalMaxPooling1D(name=f"{layer_name}_gmp")(current)
    
    def _build_flatten_layer(self,
                             input_tensor: tf.Tensor,
                             layer_name: str) -> tf.Tensor:
        """Построение Flatten-слоя"""
        return Flatten(name=f"{layer_name}_flatten")(input_tensor)
    
    def _build_pooling_layer(self,
                             input_tensor: tf.Tensor,
                             cfg: Dict,
                             layer_name: str) -> tf.Tensor:
        """
        Построение обычного pooling слоя: MaxPool1D или AveragePooling1D.
        Ожидается cfg['pooling'] = 'maxX' или 'avgX', где X – размер окна.
        """
        current = self._ensure_3d_for_conv(input_tensor, layer_name)
        pooling_cfg = cfg.get('pooling')
        
        if not isinstance(pooling_cfg, str):
            logger.warning(f"В {layer_name}: неправильный параметр pooling={pooling_cfg}")
            return current
        
        if pooling_cfg.startswith('max'):
            try:
                size = int(pooling_cfg[len('max'):])
            except:
                logger.warning(f"В {layer_name}: не удалось распарсить размер для max-пулинга '{pooling_cfg}'")
                return current
            seq_len = current.shape[1]
            if seq_len is None or seq_len >= size:
                return MaxPooling1D(pool_size=size, name=f"{layer_name}_maxpool")(current)
            else:
                logger.warning(f"Пропускаем MaxPool{size} в {layer_name}, seq_len={seq_len} < {size}")
                return current
        
        elif pooling_cfg.startswith('avg'):
            try:
                size = int(pooling_cfg[len('avg'):])
            except:
                logger.warning(f"В {layer_name}: не удалось распарсить размер для avg-пулинга '{pooling_cfg}'")
                return current
            seq_len = current.shape[1]
            if seq_len is None or seq_len >= size:
                return AveragePooling1D(pool_size=size, name=f"{layer_name}_avgpool")(current)
            else:
                logger.warning(f"Пропускаем AvgPool{size} в {layer_name}, seq_len={seq_len} < {size}")
                return current
        
        else:
            logger.warning(f"Неподдерживаемый pooling '{pooling_cfg}' в {layer_name}")
            return current
    
    def _ensure_3d_for_conv(self,
                            tensor: tf.Tensor,
                            context: str) -> tf.Tensor:
        """Гарантирует rank=3 (batch, time, features) для Conv1D"""
        if len(tensor.shape) == 2:
            logger.debug(f"{context}: расширение 2D → 3D перед Conv1D")
            return Lambda(lambda x: K.expand_dims(x, axis=-1), name=f"{context}_expanddim")(tensor)
        elif len(tensor.shape) == 3:
            return tensor
        else:
            raise ArchitectureBuilderError(f"{context}: неподдерживаемая размерность для Conv1D: {tensor.shape}")
    
    def _ensure_3d_for_rnn(self,
                           tensor: tf.Tensor,
                           context: str) -> tf.Tensor:
        """Гарантирует rank=3 (batch, time, features) для RNN"""
        if len(tensor.shape) == 2:
            logger.debug(f"{context}: расширение 2D → 3D перед RNN")
            # предполагаем, что dimension 1 — features, добавляем «time=1»
            return Lambda(lambda x: K.expand_dims(x, axis=1), name=f"{context}_expandtime")(tensor)
        elif len(tensor.shape) == 3:
            return tensor
        else:
            raise ArchitectureBuilderError(f"{context}: неподдерживаемая размерность для RNN: {tensor.shape}")
    
    def _flatten_if_needed(self,
                           tensor: tf.Tensor,
                           context: str) -> tf.Tensor:
        """Если rank>2, применяем Flatten, иначе возвращаем tensor"""
        if len(tensor.shape) > 2:
            logger.debug(f"{context}: применение Flatten")
            return Flatten(name=f"{context}_flatten")(tensor)
        return tensor
    
    def _build_regularizer(self, cfg: Dict) -> Optional[Any]:
        """Создаёт regularizer по параметрам cfg"""
        t = cfg.get('kernel_regularizer')
        coef = cfg.get('coef_regularizer', 0.01)
        if t == 'L1':
            return l1(coef)
        elif t == 'L2':
            return l2(coef)
        elif t == 'L1L2':
            return l1_l2(coef)
        else:
            return None
    
    def _try_add_skip_connection(self,
                                 current: tf.Tensor,
                                 skip: tf.Tensor,
                                 context: str) -> tf.Tensor:
        """
        Пытаемся добавить skip‐connection: если формы совпадают, просто Add.
        Иначе, если rank=2, приводим feature‐размерности через Dense.
        В иных случаях – пропускаем skip.
        """
        try:
            # если точное совпадение форм (batch, ...), то Add
            if current.shape == skip.shape:
                logger.debug(f"{context}: добавляем прямое skip‐соединение (размеры совпадают)")
                return Add(name=f"{context}_add")([current, skip])
            
            # если rank=2 и не совпадают features: приводим через Dense→Add
            if len(current.shape) == 2 and len(skip.shape) == 2:
                dim_curr = int(current.shape[-1])
                dim_skip = int(skip.shape[-1])
                target = min(dim_curr, dim_skip)
                
                if dim_curr != target:
                    current = Dense(target, name=f"{context}_adapt_curr")(current)
                if dim_skip != target:
                    skip = Dense(target, name=f"{context}_adapt_skip")(skip)
                
                return Add(name=f"{context}_add_dense")([current, skip])
            
            # иначе — несовместимые формы, пропускаем skip
            logger.debug(f"{context}: несовместимые формы для skip ({current.shape} vs {skip.shape}), пропускаем")
            return current
        except Exception as e:
            logger.warning(f"{context}: не удалось добавить skip: {e}")
            return current
    
    def _finalize_model(self,
                        input_layer: tf.Tensor,
                        output_tensor: tf.Tensor,
                        arch: Union[List[Dict], Dict]) -> Model:
        """
        Финализация модели: если нет Dense(1) в конце, добавляем.
        """
        current = output_tensor
        
        needs_final_dense = True
        if isinstance(arch, list) and len(arch) > 0:
            last = arch[-1]
            if last.get('layer') == 'Dense' and last.get('units') == 1:
                needs_final_dense = False
        
        if needs_final_dense:
            logger.info("Добавляем финальный Dense(1)")
            current = self._flatten_if_needed(current, "final_flatten")
            current = Dense(1,
                            activation=self.config.default_final_activation,
                            name='final_output')(current)
        
        model = Model(inputs=input_layer, outputs=current, name='generated_model')
        logger.info(f"Модель финализирована: вход {input_layer.shape}, выход {current.shape}, params {model.count_params():,}")
        return model
    
    def _log_tensor_shape(self, tensor: tf.Tensor, context: str) -> None:
        """Логируем форму тензора, если включено verbose"""
        if self.config.verbose_building:
            logger.info(f"{context}: tensor.shape = {tensor.shape}")


if __name__ == "__main__":
    """
    Тестируем SmartModelBuilder на нескольких архитектурах из initializer.generate_population().
    Для каждой архитектуры:
      - Выводим её текстово
      - Пробуем собрать модель (input_shape=(100, 1) условно)
      - Если удаётся — печатаем summary и число параметров
      - Если нет — печатаем ошибку
    """
    import sys
    
    # Настраиваем уровень логирования под тест
    logger.setLevel(logging.INFO)
    
    builder = SmartModelBuilder(config=BuilderConfig(verbose_building=False))
    
    # Генерируем небольшую популяцию (10 архитектур) с помощью initializer
    try:
        pop = generate_population(size=10)
    except Exception as e:
        logger.error(f"Не удалось сгенерировать популяцию: {e}")
        sys.exit(1)
    
    for idx, arch in enumerate(pop):
        logger.info(f"\n=== Тестовая архитектура #{idx+1} ===")
        print(f"Архитектура #{idx+1}: {arch}\n")
        try:
            # Предположим, что у нас вход (100 timesteps, 3 feature)
            model = builder.build_model_from_architecture(arch, input_shape=(100, 3))
            print(f"Успешно собрано. Параметров: {model.count_params():,}\n")
            # Печатаем краткий summary (2 строки)
            model.summary()
        except ArchitectureBuilderError as be:
            print(f"Ошибка сборки: {be}\n")
        except Exception as e:
            print(f"Непредвиденная ошибка: {e}\n")
