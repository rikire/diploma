import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from builder import SmartModelBuilder, BuilderConfig

# --- Конфиг ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
train_params = config['training_params']
compile_params = config['compile_params']

# --- Данные ---
DATA_PATH = 'datasets/mackey_glass_dataset.csv'
df = pd.read_csv(DATA_PATH)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# --- Архитектура ---
arch = [
    {"layer": "GRU", "units": 32, "activation": "tanh", "return_sequences": False, "dropout": 0.1, "recurrent_dropout": 0.1},
    {"layer": "Dense", "units": 1, "activation": "linear", "kernel_regularizer": None, "dropout": 0.0}
]

# --- Модель ---
input_shape = (X_train.shape[1], 1)
X_train_r = X_train.reshape((-1, X_train.shape[1], 1))
X_val_r = X_val.reshape((-1, X_val.shape[1], 1))
X_test_r = X_test.reshape((-1, X_test.shape[1], 1))

builder = SmartModelBuilder(config=BuilderConfig(verbose_building=False))
model = builder.build_model_from_architecture(arch, input_shape)
model.compile(**compile_params)
model.summary()

# --- Callbacks ---
cb = []
c = train_params.get('callbacks', {})
if 'early_stopping' in c:
    cb.append(callbacks.EarlyStopping(**c['early_stopping']))
if 'reduce_lr' in c:
    cb.append(callbacks.ReduceLROnPlateau(**c['reduce_lr']))

# --- Обучение ---
history = model.fit(
    X_train_r, y_train,
    validation_data=(X_val_r, y_val),
    epochs=train_params.get('epochs', 10),
    batch_size=train_params.get('batch_size', 32),
    callbacks=cb,
    verbose=2
)

# --- Оценка ---
metrics = model.evaluate(X_test_r, y_test, return_dict=True)
print('Test metrics:')
for k, v in metrics.items():
    print(f'{k}: {v:.4f}')

# --- Метрики ---
pred = model.predict(X_test_r).ravel()
y_test = y_test.ravel()
print('\nExtra metrics:')
print('MSE:', mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, pred)))
print('MAE:', mean_absolute_error(y_test, pred))
print('R2:', r2_score(y_test, pred))
print('Explained variance:', explained_variance_score(y_test, pred))
if not np.any(y_test == 0):
    print('MAPE:', mean_absolute_percentage_error(y_test, pred))

# --- Графики ---
PLOTS_DIR = 'logs/20250604/plots_mackey_glass'
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Training history')
plt.savefig(os.path.join(PLOTS_DIR, 'training_history_mackey.png'))
plt.close()

plt.figure(figsize=(8,6))
plt.scatter(y_test, pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Prediction vs Actual')
plt.savefig(os.path.join(PLOTS_DIR, 'prediction_vs_actual_mackey.png'))
plt.close()

errors = pred - y_test
plt.figure(figsize=(8,6))
sns.histplot(errors, kde=True)
plt.title('Error distribution')
plt.xlabel('Prediction error')
plt.savefig(os.path.join(PLOTS_DIR, 'error_distribution_mackey.png'))
plt.close()
