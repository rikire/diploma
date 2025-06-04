import h2o
from h2o.automl import H2OAutoML
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt

# Инициализация H2O
h2o.init(port=54321, nthreads=2, bind_to_localhost=True, strict_version_check=False)

print("Загрузка данных...")
fires = h2o.import_file("datasets/forest_fires_prepared.csv")
print("Данные загружены.")

print("Определение признаков и целевой переменной...")
x = [col for col in fires.columns if col != 'area']
y = 'area'

print("Разделение на train/test...")
train, test = fires.split_frame(ratios=[0.8], seed=42)
print("Данные разделены.")

print("Запуск AutoML...")
aml = H2OAutoML(max_models=20, max_runtime_secs=600, seed=42, sort_metric='RMSE')
aml.train(x=x, y=y, training_frame=train)
print("AutoML завершён.")

# Лучшая модель
leader = aml.leader
print('Best H2O model:', leader.algo)

# Оценка на тесте
perf = leader.model_performance(test_data=test)
print(perf)

# Feature importance (если есть)
if hasattr(leader, 'varimp'):  # GBM, XGBoost, DRF и др.
    print('Feature importance:')
    print(leader.varimp(use_pandas=True))

# Предсказания и сравнение
pred = leader.predict(test).as_data_frame().values.ravel()
y_true = test[y].as_data_frame().values.ravel()
print('\nSklearn metrics:')
print('MSE:', mean_squared_error(y_true, pred))
print('RMSE:', np.sqrt(mean_squared_error(y_true, pred)))
print('MAE:', mean_absolute_error(y_true, pred))
print('R2:', r2_score(y_true, pred))
print('Explained variance:', explained_variance_score(y_true, pred))

# График предсказание vs факт
plt.figure(figsize=(8,6))
plt.scatter(y_true, pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('H2O Prediction vs Actual (forest fires)')
plt.show()
