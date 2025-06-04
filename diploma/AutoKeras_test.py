import pandas as pd
import numpy as np
import autokeras as ak
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

print('AutoKeras version:', ak.__version__)
print('StructuredDataRegressor' in dir(ak))

# Загрузка данных
df = pd.read_csv('datasets/forest_dataset_lags.csv')
df = df.replace([np.inf, -np.inf], np.nan).dropna()
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AutoKeras StructuredDataRegressor для регрессии на табличных данных
reg = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=20
)
reg.fit(X_train, y_train, epochs=100, validation_split=0.2)

y_pred = reg.predict(X_test).flatten()
good_mask = np.isfinite(y_test) & np.isfinite(y_pred)
y_test_filtered = y_test[good_mask]
y_pred_filtered = y_pred[good_mask]
mae = mean_absolute_error(y_test_filtered, y_pred_filtered)
mse = mean_squared_error(y_test_filtered, y_pred_filtered)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_filtered, y_pred_filtered)
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_filtered, y_pred_filtered, alpha=0.5)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('AutoKeras StructuredDataRegressor: Actual vs Predicted')
plt.plot([y_test_filtered.min(), y_test_filtered.max()], [y_test_filtered.min(), y_test_filtered.max()], 'r--')
plt.tight_layout()
plt.savefig('logs/20250604/plots_fires/autokeras_actual_vs_pred.png')
plt.show()
reg.export_model().save('logs/20250604/models/best_model_fire_AutoKeras.h5')
