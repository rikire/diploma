import pandas as pd
import numpy as np
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def run_tpot_on_dataset(dataset_path, dataset_name):
    print(f"\n===== TPOT on {dataset_name} =====")
    df = pd.read_csv(dataset_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tpot = TPOTRegressor(generations=20, population_size=50, verbosity=2, random_state=42, n_jobs=-1)
    tpot.fit(X_train, y_train)

    y_pred = tpot.predict(X_test)
    good_mask = np.isfinite(y_test) & np.isfinite(y_pred)
    y_test_filtered = y_test[good_mask]
    y_pred_filtered = y_pred[good_mask]
    mae = mean_absolute_error(y_test_filtered, y_pred_filtered)
    mse = mean_squared_error(y_test_filtered, y_pred_filtered)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_filtered, y_pred_filtered)
    print(f"TPOT MAE: {mae:.4f}")
    print(f"TPOT RMSE: {rmse:.4f}")
    print(f"TPOT R2: {r2:.4f}")

    # Графики и папки
    plot_dir = f'logs/20250604/plots_{dataset_name}'
    model_dir = 'logs/20250604/models'
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_filtered, y_pred_filtered, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'TPOT: Actual vs Predicted ({dataset_name})')
    plt.plot([y_test_filtered.min(), y_test_filtered.max()], [y_test_filtered.min(), y_test_filtered.max()], 'r--')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/tpot_actual_vs_pred.png')
    plt.close()

    # Сохраняем лучший pipeline TPOT
    with open(f'{model_dir}/best_model_{dataset_name}_TPOT.py', 'w') as f:
        f.write(tpot.export())
    print("Pipeline:")
    print(tpot.fitted_pipeline_)

if __name__ == "__main__":
    datasets = [
        ("datasets/forest_dataset_lags.csv", "fire"),
        ("datasets/daily_sunspots_lags.csv", "sunspots"),
        ("datasets/mackey_glass_dataset.csv", "mackey_glass"),
    ]
    for path, name in datasets:
        run_tpot_on_dataset(path, name)