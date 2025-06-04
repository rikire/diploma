import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_mackey_glass(beta=0.2, gamma=0.1, n=10, tau=17, dt=0.1,
                          total_time=300, burn_in_time=100, noise_std=0.01):
    total_steps = int((total_time + burn_in_time) / dt)
    burn_in_steps = int(burn_in_time / dt)
    tau_steps = int(tau / dt)
    x0 = ((beta / gamma) - 1) ** (1 / n)
    x = np.zeros(total_steps)
    x[:tau_steps] = x0 + 0.01 * np.random.randn(tau_steps)
    for t in range(tau_steps, total_steps):
        dx = dt * (beta * x[t - tau_steps] / (1 + x[t - tau_steps] ** n) -
                   gamma * x[t - 1])
        # Добавляем шум
        x[t] = x[t - 1] + dx + np.random.normal(0, noise_std)
    return x[burn_in_steps:]

def make_windowed_dataset(series: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    X = np.array(X)
    y = np.array(y)
    return X, y

if __name__ == "__main__":
    # Параметры для хорошего тестового ряда
    series = generate_mackey_glass(
        beta=0.2, gamma=0.1, n=15, tau=30, dt=0.05,
        total_time=1000, burn_in_time=200, noise_std=0.02
    )
    window = 30
    X, y = make_windowed_dataset(series, window)

    # Индексы времени для каждого окна (t соответствует первому элементу окна)
    t_idx = np.arange(len(X))

    # Визуализация временного ряда
    plt.figure(figsize=(12, 4))
    plt.plot(series, label="Mackey-Glass series")
    plt.title("Синтетический временной ряд Маккея-Гласса")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Сохраняем исходный временной ряд
    pd.DataFrame({"t": np.arange(len(series)), "x": series}).to_csv("mackey_glass_series.csv", index=False)

    # Формируем DataFrame с временными окнами
    df = pd.DataFrame(X, columns=[f"x_t-{window-i-1}" for i in range(window)])
    df.insert(0, "t", t_idx)
    df["target"] = y
    df.to_csv("mackey_glass_dataset.csv", index=False)
    print("mackey_glass_dataset.csv и mackey_glass_series.csv успешно созданы!")