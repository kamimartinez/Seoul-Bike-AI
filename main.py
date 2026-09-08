import pandas as pd
import matplotlib.pyplot as plt
import random

df_model = pd.read_csv('seoul-clean.csv')

print(df_model.shape)
df_model.head()

__errors__ = []      # error de train en cada época
__val_errors__ = []  # error de validation en cada época

def h(params, sample):
    """Hipótesis lineal: h(x) = a + b*x1 + c*x2 + ... + n*xn"""
    acum = 0
    for i in range(len(params)):
        acum = acum + params[i] * sample[i]
    return acum

def show_errors(params, samples, y, errors_list, verbose=False):
    """Calcula el error cuadrático medio con los parámetros actuales y lo guarda en errors_list"""
    error_acum = 0
    for i in range(len(samples)):
        hyp = h(params, samples[i])
        if verbose:
            print("hyp %f  y %f" % (hyp, y[i]))
        error = hyp - y[i]
        error_acum += error ** 2 
    mean_error_param = error_acum / len(samples)
    rmse = mean_error_param ** 0.5 
    errors_list.append(rmse)

def GD(params, samples, y, alfa):
    """Un paso de descenso de gradiente sobre todos los parámetros"""
    temp = list(params)
    for j in range(len(params)):
        acum = 0
        for i in range(len(samples)):
            error = h(params, samples[i]) - y[i]
            acum = acum + error * samples[i][j]
        temp[j] = params[j] - alfa * (1 / len(samples)) * acum
    return temp

def r2_score(params, samples, y):
    """Calcula R^2 manualmente para un conjunto dado"""
    predictions = [h(params, s) for s in samples]
    y_mean = sum(y) / len(y)
    ss_res = sum((real - pred) ** 2 for pred, real in zip(y, predictions))
    ss_tot = sum((real - y_mean) ** 2 for real in y)
    return 1 - (ss_res / ss_tot)

target_col = 'Rented Bike Count'
feature_cols = [c for c in df_model.columns if c != target_col]

y = df_model[target_col].tolist()
samples = df_model[feature_cols].values.tolist()

samples = [[1] + row for row in samples]

print("features:", feature_cols)
print("num samples:", len(samples), "| num params:", len(samples[0]))


random.seed(42)
idx = list(range(len(samples)))
random.shuffle(idx)

train_end = int(len(idx) * 0.6)
val_end = int(len(idx) * 0.8)  # 0.6 + 0.2

train_idx = idx[:train_end]
val_idx = idx[train_end:val_end]
test_idx = idx[val_end:]

samples_train = [samples[i] for i in train_idx]
y_train = [y[i] for i in train_idx]

samples_val = [samples[i] for i in val_idx]
y_val = [y[i] for i in val_idx]

samples_test = [samples[i] for i in test_idx]
y_test = [y[i] for i in test_idx]

print("train:", len(samples_train), "| validation:", len(samples_val), "| test:", len(samples_test))

params = [0] * len(samples_train[0])
alfa = 0.01
tolerancia = 1e-6
max_epochs = 2000

epochs = 0
while True:
    oldparams = list(params)
    params = GD(params, samples_train, y_train, alfa)
    show_errors(params, samples_train, y_train, __errors__)
    show_errors(params, samples_val, y_val, __val_errors__)
    epochs += 1

    max_change = max(abs(p - op) for p, op in zip(params, oldparams))
    if max_change < tolerancia or epochs == max_epochs:
        print("épocas:", epochs)
        print("parámetros finales:", params)
        break


plt.plot(__errors__, label="train")
plt.plot(__val_errors__, label="validation")
plt.xlabel("Época")
plt.ylabel("Error (MSE)")
plt.title("Curva de error - Gradiente Descendente")
plt.legend()
plt.show()

rmse_train_list = []
show_errors(params, samples_train, y_train, rmse_train_list)
rmse_train = rmse_train_list[0]

rmse_val_list = []
show_errors(params, samples_val, y_val, rmse_val_list)
rmse_val = rmse_val_list[0]

rmse_test_list = []
show_errors(params, samples_test, y_test, rmse_test_list)
rmse_test = rmse_test_list[0]

r2_train = r2_score(params, samples_train, y_train)
r2_val = r2_score(params, samples_val, y_val)
r2_test = r2_score(params, samples_test, y_test)

gap_rmse_val = rmse_val - rmse_train
gap_rmse_test = rmse_test - rmse_train
gap_r2_val = r2_train - r2_val
gap_r2_test = r2_train - r2_test

print("\n--- Métricas finales ---")
print(f"RMSE train:      {rmse_train:.4f}")
print(f"RMSE validation: {rmse_val:.4f}")
print(f"RMSE test:       {rmse_test:.4f}")
print()
print(f"R2 train:       {r2_train:.4f}")
print(f"R2 validation:  {r2_val:.4f}")
print(f"R2 test:        {r2_test:.4f}")
print()
print(f"Gap RMSE (val - train):  {gap_rmse_val:.4f}")
print(f"Gap RMSE (test - train): {gap_rmse_test:.4f}")
print(f"Gap R2 (train - val):   {gap_r2_val:.4f}")
print(f"Gap R2 (train - test):  {gap_r2_test:.4f}")

# Gráfica: Predicciones vs Valores Reales (test set)
predictions_test = [h(params, s) for s in samples_test]

plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions_test, alpha=0.4, edgecolors='k', linewidths=0.3)

# Línea de referencia y = x (predicción perfecta)
min_val = min(min(y_test), min(predictions_test))
max_val = max(max(y_test), max(predictions_test))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Predicción perfecta (y=x)")

plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Predicciones vs Valores Reales (Test)")
plt.legend()
plt.tight_layout()
plt.show()