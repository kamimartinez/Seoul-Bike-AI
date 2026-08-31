import pandas as pd

df_model = pd.read_csv('seoul-clean.csv')

if 'Date' in df_model.columns:
    df_model = df_model.drop(columns=['Date'])

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
    errors_list.append(mean_error_param)

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

target_col = 'Rented Bike Count'
feature_cols = [c for c in df_model.columns if c != target_col]

y = df_model[target_col].tolist()
samples = df_model[feature_cols].values.tolist()

samples = [[1] + row for row in samples]

print("features:", feature_cols)
print("num samples:", len(samples), "| num params:", len(samples[0]))


import random

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

import matplotlib.pyplot as plt

plt.plot(__errors__, label="train")
plt.plot(__val_errors__, label="validation")
plt.xlabel("Época")
plt.ylabel("Error (MSE)")
plt.title("Curva de error - Gradiente Descendente")
plt.legend()
plt.show()

predictions = [h(params, s) for s in samples_test]

# MSE manual
mse = sum((pred - real) ** 2 for pred, real in zip(predictions, y_test)) / len(y_test)

# R^2 manual: 1 - (suma de errores^2 / suma de (y - media(y))^2)
y_mean = sum(y_test) / len(y_test)
ss_res = sum((real - pred) ** 2 for pred, real in zip(predictions, y_test))
ss_tot = sum((real - y_mean) ** 2 for real in y_test)
r2 = 1 - (ss_res / ss_tot)

print("MSE en test:", mse)
print("R2 en test:", r2)