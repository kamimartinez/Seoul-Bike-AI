import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

df_model = pd.read_csv('seoul-clean.csv')

target_col = 'Rented Bike Count'
feature_cols = [c for c in df_model.columns if c != target_col]

X = df_model[feature_cols].values
y = df_model[target_col].values

print("features:", feature_cols)
print("num samples:", len(X), "| num features original:", X.shape[1])

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("train:", len(X_train), "| validation:", len(X_val), "| test:", len(X_test))

cols_to_standardize = [
    'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
    'Visibility (10m)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)'
]

idx_to_standardize = [feature_cols.index(c) for c in cols_to_standardize if c in feature_cols]
idx_passthrough = [i for i in range(len(feature_cols)) if i not in idx_to_standardize]

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), idx_to_standardize),
    ],
    remainder='passthrough' 
)

X_train_scaled = preprocessor.fit_transform(X_train)
X_val_scaled = preprocessor.transform(X_val)
X_test_scaled = preprocessor.transform(X_test)

pca = PCA(n_components=0.99, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("componentes retenidos:", pca.n_components_)

def get_model(input_shape, weight_decay=0.0001):
    model = Sequential([
        Dense(589, input_shape=input_shape, activation="relu",
              kernel_regularizer=regularizers.l2(weight_decay)),
        Dense(1, activation="linear")
    ])
    return model

model = get_model((X_train_pca.shape[1],))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
)

model.summary()

history = model.fit(
    X_train_pca, y_train,
    validation_data=(X_val_pca, y_val),
    epochs=500,
    batch_size=32,
    verbose=0
)

print("Épocas ejecutadas:", len(history.history['loss']))

plt.plot(history.history['loss'], label="train")
plt.plot(history.history['val_loss'], label="validation")
plt.xlabel("Época")
plt.ylabel("Loss (MSE)")
plt.title("Curva de Loss - Sequential (Keras)")
plt.legend()
plt.show()

plt.plot(history.history['rmse'], label="train")
plt.plot(history.history['val_rmse'], label="validation")
plt.xlabel("Época")
plt.ylabel("RMSE")
plt.title("Curva de RMSE - Sequential (Keras)")
plt.legend()
plt.show()


pred_train = model.predict(X_train_pca, verbose=0).flatten()
pred_val = model.predict(X_val_pca, verbose=0).flatten()
pred_test = model.predict(X_test_pca, verbose=0).flatten()

rmse_train = mean_squared_error(y_train, pred_train) ** 0.5
rmse_val = mean_squared_error(y_val, pred_val) ** 0.5
rmse_test = mean_squared_error(y_test, pred_test) ** 0.5

r2_train = r2_score(y_train, pred_train)
r2_val = r2_score(y_val, pred_val)
r2_test = r2_score(y_test, pred_test)

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

# Predicciones vs valores reales (test set)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred_test, alpha=0.4, edgecolors='k', linewidths=0.3)

min_val = min(y_test.min(), pred_test.min())
max_val = max(y_test.max(), pred_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Predicción perfecta (y=x)")

plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Predicciones vs Valores Reales (Test) - Sequential")
plt.legend()
plt.tight_layout()
plt.show()