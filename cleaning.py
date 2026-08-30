import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('SeoulBikeData.csv', encoding='unicode_escape')
print(df.head())

# Forma, tipo y datos duplicados
print (df.describe(include=object))

print(df.shape)

print(df.dtypes)

print(df.duplicated().sum())

# Valores nulos
df.isna().sum()

# Variables categoricas

print(df['Seasons'].unique())
print(df['Holiday'].unique())
print(df['Functioning Day'].unique())

print(df[df['Functioning Day'] == 'No']['Rented Bike Count'].describe())

before = len(df)
df = df[df['Functioning Day'] == 'Yes'].reset_index(drop=True)
df = df.drop(columns=['Functioning Day'])

print(f"filas eliminadas: {before - len(df)}")
print(df.shape)

df_clean = pd.get_dummies(df, columns=['Seasons', 'Holiday'], dtype=int)

print(df_clean.head())

# Distribución de las variables

df.hist(figsize=(14, 10), bins=30, edgecolor='black')
plt.tight_layout()
plt.show()

# Relación entre las variables, utilizando Season (variable categorica),
# como hue
sns.pairplot(df, hue='Seasons', size=1.5);

# Matriz de correlación
df.select_dtypes(include=['number']).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Matriz de Correlación")
plt.show()

# Eliminamos la columna Dew point Temperature (°C) y Date 
df_final = df_clean.drop(columns=['Dew point temperature(°C)', 'Date'])
print(df_final.head())

# Estandarización de variables 
# Excluyendo Hour, Rented Bike Count y las variables categoricas
cols_standarization = [ 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)',
    'Visibility (10m)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)'
]

for col in cols_standarization:
    mu = df_final[col].mean()
    sigma = df_final[col].std()
    df_final[col] = (df_final[col] - mu) / sigma

print(df_final[cols_standarization].describe())

# Exportar nuevo dataset "limpio"

df_final.to_csv('seoul-clean.csv', index=False)
print("Dataset exportado como seoul-clean.csv")
print(df_final.shape)




