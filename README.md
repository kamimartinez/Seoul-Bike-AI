# Seoul Bike Sharing Demand 
## Implementación de una técnica de aprendizaje máquina 

**[Ver el paper: Evidencia01_ConcentracionIA_A01711833](./Evidencia01_ConcentracionIA_A01711833.pdf)**

## Descripción

Este proyecto implementa dos enfoques de modelado para predecir la demanda horaria de bicicletas rentadas en Seúl a partir de variables climáticas y de calendario: una regresión lineal múltiple entrenada mediante descenso de gradiente, implementada desde cero sin frameworks de machine learning, y una red neuronal con reducción de dimensionalidad mediante PCA.

El dataset utilizado es [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), del UCI Machine Learning Repository.

## Requisitos

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow

Instalación de dependencias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Uso

El proyecto se ejecuta en dos pasos, en este orden:

1. **Limpieza y transformación de datos** 
 exporta el dataset listo para entrenar como `seoul-clean.csv`:

   ```bash
   python cleaning.py
   ```

2. **Entrenamiento del modelo sin framework** entrena el modelo y muestra las métricas de evaluación (RMSE, R²):

   ```bash
   python main.py
   ```

3. **Entrenamiento del modelo con framework** entrena el modelo y muestra las métricas de evaluación (RMSE, R²):

   ```bash
   python main-framework.py
   ```

> ojito, `main.py` y `main-framework.py` depende del archivo generado por `cleaning.py`, así que es necesario correr `cleaning.py` primero.
