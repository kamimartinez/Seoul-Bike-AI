# Seoul Bike Sharing Demand 
## Implementación de una técnica de aprendizaje máquina sin el uso de un framework

**[Ver el paper: Evidencia01_ConcentracionIA_A01711833](./Evidencia01_ConcentracionIA_A01711833.pdf)**

## Descripción

Este proyecto implementa un modelo de regresión lineal múltiple, entrenado mediante descenso de gradiente para predecir la demanda horaria de bicicletas rentadas en Seúl a partir de variables climáticas y de calendario.

El dataset utilizado es [Seoul Bike Sharing Demand](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), del UCI Machine Learning Repository.

## Requisitos

- Python 3.x
- pandas

Instalación de dependencias:

```bash
pip install pandas
```

## Uso

El proyecto se ejecuta en dos pasos, en este orden:

1. **Limpieza y transformación de datos** 
 exporta el dataset listo para entrenar como `seoul-clean.csv`:

   ```bash
   python cleaning.py
   ```

2. **Entrenamiento del modelo** — carga `seoul-clean.csv`, entrena el modelo y muestra las métricas de evaluación (MSE, R²):

   ```bash
   python main.py
   ```

> ojito, `main.py` depende del archivo generado por `cleaning.py`, así que es necesario correr `cleaning.py` primero.
