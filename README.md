# Adult Income Prediction - Análisis y Preprocesamiento

Este repositorio contiene el flujo completo de análisis exploratorio de datos (EDA), preprocesamiento y entrenamiento de modelos para la predicción de ingresos, utilizando el dataset **Adult Income**.  

El objetivo del proyecto es analizar las características demográficas y laborales que influyen en si una persona gana más o menos de $50,000 al año, y preparar los datos para desarrollar modelos de clasificación que permitan realizar esta predicción con precisión.  

---

# Autores

Proyecto desarrollado por Esteban Gómez Valerio, Oscar Josue Rocha Hernandez, Rafael Takata Garcia

## ⚙️ Requisitos previos

- **Python 3.10+** (se recomienda utilizar un entorno virtual)
- **uv** para la gestión de dependencias
- **Jupyter Notebook** para ejecutar los notebooks reproducibles
- Acceso a **Kaggle** para descargar el dataset

---

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/EstebanGomezVal/Data_Science_Final_Project.git
   cd Data_Science_Final_Project
   ```

2. Sincroniza las librerías necesarias con uv:
   ```bash
   uv sync
   ```

3. Descarga el data set desde kaggle `https://www.kaggle.com/datasets/mosapabdelghany/adult-income-prediction-datasety` colocalo en la siguiente carpeta
    ``` bash
    data/raw/
    ```

4. Ejecuta los notebooks en el siguiente orden:
`00_informe_inicial.ipynb` → descripción general del dataset y objetivos iniciales

`01_eda_inicial.ipynb` → análisis exploratorio de datos (EDA)

`02_data_wrangling.ipynb` → limpieza, transformación y preprocesamiento de los datos

`03_model_training.ipynb` → entrenamiento y evaluación de modelos predictivos

Ambos notebooks son totalmente reproducibles.

___

Principales librerías utilizadas

- pandas

- numpy

- matplotlib / seaborn

- scikit-learn

- uv (gestión de entorno y dependencias)