# Adult Income Prediction - Análisis y Preprocesamiento

Este repositorio contiene el flujo inicial de análisis exploratorio de datos (EDA) y preprocesamiento de un dataset sobre predicción de ingresos.  

## Requisitos previos

- **Python 3.10+** (se recomienda utilizar un entorno virtual)
- **uv** para la gestión de dependencias
- **Jupyter Notebook** para ejecutar los notebooks reproducibles
- Acceso a **Kaggle** para descargar el dataset

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
- `01_eda_inicial.ipynb → análisis exploratorio de datos (EDA)`
- `02_data_wrangling.ipynb → limpieza y preprocesamiento`


Ambos notebooks son totalmente reproducibles.