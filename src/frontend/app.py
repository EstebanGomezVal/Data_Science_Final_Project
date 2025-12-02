import streamlit as st
import requests

st.write("""
# Income Prediction
Introduce los datos de una persona y obtén la predicción.
""")

st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)

    workclass = st.sidebar.text_input("Workclass", "Private")
    fnlwgt = st.sidebar.number_input("FNLWGT", min_value=0, value=150000)

    education = st.sidebar.text_input("Education", "Bachelors")
    education_num = st.sidebar.number_input("Education Num", min_value=0, max_value=20, value=13)

    marital_status = st.sidebar.text_input("Marital Status", "Never-married")
    occupation = st.sidebar.text_input("Occupation", "Prof-specialty")
    relationship = st.sidebar.text_input("Relationship", "Not-in-family")
    race = st.sidebar.text_input("Race", "White")
    sex = st.sidebar.text_input("Sex", "Male")

    capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.sidebar.number_input("Hours per Week", min_value=1, max_value=100, value=40)
    native_country = st.sidebar.text_input("Native Country", "United-States")

    input_dict = {
        "age": int(age),
        "workclass": workclass,
        "fnlwgt": int(fnlwgt),
        "education": education,
        "education_num": int(education_num),
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": int(capital_gain),
        "capital_loss": int(capital_loss),
        "hours_per_week": int(hours_per_week),
        "native_country": native_country
    }

    return input_dict


input_data = user_input_features()

API_URL = "http://127.0.0.1:8000/predict"

if st.button("Predecir"):
    try:
        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            result = response.json()

            # Interpretación del modelo
            if result["class"] == ">50K":
                texto = "Esta persona gana **más de 50,000 dólares** al año."
            else:
                texto = "Esta persona gana **menos o igual a 50,000 dólares** al año."

            st.success(texto)

            # Mostrar respuesta cruda opcional
            with st.expander("Ver detalles de la predicción"):
                st.json(result)

        else:
            st.error("Error en la API:")
            st.write(response.text)

    except Exception as e:
        st.error("No se pudo conectar con la API.")
        st.write(e)
