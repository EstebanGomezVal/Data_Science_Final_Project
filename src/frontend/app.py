import streamlit as st
import requests

st.write("""
# Income Prediction
Introduce los datos de una persona y obtén la predicción.
""")

st.sidebar.header("User Input Parameters")

def user_input_features():
    # Parámetros Numéricos
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
    fnlwgt = st.sidebar.number_input("FNLWGT", min_value=0, value=1500000)
    education_num = st.sidebar.number_input("Education Num", min_value=0, max_value=16, value=10)
    capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
    hours_per_week = st.sidebar.number_input("Hours per Week", min_value=1, max_value=100, value=40)

    st.sidebar.markdown("---")

    # Parámetros Categóricos
    
    # Opciones basadas en el gráfico de 'Distibución de workclass'
    workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 
                         'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
    workclass = st.sidebar.selectbox("Workclass", workclass_options)

    # Opciones basadas en el gráfico de 'Distibución de education'
    education_options = ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 
                         'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Prof-school', '5th-6th', 
                         '10th', 'Preschool', '12th', '1st-4th', 'Doctorate']
    education = st.sidebar.selectbox("Education", education_options, index=0)

    # Opciones basadas en el gráfico de 'Distibución de marital-status'
    marital_status_options = ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 
                              'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)

    # Opciones basadas en el gráfico de 'Distibución de occupation'
    occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 
                          'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 
                          'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 
                          'Priv-house-serv', '? / Armed-Forces'] # Agrupar "?"
    occupation = st.sidebar.selectbox("Occupation", occupation_options)

    # Opciones basadas en el gráfico de 'Distibución de relationship'
    relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
    relationship = st.sidebar.selectbox("Relationship", relationship_options)

    # Opciones basadas en el gráfico de 'Distibución de race'
    race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
    race = st.sidebar.selectbox("Race", race_options)

    # Opciones basadas en el gráfico de 'Distibución de sex'
    sex_options = ['Male', 'Female']
    sex = st.sidebar.selectbox("Sex", sex_options)

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
                texto = "Esta persona gana **más de 50,000 dólares** al año. 🥳"
            else:
                texto = "Esta persona gana **menos o igual a 50,000 dólares** al año. 😔"

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