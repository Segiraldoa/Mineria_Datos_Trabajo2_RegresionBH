import streamlit as st
import numpy as np
import gzip
import pickle

# Cargar el modelo entrenado
def load_model():
    filename = "model_trained_regressor.pkl.gz"
    with gzip.open(filename, "rb") as f:
        model = pickle.load(f)
    return model

# Estilos CSS personalizados
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            text-align: center;
            color: #2E86C1;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #566573;
        }
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #1A5276;
        }
        .prediction-box {
            text-align: center;
            font-size: 24px;
            color: #154360;
            font-weight: bold;
            background-color: #D6EAF8;
            padding: 10px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Interfaz en Streamlit
def main():
    st.markdown("<h1 class='main-title'>Predicción del Precio de Casas Cercanas al Mar</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>El modelo final utiliza el método de clasificación basado en el voto de k-vecinos más cercanos con los hiperparámetros <b>n_neighbors: 4, p: 3</b>.</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Ingrese las características de la casa para obtener la predicción")
    
    # Entrada de datos por el usuario
    crim = st.number_input("Tasa de criminalidad per cápita", min_value=0.0, format="%.5f")
    zn = st.number_input("Proporción de terrenos residenciales", min_value=0.0, format="%.2f")
    indus = st.number_input("Proporción de acres comerciales", min_value=0.0, format="%.2f")
    chas = st.selectbox("Cercanía al río Charles", [0, 1])
    nox = st.number_input("Concentración de óxidos de nitrógeno (NOX)", min_value=0.0, format="%.3f")
    rm = st.number_input("Número medio de habitaciones por vivienda", min_value=1.0, format="%.2f")
    age = st.number_input("Proporción de casas construidas antes de 1940", min_value=0.0, format="%.1f")
    dis = st.number_input("Distancia a centros de empleo", min_value=0.0, format="%.2f")
    rad = st.number_input("Índice de accesibilidad a carreteras", min_value=1, max_value=24)
    tax = st.number_input("Tasa de impuestos a la propiedad", min_value=0, format="%d")
    ptratio = st.number_input("Ratio de alumnos por profesor", min_value=0.0, format="%.1f")
    b = st.number_input("Proporción de residentes afroamericanos", min_value=0.0, format="%.2f")
    lstat = st.number_input("Porcentaje de población con bajo nivel socioeconómico", min_value=0.0, format="%.2f")
    
    # Botón para realizar la predicción
    if st.button("Realizar predicción"):
        model = load_model()
        input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
        prediction = model.predict(input_data)[0]
        st.markdown(f"<div class='prediction-box'>El valor estimado de la casa es: <br> <b>${prediction * 1000:.2f} USD</b></div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
