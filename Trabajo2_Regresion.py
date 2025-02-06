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

# Interfaz en Streamlit
def main():
    st.title("Predecir el precio de una casa cercana al mar")
    st.markdown("El modelo final fue desarrollado con el metodo de clasificación implementa el voto de k- vecinos más cercanos con los hiperparametros n_neighbors:4, p:3")
    st.markdown("Indique las características de la casa para hacer la predicción")

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
        st.markdown(f"### El valor estimado de la casa es: **${prediction * 1000:.2f} USD**")

if __name__ == '__main__':
    main()
