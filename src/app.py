import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import logging
import requests

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Función para geocodificar una dirección
def geocode_address(address):
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': address,
        'format': 'json',
        'limit': 1
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200 and response.json():
        location = response.json()[0]
        return float(location['lat']), float(location['lon'])
    else:
        return None, None

# Cargar el modelo
@st.cache_resource
def cargar_modelo():
    modelo_path = os.path.join('..', 'Models', 'modelo_prediccion_precios_capas.h5')
    try:
        modelo = tf.keras.models.load_model(modelo_path)
        logging.info("Modelo cargado correctamente.")
        return modelo
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {str(e)}")
        st.error(f"No se pudo cargar el modelo: {str(e)}")
        return None

# Cargar preprocesadores
@st.cache_resource
def cargar_preprocesadores():
    try:
        preprocessor_catastro = joblib.load('preprocessor_catastro.joblib')
        preprocessor_satelital = joblib.load('preprocessor_satelital.joblib')
        preprocessor_poi = joblib.load('preprocessor_poi.joblib')
        preprocessor_idealista = joblib.load('preprocessor_idealista.joblib')
        logging.info("Preprocesadores cargados correctamente.")
        return preprocessor_catastro, preprocessor_satelital, preprocessor_poi, preprocessor_idealista
    except Exception as e:
        logging.error(f"Error al cargar los preprocesadores: {str(e)}")
        st.error(f"No se pudieron cargar los preprocesadores: {str(e)}")
        return None, None, None, None

# Cargar modelo y preprocesadores
modelo = cargar_modelo()
preprocessor_catastro, preprocessor_satelital, preprocessor_poi, preprocessor_idealista = cargar_preprocesadores()

# Verificar si se cargaron correctamente
if modelo is None or None in (preprocessor_catastro, preprocessor_satelital, preprocessor_poi, preprocessor_idealista):
    st.error("No se pudo cargar el modelo o los preprocesadores. Verifica la compatibilidad y las versiones de las bibliotecas.")
else:
    st.success("Modelo y preprocesadores cargados correctamente.")

# Título de la aplicación
st.title('Valoración de Propiedades en Benalmádena')

# Campos de entrada para las características de la vivienda
st.header("Características de la Vivienda")
metros_cuadrados = st.number_input("Metros cuadrados construidos", min_value=20, max_value=1000, value=100)
habitaciones = st.number_input("Número de habitaciones", min_value=1, max_value=10, value=3)
banos = st.number_input("Número de baños", min_value=1, max_value=5, value=2)
tipo_propiedad = st.selectbox("Tipo de Propiedad", ["Piso", "Casa", "Chalet", "Ático"])

# Características adicionales
st.header("Características Adicionales")
ascensor = st.checkbox("Ascensor")
piscina = st.checkbox("Piscina")
terraza = st.checkbox("Terraza")
parking = st.checkbox("Parking")
aire_acondicionado = st.checkbox("Aire acondicionado")

# Ubicación
st.header("Ubicación")
calle = st.text_input("Calle")
numero = st.text_input("Número")
codigo_postal = st.text_input("Código Postal")

# Botón para geocodificar
if st.button('Obtener Coordenadas'):
    direccion_completa = f"{calle} {numero}, {codigo_postal} Benalmádena, España"
    latitud, longitud = geocode_address(direccion_completa)
    if latitud and longitud:
        st.success(f"Coordenadas obtenidas: Latitud {latitud}, Longitud {longitud}")
    else:
        st.error("No se pudieron obtener las coordenadas. Por favor, verifica la dirección.")

# Botón para realizar la valoración
if st.button('Valorar Propiedad'):
    try:
        # Obtener coordenadas
        direccion_completa = f"{calle} {numero}, {codigo_postal} Benalmádena, España"
        latitud, longitud = geocode_address(direccion_completa)
        
        if not latitud or not longitud:
            st.error("No se pudieron obtener las coordenadas. Por favor, verifica la dirección.")
            st.stop()

        # Preparar los datos para cada capa
        catastro_valores = {
            'area_parcela': metros_cuadrados,
            'num_plantas_numeric': 1 if tipo_propiedad == "Piso" else 2,
            'latitud': latitud,
            'longitud': longitud,
            # Añade aquí más campos del catastro según tu modelo
        }
        
        satelital_valores = {
            'latitud': latitud,
            'longitud': longitud,
            # Añade aquí campos satelitales
        }
        
        poi_valores = {
            'latitud': latitud,
            'longitud': longitud,
            # Añade aquí conteo de POIs cercanos
        }
        
        idealista_valores = {
            'Metros cuadrados construidos': metros_cuadrados,
            'Habitaciones': habitaciones,
            'Baños': banos,
            'Ascensor (Sí/No)': ascensor,
            'Piscina (Sí/No)': piscina,
            'Terraza (Sí/No)': terraza,
            'Parking (Sí/No)': parking,
            'Aire acondicionado (Sí/No)': aire_acondicionado,
            'latitud': latitud,
            'longitud': longitud,
            # Añade aquí más campos de Idealista según tu modelo
        }

        # Convertir a DataFrames
        catastro_df = pd.DataFrame([catastro_valores])
        satelital_df = pd.DataFrame([satelital_valores])
        poi_df = pd.DataFrame([poi_valores])
        idealista_df = pd.DataFrame([idealista_valores])

        # Preprocesar los datos
        X_catastro_prep = preprocessor_catastro.transform(catastro_df)
        X_satelital_prep = preprocessor_satelital.transform(satelital_df)
        X_poi_prep = preprocessor_poi.transform(poi_df)
        X_idealista_prep = preprocessor_idealista.transform(idealista_df)

        # Realizar la predicción
        prediccion = modelo.predict([X_catastro_prep, X_satelital_prep, X_poi_prep, X_idealista_prep])

        # Mostrar el resultado
        st.success(f'El valor estimado de la propiedad es: {prediccion[0][0]:.2f} €')
    except Exception as e:
        logging.error(f"Error durante la predicción: {str(e)}")
        st.error(f"Ocurrió un error durante la valoración: {str(e)}")

# Información adicional sobre el modelo
with st.expander("Información sobre el modelo", expanded=False):
    st.write("""
    Este modelo de valoración de propiedades en Benalmádena utiliza técnicas de aprendizaje profundo
    y se basa en datos de diversas fuentes, incluyendo:
    - Datos del Catastro
    - Imágenes satelitales
    - Puntos de interés (POIs)
    - Datos de Idealista

    El modelo combina estas diferentes fuentes de datos para proporcionar una estimación precisa
    del valor de la propiedad basándose en sus características y ubicación.
    """)

# Información de versiones
with st.expander("Información técnica", expanded=False):
    st.info(f"""
    Versiones de bibliotecas:
    TensorFlow: {tf.__version__}
    NumPy: {np.__version__}
    Pandas: {pd.__version__}
    Scikit-learn: {sklearn.__version__}
    """)