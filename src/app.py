import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cargar el modelo
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model(r'C:\Users\lenovo\Desktop\Valoracion de Propiedades\Models\modelo_prediccion_precios_capas.h5')

modelo = cargar_modelo()

# Inicializar el StandardScaler
scaler = StandardScaler()

st.title('Valoración de Propiedades en Benalmádena')

# Campos de entrada
tipo_propiedad = st.selectbox('Tipo de Propiedad', ['Piso', 'Chalet Pareado', 'Chalet Adosado', 'Casa Independiente', 'Chalet Independiente', 'Ático', 'Estudio', 'Dúplex', 'Casa Rural', 'Chalet'])
metros_cuadrados = st.number_input('Metros cuadrados construidos', min_value=20, max_value=1000, value=100)
habitaciones = st.number_input('Número de habitaciones', min_value=0, max_value=10, value=2)
banos = st.number_input('Número de baños', min_value=1, max_value=5, value=1)
planta = st.number_input('Planta', min_value=0, max_value=20, value=0)

# Características adicionales
ascensor = st.checkbox('Ascensor')
obra_nueva = st.checkbox('Obra nueva')
piscina = st.checkbox('Piscina')
terraza = st.checkbox('Terraza')
parking = st.checkbox('Parking')
aire_acondicionado = st.checkbox('Aire acondicionado')
trastero = st.checkbox('Trastero')
jardin = st.checkbox('Jardín')

# Ubicación
referencia_catastral = st.text_input('Referencia Catastral')

def preprocesar_caracteristicas(caracteristicas):
    # Convertir a DataFrame
    df = pd.DataFrame([caracteristicas], columns=[
        'metros_cuadrados', 'habitaciones', 'banos', 'planta',
        'ascensor', 'obra_nueva', 'piscina', 'terraza', 'parking',
        'aire_acondicionado', 'trastero', 'jardin',
        'Piso', 'Chalet Pareado', 'Chalet Adosado', 'Casa Independiente',
        'Chalet Independiente', 'Ático', 'Estudio', 'Dúplex', 'Casa Rural', 'Chalet',
        'referencia_catastral'
    ])
    
    # Aplicar codificación one-hot a la referencia catastral
    df['referencia_catastral'] = pd.Categorical(df['referencia_catastral']).codes
    
    # Escalar las características numéricas
    numeric_features = ['metros_cuadrados', 'habitaciones', 'banos', 'planta', 'referencia_catastral']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    return df.values

if st.button('Valorar Propiedad'):
    # Preparar características
    caracteristicas = [
        metros_cuadrados, habitaciones, banos, planta,
        ascensor, obra_nueva, piscina, terraza, parking, aire_acondicionado, trastero, jardin
    ]
    
    # Codificación one-hot para el tipo de propiedad
    tipos_propiedad = ['Piso', 'Chalet Pareado', 'Chalet Adosado', 'Casa Independiente', 'Chalet Independiente', 'Ático', 'Estudio', 'Dúplex', 'Casa Rural', 'Chalet']
    caracteristicas.extend([1 if tp == tipo_propiedad else 0 for tp in tipos_propiedad])
    
    # Agregar referencia catastral
    caracteristicas.append(referencia_catastral)
    
    # Preprocesar características
    caracteristicas_procesadas = preprocesar_caracteristicas(caracteristicas)
    
    # Hacer predicción
    prediccion = modelo.predict(caracteristicas_procesadas)
    
    # Mostrar resultado
    st.success(f'El valor estimado de la propiedad es: {prediccion[0][0]:.2f} €')

# Información adicional sobre el modelo
st.subheader('Información sobre el modelo')
st.write("""
Este modelo de valoración de propiedades en Benalmádena utiliza técnicas de aprendizaje profundo
y se basa en datos de diversas fuentes, incluyendo:
- Datos de Idealista
- Imágenes satelitales
- Datos catastrales
- Puntos de interés (POIs)

Las características más influyentes en el precio de las propiedades incluyen:
- Metros cuadrados construidos
- Número de baños y habitaciones
- Ubicación (referencia catastral)
- Presencia de amenidades como piscina, parking y aire acondicionado
""")

# Mapa de Benalmádena
st.subheader('Mapa de Benalmádena')
datos_mapa = pd.DataFrame({
    'lat': [36.5982],
    'lon': [-4.5162]
})
st.map(datos_mapa)