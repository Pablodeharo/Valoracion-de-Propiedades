FROM python:3.8

WORKDIR /app

# Instalar dependencias del sistema necesarias para TensorFlow
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# El resto del Dockerfile permanece igual
COPY src /app/src
RUN mkdir -p /app/Models
COPY Models/modelo_prediccion_precios_capas.h5 /app/Models/
EXPOSE 8501
WORKDIR /app/src
CMD ["streamlit", "run", "app.py"]