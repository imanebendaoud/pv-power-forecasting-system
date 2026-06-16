# Adaptez la version Python selon vos dépendances TensorFlow/pvlib
FROM python:3.11-slim

# Timezone système (utile pour les logs et certains appels système)
ENV TZ=Africa/Casablanca
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Installer les dépendances d'abord (meilleur cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY . .

# Le dossier data sera monté en volume (voir docker-compose.yml)
RUN mkdir -p /app/data

EXPOSE 5000

CMD ["python", "app.py"]