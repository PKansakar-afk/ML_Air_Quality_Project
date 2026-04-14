FROM python:3.10-slim

WORKDIR /app

# Add this block to install the missing system library
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Rest stays the same
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY data/ ./data/
COPY city_day_with_weather_complete.csv .
COPY app.py .
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 5000
CMD ["python", "app.py"]