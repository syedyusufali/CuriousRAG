# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy everything into the image
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
