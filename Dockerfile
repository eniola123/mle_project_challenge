# ==============================
# Sound Realty House Price API
# ==============================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Train model (optional, if model artifacts not yet created)
# RUN python train_model.py

# Start the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
