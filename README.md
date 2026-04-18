# Taxi Tip Prediction API

A FastAPI-based machine learning service that predicts taxi tip amounts for NYC Yellow Taxi trips.  
This project demonstrates MLOps concepts like API deployment, Docker containerization, and MLflow tracking.

---

## Overview

This API serves a trained ML model that predicts taxi tip amounts based on trip details.  
It was built for COMP 3610 and shows a simple end-to-end ML deployment workflow.

---

## Features

- Single prediction for one trip
- Batch predictions (up to 100 records)
- Model info endpoint (metrics + features)
- Health check endpoint
- Input validation using Pydantic
- Swagger UI for testing

---

## Tech Stack

- FastAPI
- Uvicorn
- Pandas / NumPy
- scikit-learn
- MLflow
- Docker / Docker Compose
- pytest

---

## Setup (Local)

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run API

uvicorn app:app --reload

### 3. Open in browser

http://localhost:8000/docs

---

## Docker Setup

`docker build -t my-ml-api .`

`docker run -p 8000:8000 my-ml-api`

---

## Docker Compose

`docker compose up --build`

`docker compose down`

---

## API Endpoints

- POST /predict
- POST /predict/batch
- GET /health
- GET /model/info

---

## Testing

pytest test_app.py -v

---

## AI Tools Used

ChatGPT was used to assist with debugging, Docker setup, and documentation.
