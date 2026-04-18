# use a lightweight Python image so the container isn't too big
FROM python:3.11-slim

# set the working directory inside the container
# everything will run from /app
WORKDIR /app

# copy requirements first
COPY requirements.txt .

# install all the Python dependencies needed for the API
RUN pip install --no-cache-dir -r requirements.txt


# copy the main FastAPI app file into the container
COPY app.py .

# copy the trained model files (needed for predictions)
COPY models/ ./models/

# copy the taxi lookup file used in the app
COPY data/raw/taxi_lookup.csv ./data/raw/taxi_lookup.csv

# expose port 8000 so we can access the API from outside the container
EXPOSE 8000

# run the FastAPI app using uvicorn
# 0.0.0.0 makes it accessible outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]