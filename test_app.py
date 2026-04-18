from fastapi.testclient import TestClient
from app import app

# sample valid input I used for most tests
valid_record = {
    "pickup_hour": 14,
    "pickup_day_of_week": 2,
    "is_weekend": False,
    "trip_duration_minutes": 18.5,
    "trip_speed_mph": 6.49,
    "log_trip_distance": 1.10,
    "passenger_count": 1,
    "trip_distance": 2.0,
    "fare_amount": 12.5,
    "PULocationID": 161,
    "DOLocationID": 230,
}


def test_single_prediction_success():
    # basic check that /predict works
    with TestClient(app) as client:
        res = client.post("/predict", json=valid_record)

        assert res.status_code == 200
        data = res.json()

        # just checking important fields are returned
        assert "predicted_tip_amount" in data
        assert "model_version" in data
        assert "prediction_id" in data


def test_batch_prediction_success():
    # testing multiple records in one request
    payload = {
        "records": [
            valid_record,
            {
                "pickup_hour": 20,
                "pickup_day_of_week": 5,
                "is_weekend": True,
                "trip_duration_minutes": 25.0,
                "trip_speed_mph": 7.2,
                "log_trip_distance": 1.39,
                "passenger_count": 2,
                "trip_distance": 3.0,
                "fare_amount": 18.0,
                "PULocationID": 230,
                "DOLocationID": 170,
            },
        ]
    }

    with TestClient(app) as client:
        res = client.post("/predict/batch", json=payload)

        assert res.status_code == 200
        data = res.json()

        assert data["count"] == 2
        assert len(data["predictions"]) == 2
        assert "processing_time_ms" in data


def test_invalid_input_missing_field():
    # remove one required field
    bad = valid_record.copy()
    bad.pop("fare_amount")

    with TestClient(app) as client:
        res = client.post("/predict", json=bad)

        assert res.status_code == 422


def test_invalid_input_wrong_type():
    # wrong type for pickup_hour
    bad = valid_record.copy()
    bad["pickup_hour"] = "fourteen"

    with TestClient(app) as client:
        res = client.post("/predict", json=bad)

        assert res.status_code == 422


def test_invalid_input_out_of_range():
    # pickup_hour should be between 0-23
    bad = valid_record.copy()
    bad["pickup_hour"] = 25

    with TestClient(app) as client:
        res = client.post("/predict", json=bad)

        assert res.status_code == 422


def test_health_check_success():
    # simple health endpoint check
    with TestClient(app) as client:
        res = client.get("/health")

        assert res.status_code == 200
        data = res.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


def test_zero_distance_rejected():
    # edge case: distance should not be 0
    bad = valid_record.copy()
    bad["trip_distance"] = 0.0

    with TestClient(app) as client:
        res = client.post("/predict", json=bad)

        assert res.status_code == 422


def test_extreme_fare_still_works():
    # large fare value (should still work unless restricted)
    record = valid_record.copy()
    record["fare_amount"] = 9999.99

    with TestClient(app) as client:
        res = client.post("/predict", json=record)

        assert res.status_code == 200


def test_docs_accessible():
    # just checking swagger loads
    with TestClient(app) as client:
        res = client.get("/docs")

        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]