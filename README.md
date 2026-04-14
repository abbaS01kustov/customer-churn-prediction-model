# Customer Churn Prediction API

### End-to-End Machine Learning System with Dockerized Deployment

## Project Overview

Customer churn directly impacts revenue, making early prediction critical for retention strategies. This project delivers a **production-ready machine learning system** that predicts churn probability and serves predictions via a **Dockerized REST API**.

From a data leadership perspective, the system emphasizes:

* **Reproducibility (Docker)**
* **Scalability (API-first design)**
* **Business impact (actionable predictions)**


## Key Objectives

* Predict customer churn probability using structured data
* Serve predictions via a **REST API**
* Ensure **environment consistency using Docker**
* Enable real-time decision-making for retention strategies


## Machine Learning Approach

* **Model**: Regularized Logistic Regression
* **Feature Engineering**:

  * `DictVectorizer` for categorical encoding
  * Numerical feature integration
* **Validation**:

  * Stratified K-Fold Cross Validation
* **Evaluation Metric**:

  * ROC AUC (robust for imbalanced classification)
    

## Model Performance

| Metric    | Value     |
| --------- | --------- |
| ROC AUC   | **~0.84** |
| Stability | ±0.01     |

### Interpretation

* Strong separation between churners and non-churners
* Consistent performance across folds
* Reliable baseline for production deployment


## System Architecture

```text
Client (API request)
        ↓
Docker Container (Flask + Gunicorn)
        ↓
Model Inference (Logistic Regression)
        ↓
JSON Response (Probability + Decision)
```


# Dockerized Deployment

## 🔧 Build Docker Image

```bash
docker build -t churn-app .
```

## Run Container

```bash
docker run -it --rm -p 9696:9696 churn-app
```

## Access API

```text
http://localhost:9696
```

## Why Docker Matters Here

* Eliminates "works on my machine" issues
* Ensures identical environments across dev, test, production
* Enables seamless cloud deployment
* Simplifies dependency management


# API Usage

## Endpoint

```http
POST /predict
```

## Python Example

```python
import requests

url = "http://localhost:9696/predict"

customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}

response = requests.post(url, json=customer)
print(response.json())
```

## cURL Example

```bash
curl -X POST http://localhost:9696/predict \
-H "Content-Type: application/json" \
-d '{
  "tenure": 1,
  "monthlycharges": 29.85,
  "totalcharges": 29.85,
  "contract": "month-to-month",
  "internetservice": "dsl"
}'
```

## Sample Response

```json
{
  "churn_probability": 0.6283,
  "churn": true
}
```


# Project Structure

```text
customer_churn/
│
├── app/
│   ├── predict.py        # API logic
│   └── app.py            # Flask entry point
├── model_C=1.0.bin       # Trained model
├── train.ipynb           # Model development
├── Dockerfile            # Container definition
├── requirements.txt      # Dependencies
├── README.md
└── .gitignore
```


# Business Impact

This system enables:

* Targeted retention campaigns
* Improved decision-making
* Reduced churn rate
* Improved customer lifetime value
* Real-time scoring in production systems


# Future Improvements

* Upgrade model (XGBoost / LightGBM)
* Add feature store (behavioral + temporal data)
* Deploy to cloud (AWS/GCP/Azure)
* Add monitoring & logging
* Build frontend dashboard
