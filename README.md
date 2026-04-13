
# Customer Churn Prediction API

### End-to-End Machine Learning System for Proactive Customer Retention

## Project Overview

Customer churn is a critical business problem that directly impacts revenue and growth. This project delivers an **end-to-end machine learning solution** that predicts the likelihood of customer churn and exposes the model via a **production-ready REST API**.

From a data leadership perspective, this system is designed not just for modeling accuracy, but for **real-world deployment and business decision-making**.

### Key Objectives

* Predict customer churn probability using structured customer data
* Enable real-time scoring via API
* Support proactive retention strategies (e.g., targeted promotions)
* Deliver a scalable, reproducible ML pipeline

## Machine Learning Approach

* **Model**: Logistic Regression (regularized)
* **Feature Engineering**:

  * Categorical encoding via `DictVectorizer`
  * Numerical feature integration
* **Validation Strategy**:

  * Stratified K-Fold Cross Validation (to handle class imbalance)
* **Evaluation Metric**:

  * ROC AUC (robust for imbalanced classification)


## Model Performance

| Metric    | Value               |
| --------- | ------------------- |
| ROC AUC   | **~0.84**           |
| Stability | ± 0.01 across folds |

### Interpretation

* The model demonstrates **strong discriminatory power**
* Performance is consistent across validation folds
* Indicates a **reliable baseline model** suitable for deployment

## API Architecture
This project includes a **Flask-based prediction service**:

```text
Client (predict.py / external system)
        ↓
HTTP POST Request
        ↓
Flask API (/predict)
        ↓
Model Inference (Logistic Regression)
        ↓
JSON Response (probability + decision)
```


## Running the API Locally

### 1. Clone repository

```bash
git clone https://github.com/abbaS01kustov/customer-churn-project.git
cd customer-churn-project
```

### 2. Activate environment

```bash
source venv/bin/activate
```

### 3. Start API server

```bash
python app/app.py
```

Server will run at:

```text
http://localhost:9696
```

## API Usage Example

### Endpoint

```http
POST /predict
```

### Sample Request (Python)

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

### Sample Response

```json
{
  "churn_probability": 0.6283,
  "churn": true
}
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


## Example Output (CLI)

```text
{'churn_probability': 0.6283, 'churn': True}
sending promo email to xyz-123
```

## Project Structure

```text
customer_churn/
│
├── app/
│   ├── app.py          # Flask API
│   └── predict.py      # Client script
├── model_C=1.0.bin     # Trained model
├── train.ipynb         # Model development
├── requirements.txt
├── README.md
└── .gitignore
```

## Business Impact

This system enables:

* **Targeted retention campaigns**
* Reduced customer acquisition cost
* Data-driven decision making
* Real-time scoring for operational systems


## Future Improvements

* Model upgrade (XGBoost / LightGBM)
* Feature enrichment (behavioral + temporal features)
* Threshold optimization based on ROI
* Deployment (Docker + Cloud)
* Frontend dashboard for business users


