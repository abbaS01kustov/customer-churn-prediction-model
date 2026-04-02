# Customer Churn Prediction ML Project

## Data Preparation

### 1. Data Ingestion

* Imported required Python libraries for data manipulation and modeling
* Loaded the dataset from a CSV file into a Pandas DataFrame
* Used `.T` (transpose) where necessary for improved readability during inspection

### 2. Data Cleaning & Standardization

To ensure consistency and model readiness:

* Standardized column names (lowercase, replaced spaces with underscores)
* Normalized categorical values for uniformity
* Verified correct data types across all columns

### 3. Target Variable Encoding

* Converted `churn` from categorical (`"yes"/"no"`) to numerical:

  * `1` → Churn (customer left)
  * `0` → Retained

This transformation enables compatibility with machine learning algorithms.

## Validation Framework

### Train / Validation / Test Split
We implemented a **three-way split** using `train_test_split`:
* **60%** → Training
* **20%** → Validation
* **20%** → Test

Steps:
1. Split full dataset into:
   * 80% training (full_train)
   * 20% test
2. Split full_train into:
   * 75% train
   * 25% validation (i.e., 20% of total data)

Additional steps:
* Reset indices using `reset_index(drop=True)`
* Separated features (`X`) and target (`y`)
* Removed `churn` from feature datasets to prevent leakage

## Exploratory Data Analysis (EDA)

### Missing Values
* Verified that the dataset contains **no missing values**

### Target Distribution
* Total customers: **5634**
* Churned: **1521 (~27%)**
* Retained: **4113 (~73%)**

Insight: The dataset is **imbalanced**, which impacts model evaluation strategy.

### Churn Rate
* Computed using:
  * `value_counts(normalize=True)`
  * `.mean()` (valid for binary variables)

### Feature Types
* **Numerical Features**:
  * `tenure`, `monthlycharges`, `totalcharges`
* **Categorical Features**:
  * All remaining variables

## Feature Importance

### 1. Churn Rate by Group
* Compared churn rates across categorical segments
* Identified high-risk groups (e.g., contract type)

### 2. Risk Ratio
* Measures relative likelihood of churn:

  [
  \text{Risk Ratio} = \frac{\text{Group Churn Rate}}{\text{Global Churn Rate}}
  ]

* Interpretation:
  * > 1 → Higher churn risk
  * < 1 → Lower churn risk

### 3. Mutual Information
* Quantifies how much information a feature provides about churn
* Enables ranking of categorical variables by importance

### 4. Correlation (Numerical Features)
* Used **Pearson correlation coefficient**
* Evaluated strength of linear relationship with churn
* Ranked features using absolute correlation values

## Feature Engineering

### One-Hot Encoding
* Converted categorical variables into numeric format using `DictVectorizer`
* Workflow:

  1. Convert DataFrame → list of dictionaries (`to_dict(orient='records')`)
  2. Fit `DictVectorizer` on training data
  3. Transform validation/test data using same encoder

## Model: Logistic Regression

### Why Logistic Regression?
* Suitable for **binary classification**
* Outputs probabilities (0–1), not just class labels
* Fast, interpretable, and effective baseline model

### Model Formulation
[
g(x) = \sigma(w_0 + w^T x)
]

Where:
* ( \sigma(z) = \frac{1}{1 + e^{-z}} ) (sigmoid function)

## Model Training
* Trained using Scikit-Learn’s `LogisticRegression`
* Extracted:
  * Coefficients (`coef_`)
  * Intercept (`intercept_`)

### Predictions
* **Hard predictions** → `predict()`
* **Probabilities** → `predict_proba()[:, 1]`

## Model Evaluation
### Accuracy
* Achieved ~**80% accuracy** on validation set
Limitation:
* Baseline (dummy model) achieves ~73% → Accuracy alone is insufficient.

## Advanced Metrics
### Confusion Matrix
* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

### Precision & Recall
* **Precision** → Accuracy of positive predictions
* **Recall** → Ability to detect actual churners

### F1 Score
* Harmonic mean of precision and recall

## ROC Curve & AUC
### ROC Curve
* Plots:
  * True Positive Rate (TPR)
  * False Positive Rate (FPR)

### AUC (Area Under Curve)
* 0.5 → Random model
* 1.0 → Perfect model
* Model performance:
  * ~0.8 → Good

Interpretation:
Probability that the model ranks a random churner higher than a non-churner.

## Cross-Validation
### K-Fold Strategy
* Split data into K subsets
* Train on K-1 folds, validate on remaining fold
* Repeat K times

### Results
* Mean AUC: ~**0.84**
* Low standard deviation → Stable model

## Hyperparameter Tuning
### Regularization Parameter (`C`)
* Controls model complexity:
  * Small `C` → Strong regularization
  * Large `C` → Less regularization

### Approach
* Tested multiple `C` values
* Selected optimal value based on validation AUC

## Deployment Perspective
* Model accepts **customer data as input (dictionary)**
* Outputs:
  * Churn probability score
* Enables:
  * Real-time predictions
  * Targeted retention campaigns

## Business Impact
This model enables:
* Proactive churn prevention
* Optimized marketing spend
* Increased customer lifetime value (CLV)
