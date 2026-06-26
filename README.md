# PREDICTING-CUSTOMER-LIFETIME-VALUE-IN-DIGITAL-COMMERCE-USING-MACHINE-LEARNING-AND-SURVIVAL-ANALYSIS-
# Predicting Customer Lifetime Value in Digital Commerce Using Machine Learning and Survival Analysis

**Gisma University of Applied Sciences — MSc Data Science, AI, and Digital Business**  
**Author:** Jnanadeep Aravapalli (GH ID: GH1043106)  
**Supervisor:** Amirhossein Jamalian  
**Academic Year:** 2025–2026

---

## Overview

This repository contains the full implementation and analysis conducted for a master's thesis focused on predicting Customer Lifetime Value (CLV) in digital commerce environments. The study integrates RFM-based feature engineering with machine learning regression models and Cox Proportional Hazards survival analysis to model both future customer spending and churn behaviour from historical transaction data.

---

## Research Objectives

- Preprocess and explore customer transactional data from a real-world online retail dataset
- Engineer customer-level behavioural features (Recency, Frequency, Monetary Value, Tenure, and more)
- Develop and compare predictive models for estimating future Customer Lifetime Value
- Apply survival analysis to understand and predict customer churn behaviour
- Evaluate model effectiveness across multiple performance metrics

---

## Research Questions

1. How accurately can machine learning models predict Customer Lifetime Value from customer behavioural features?
2. How can survival analysis improve customer lifetime and churn prediction?
3. What are the key behavioural drivers of customer value and retention in digital commerce?

---

## Dataset

The study uses the **Online Retail II** dataset — a publicly available dataset containing over **525,000 transaction records** from a UK-based online retailer (December 2009 – December 2010).

> Due to size constraints, the full dataset is not included in this repository. A sample is provided for demonstration.

**Full dataset:** [Kaggle — Online Retail Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/online-retail-dataset)

**Key raw variables:**

| Field | Description |
|---|---|
| `Customer ID` | Unique customer identifier |
| `InvoiceDate` | Transaction date and time |
| `Invoice` | Invoice number |
| `StockCode` | Product code |
| `Quantity` | Units purchased |
| `Price` | Unit price |
| `Country` | Customer country |

---

## Methodology

### Data Preprocessing

- Removed records with missing `Customer ID` (107,927 dropped)
- Removed duplicate transactions
- Filtered out negative quantities (returns/cancellations)
- Standardised `InvoiceDate` to datetime format
- Derived `TotalPrice = Quantity × Price`

After cleaning: **400,947 valid transactions** retained across **9 variables**.

### Chronological Data Partitioning

To prevent **data leakage**, the dataset was split chronologically:

| Split | Records |
|---|---|
| Historical (features) | 252,765 |
| Future (CLV target) | 148,182 |

The last 3 months of the observation window (~one quarter) were reserved as the future period — consistent with retail industry churn benchmarks.

### Feature Engineering

Nine behavioural features were engineered from historical transactions:

| Feature | Description |
|---|---|
| `Recency` | Days since last purchase |
| `Frequency` | Number of distinct transactions |
| `Monetary` | Total historical spend |
| `Tenure` | Days between first and last purchase |
| `AvgPurchaseInterval` | Mean days between purchases |
| `UniqueProducts` | Number of distinct products bought |
| `AvgOrderValue` | Average spend per transaction |
| `TotalQuantity` | Total units purchased |
| `StdGap` | Standard deviation of purchase gaps |
| `Momentum` | Frequency / (Tenure + 1) |

The **target variable (CLV)** was computed as total spend in the future window — fully isolated from the feature set.

Final modelling dataset: **1,946 customers**, 9 features, 0 missing values.

### Modelling Approaches

#### Machine Learning — CLV Prediction

Three regression models were trained and evaluated using **5-fold TimeSeriesSplit cross-validation** and **RandomizedSearchCV** for hyperparameter tuning:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **829.36** | **2633.56** | **0.7188** |
| Random Forest | 841.98 | 3225.49 | 0.5783 |
| XGBoost | 946.48 | 3982.71 | 0.3570 |

Linear Regression achieved the best performance across all metrics, explaining ~71.88% of variance in future customer spend. The largely linear relationship between engineered behavioural features and future spending, combined with the relatively small modelling dataset (1,946 customers), likely contributed to overfitting in the tree-based models.

**Evaluation metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R²

#### Feature Importance

Feature importance was assessed via the Random Forest model:

| Feature | Importance |
|---|---|
| TotalQuantity | 76.46% |
| Frequency | 7.02% |
| Tenure | 4.15% |
| Momentum | 3.75% |
| AvgOrderValue | 3.10% |
| Others | <2% each |

> ⚠️ The dominant importance of `TotalQuantity` may reflect multicollinearity with other volume-based features (Frequency, Monetary). Future work should investigate dimensionality reduction (e.g., PCA) to address this.

#### Survival Analysis — Churn Prediction

A **Cox Proportional Hazards model** was fitted on 1,946 customers (656 churned, 1,290 retained), using a **90-day inactivity threshold** to define churn.

| Variable | Coefficient | Hazard Ratio | Interpretation |
|---|---|---|---|
| Recency | +0.02 | 1.02 | Higher recency → increased churn risk |
| Frequency | −0.71 | 0.49 | More purchases → 51% lower churn risk |
| AvgPurchaseInterval | −0.21 | 0.81 | More regular buyers → lower churn risk |
| UniqueProducts | −0.01 | 0.99 | Greater product diversity → slightly lower churn risk |

All variables significant at **p < 0.005**. Model **concordance index (C-index): 0.96**.

> ⚠️ The high C-index should be interpreted cautiously given the class imbalance and the dominance of Frequency as a predictor. External validation on larger datasets is recommended.

---

## Key Findings

- **Linear Regression** outperformed ensemble models (R² = 0.7188), suggesting the CLV–behaviour relationship in this retail context is largely linear
- **TotalQuantity** was the strongest predictor of future customer value (76.46% feature importance)
- **Purchase Frequency** was the most powerful churn deterrent — highly active customers were 51% less likely to churn
- **Recency** was positively associated with churn risk, reinforcing the importance of consistent customer re-engagement
- Product diversity and purchase regularity also contributed meaningfully to retention

---

## Limitations

- Single dataset from one UK retailer; findings may not generalise across sectors or geographies
- No demographic, browsing, or marketing engagement data available
- Churn threshold (90 days) is rule-based; alternative definitions may yield different survival results
- Observation window limited to ~1 year; longer horizons may reveal additional behavioural patterns
- Potential multicollinearity among volume-related features (TotalQuantity, Frequency, Monetary)

---

## Future Work

- Apply **deep learning models** (LSTM, Transformer-based) on transaction-level sequential data to capture temporal purchase patterns
- Explore **DeepSurv** and neural network survival models for nonlinear churn modelling
- Extend the feature set with demographic data, product categories, and marketing interactions
- Conduct **variance inflation factor (VIF) analysis** to address multicollinearity in features
- Deploy the framework in a live CRM environment to measure real-world retention impact

---

## Repository Structure

```
CLV-Prediction-Project/
│
├── notebook/
│   └── clv_prediction_model.ipynb
│
├── data/
│   └── sample_data.csv
│
├── report/
│   └── CLV_Prediction_Report.pdf
│
├── requirements.txt
└── README.md
```

---

## Reproducibility

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the notebook:
   ```bash
   jupyter notebook notebook/clv_prediction_model.ipynb
   ```

**Core libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lifelines`, `matplotlib`, `seaborn`

---

## Citation

If you reference this work, please cite:

> Aravapalli, J. (2026). *Predicting Customer Lifetime Value in Digital Commerce Using Machine Learning and Survival Analysis*. Master's Thesis, Gisma University of Applied Sciences.

---

## License

This repository is intended for **academic and research purposes only**. The implementation reflects the methodological considerations and limitations discussed in the accompanying dissertation.
