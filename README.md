# Credit Card Fraud Detection

Detect fraudulent credit-card transactions on a highly imbalanced dataset using classical ML and PyCaret AutoML.

Fraud is ~0.17% of rows, so accuracy is misleading. The notebook compares models with precision, recall, F1, and AUPRC.

## Skills

- Exploratory data analysis on anonymized PCA features (`V1`–`V28`, `Amount`, `Time`)
- Class-imbalance awareness (accuracy vs. precision-recall)
- Isolation Forest / Local Outlier Factor style anomaly checks
- Supervised baselines (Logistic Regression, Random Forest)
- PyCaret classification AutoML

## Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place `creditcard.csv` in this folder. It is **not** committed (147 MB, over GitHub’s 100 MB file limit).

```bash
# after downloading from Kaggle
move creditcard.csv credit-card-fraud-detection\
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

## Structure

```
credit-card-fraud-detection/
├── Credit_Card_Fraud_Detection.ipynb
├── requirements.txt
└── README.md
```
