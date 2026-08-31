# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions in a highly imbalanced dataset, comparing multiple models and imbalance-handling techniques.

## Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 284,807 transactions over two days, of which only 492 (0.17%) are fraudulent
- Features `V1`–`V28` are PCA-transformed components (anonymized for confidentiality); `Time` and `Amount` are the only original, unaltered features

## Problem Framing

With fraud making up just 0.17% of transactions, this is a severely imbalanced classification problem. A model that predicts "not fraud" for every transaction would score 99.8% accuracy while being completely useless. Because of this, **accuracy is not used as an evaluation metric** — precision, recall, F1-score, and PR-AUC are used instead, since they reflect performance on the minority (fraud) class specifically.

## Approach

1. **EDA** — examined class distribution, `Amount` distribution by class, and V-feature separation between fraud and normal transactions
2. **Feature engineering** — derived `hour` of day from `Time`; scaled `Amount` using `RobustScaler` (fit only on training data to avoid data leakage)
3. **Train/test split** — 80/20 split, stratified by class to preserve the fraud ratio in both sets
4. **Models trained and compared:**
   - Logistic Regression (baseline)
   - Logistic Regression + `class_weight='balanced'`
   - Logistic Regression + SMOTE (synthetic oversampling)
   - Random Forest + SMOTE
   - XGBoost + SMOTE

## Results

| Model | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| LogReg baseline | 0.64 | 0.82 | 0.72 | 0.958 | 0.737 |
| LogReg + class_weight | 0.91 | 0.06 | 0.11 | 0.972 | 0.721 |
| LogReg + SMOTE | 0.91 | 0.06 | 0.10 | 0.972 | 0.728 |
| Random Forest + SMOTE | 0.83 | 0.85 | 0.84 | 0.973 | 0.855 |
| XGBoost + SMOTE | 0.87 | 0.73 | 0.79 | 0.981 | 0.867 |

**Final model: Random Forest + SMOTE** — selected for its best F1-score (0.84) and lowest false-positive count (14) at default threshold, correctly catching 81 of 98 fraud cases without significantly disrupting legitimate transactions.

## Key Findings

- **ROC-AUC was misleading for this dataset.** All five models scored within a narrow 0.958–0.981 range, which would wrongly suggest similar performance. **PR-AUC** told a much clearer story, since it isn't inflated by the large number of easy true negatives.
- **Logistic Regression could not escape a precision/recall tradeoff**, regardless of whether the imbalance was handled via `class_weight` or SMOTE — both techniques pushed the linear decision boundary in a similar direction, producing near-identical results (91% recall, but only 6% precision).
- **Random Forest, a non-linear model, broke past that tradeoff** when combined with SMOTE — achieving strong recall (0.83) without sacrificing precision (0.85), the best balance among all models tested.
- **XGBoost achieved the highest PR-AUC/ROC-AUC**, indicating strong overall ranking ability, but its default decision threshold produced significantly more false positives (31) than Random Forest (14) for a similar recall — making Random Forest the more practical choice without further threshold tuning.
- **Feature importance** (Random Forest) showed `V14`, `V4`, and `V10` as the strongest fraud predictors — despite these being anonymized PCA components with no directly interpretable real-world meaning.

## What I'd Improve Next

- Threshold tuning on XGBoost to test whether its higher PR-AUC translates into better practical performance
- Try `scale_pos_weight` (XGBoost's built-in imbalance handling) as an alternative to SMOTE
- Address a minor data-leakage issue caught during development, where `Amount` was initially scaled before the train/test split — later corrected by refitting the scaler on training data only
- Deploy the final model behind a simple API or Streamlit app for interactive testing

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook CreditCardFraud.ipynb
```

Download the dataset from Kaggle and place `creditcard.csv` in the project folder before running.

## Files

- `CreditCardFraud.ipynb` — full analysis and model training notebook
- `rf_fraud_model.pkl` — final trained Random Forest model
- `amount_scaler.pkl` — fitted RobustScaler for the `Amount` feature
- `requirements.txt` — Python dependencies
