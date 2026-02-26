# Machine Learning Model for Fraud Detection in MTN Mobile Transactions

### Project Overview

In this project I trained several machine learning models to detect fraudulent mobile money transactions. I started with five baseline models: Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost, and Support Vector Machine. Based on their initial training and test accuracy, I selected the top two performers, XGBoost and Random Forest, for further optimization.

I ran five iterations that included hyperparameter tuning via Grid Search, label balancing with SMOTE, and subsampling from the original dataset. Both models were evaluated on a dataset containing all fraud records and a random sample of safe transactions. XGBoost consistently outperformed Random Forest across all iterations.

**Final result: XGBoost achieved 99% accuracy on both the training and test sets.** Accuracy was measured using the Area Under the Receiver Operating Characteristic Curve **(ROC AUC)**.

### Data

I used Kaggle's PaySim dataset, which simulates mobile money transactions based on one month of real financial logs from a mobile money service operating in an African country. The logs were provided by a multinational company whose platform is currently running in more than 14 countries worldwide.

Dataset source: https://www.kaggle.com/ntnu-testimon/paysim1

**Dataset columns:**

**step** - Maps a unit of time in the real world. In this case, 1 step equals 1 hour. Total steps: 744 (30-day simulation).

**type** - CASH-IN, CASH-OUT, DEBIT, PAYMENT, and TRANSFER.

**amount** - Amount of the transaction in local currency.

**nameOrig** - The customer who initiated the transaction.

**oldbalanceOrg** - The sender's account balance before the transaction.

**newbalanceOrig** - The sender's account balance after the transaction.

**nameDest** - The customer who received the transaction.

**oldbalanceDest** - The recipient's account balance before the transaction. Note: there is no balance information for customers whose names start with M (Merchants).

**newbalanceDest** - The recipient's account balance after the transaction. Note: there is no balance information for customers whose names start with M (Merchants).

**isFraud** - Indicates transactions made by fraudulent agents in the simulation. In this dataset, fraudulent behavior consists of taking control of a customer's account and attempting to empty the funds by transferring to another account and then cashing out.

**isFlaggedFraud** - Flags illegal transfer attempts. An illegal attempt is defined as an attempt to transfer more than 200,000 in a single transaction.

### Project Steps

- 1. Loading Data and EDA
- 2. Feature Engineering
- 3. Machine Learning
    - 3.1. Baseline Models
    - 3.2. Grid Search for Best Hyperparameters
    - 3.3. Dealing with Unbalanced Data
        - 3.3.1. Balancing Data via Oversampling with SMOTE
        - 3.3.2. Subsampling Data from the Original Dataset
        - 3.3.3. Performing SMOTE on the New Data
- 4. Machine Learning Pipeline
- 5. Feature Importance
- 6. Conclusion
- 7. Future Work

## 1. Loading Data and EDA

```python
import os
import math
from numpy import *
import numpy as np
import pandas as pd
import random
import seaborn as sns          # for visualization
import matplotlib.pyplot as plt  # for visualization
```

```python
# Load data
data = pd.read_csv('paysim.csv')
```

```python
# Check for any null values
data.isna().sum().sum()
# 0

# Check for duplicate values
data.duplicated(keep='first').any()
# False
```

There are no null values or duplicate rows, so the dataset is clean from the start.

### Examining the Data by Labels

I filtered the data by the `isFraud` label to compare safe and fraudulent transactions side by side.

```python
# Filter data by label: safe and fraud transactions
safe = data[data['isFraud'] == 0]
fraud = data[data['isFraud'] == 1]

# Plot frequency of transactions for each class over time
plt.figure(figsize=(10, 3))
sns.distplot(safe.step, label="Safe Transaction")
sns.distplot(fraud.step, label='Fraud Transaction')
plt.xlabel('Hour')
plt.ylabel('Number of Transactions')
plt.title('Distribution of Transactions over Time')
plt.legend()
```

Although safe transactions slow down during the 3rd and 4th days and again after the 16th day of the month, fraud transactions occur at a steady pace throughout. In the second half of the month, safe transaction volume drops significantly while fraud activity remains constant.

### Hourly Transaction Amounts

```python
# Use a small portion of the data for the scatter plot
smalldata = data.sample(n=100000, random_state=1)
smalldata = smalldata.sort_index()
smalldata = smalldata.reset_index(drop=True)

# Plot hourly transaction amounts
plt.figure(figsize=(18, 6))
plt.ylim(0, 10000000)
plt.title('Hourly Transaction Amounts')
ax = sns.scatterplot(x="step", y="amount", hue="isFraud", data=smalldata)
```

The plot shows a clear 24-hour seasonality pattern in transaction volume, with clusters of activity mid-cycle. To see if fraud transactions follow the same pattern:

```python
# Hourly amounts for fraud transactions only
plt.figure(figsize=(18, 6))
plt.ylim(0, 10000000)
plt.title('Hourly Fraud Transaction Amounts')
ax = sns.scatterplot(x="step", y="amount", color='orange', data=fraud)
```

Fraud transactions do not show the same day-night pattern. They occur at nearly the same frequency every hour. There are more fraud incidents at lower amounts, but the timing pattern stays consistent throughout the month.

### Transaction Amount Distributions

There is an interesting spike at the $1M mark. Safe transactions also peak more frequently at lower amounts.

```python
# Fraud transaction amount value counts
fraud.amount.value_counts()
```

```
10000000.00    287
0.00            16
429257.45        4
1165187.89       4
76646.05         2
               ...
3576297.10       1
23292.30         1
1078013.76       1
112486.46        1
4892193.09       1
Name: amount, Length: 3977, dtype: int64
```

The highest recurring fraud amount is $10M, which appears 287 times. Fraud transaction amounts range from $119 to $10M and are positively skewed — most frauds involve smaller amounts. There are also 16 fraud-labeled transactions with a $0 amount. While this appears to be noise, it could represent an intentional tactic to mask real fraud activity, so I kept these records in the dataset.

### Type of Transactions

```python
# Checking types of fraud transactions
fraud.type.value_counts()
```

```
CASH_OUT    4116
TRANSFER    4097
Name: type, dtype: int64
```

Fraud only occurs in TRANSFER and CASH_OUT transactions. Since DEBIT, PAYMENT, and CASH_IN transactions contain zero fraud cases, I filtered the dataset to include only TRANSFER and CASH_OUT records for model training.

### Rate of Fraud Transactions

```python
# Proportion of fraud transactions by count
data.isFraud.value_counts()[1] / (data.isFraud.value_counts()[0] + data.isFraud.value_counts()[1])
# 0.001290820448180152
```

```python
# Proportion of fraud transactions by total amount
fraud.amount.sum() / (safe.amount.sum() + fraud.amount.sum())
# 0.010535206008606473
```

Fraud transactions make up only 0.13% of all transactions by count, but account for about 1.05% of the total money transferred — meaning fraudsters tend to target higher-value transactions.

## 2. Feature Engineering

### Filtering by Transaction Type

Since fraud only occurs in TRANSFER and CASH_OUT transactions, I limited the dataset to those types.

```python
# Filter to only TRANSFER and CASH_OUT transactions
data_by_type = data[data['type'].isin(['TRANSFER', 'CASH_OUT'])]
```

The full filtered dataset is too large to be practical for machine learning, so I drew a random subsample of 100,000 records.

```python
# Take a random subsample of 100,000 records
df = data_by_type.sample(n=100000, random_state=1)
df = df.sort_index()
df = df.reset_index(drop=True)
```

### Handling Name Columns

The `nameOrig` and `nameDest` columns represent transaction parties. I checked whether any two parties had repeated transactions, as that could be a useful signal for the classifier.

```python
# Check for repeated transactions between the same two parties
list1 = np.array(df.nameOrig)
list2 = np.array(df.nameDest)
list3 = list1 + list2
repeat = pd.DataFrame(list3, columns=['comb'])
comb_cnt = repeat.comb.value_counts()
comb_cnt.value_counts()

# 1    100000
# Name: comb, dtype: int64
```

Every transaction pair is unique — no repeated combinations exist. These string columns add no predictive value, so I dropped them.

```python
# Drop name columns and binary-encode transaction type
df = df.drop(['nameOrig', 'nameDest'], axis=1)
df.loc[df.type == 'CASH_OUT', 'type'] = 1
df.loc[df.type == 'TRANSFER', 'type'] = 0
```

Note: Some transactions show a balance of $0 for both the old and new balances even when a transaction occurred. This inconsistency exists in the source data and was left as-is for this iteration.

## 3. Machine Learning

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
```

```python
# Split features and target
features = df.drop('isFraud', axis=1)
target = df.isFraud

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
```

### 3.1. Baseline Models

I ran five classifiers with their default parameters to establish a performance baseline. Since the dataset is highly imbalanced (fraud accounts for only ~0.13% of records), I measured performance using the **Area Under the ROC Curve (ROC AUC)** rather than simple accuracy. Confusion matrix accuracy is misleading for imbalanced classification problems.

```python
# General function to run a classifier with default parameters
def ml_func(algorithm):
    model = algorithm()
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_accuracy = roc_auc_score(y_train, train_preds)
    test_accuracy = roc_auc_score(y_test, test_preds)

    print(str(algorithm))
    print(f"Training Accuracy: {(train_accuracy * 100):.4}%")
    print(f"Test Accuracy:     {(test_accuracy * 100):.4}%")
```

```python
# Run all five baseline classifiers
algorithms = [LogisticRegression, KNeighborsClassifier, RandomForestClassifier, XGBClassifier, svm.SVC]

for algorithm in algorithms:
    ml_func(algorithm)
```

### 3.2. Grid Search for Best Hyperparameters

Random Forest had the best training accuracy, while XGBoost had the best test accuracy. I optimized both using Grid Search to find the hyperparameter combinations that yield the most accurate results.

```python
# General function for grid search
def grid_src(classifier, param_grid):
    grid_search = GridSearchCV(classifier, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(str(classifier) + ' Best Parameters')
    print(grid_search.best_params_)
    return grid_search.best_params_
```

```python
# Grid Search for Random Forest
param_grid_rf = {
    'n_estimators': [10, 80, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10],
    'min_samples_split': [2, 3, 4]
}
rf_params = grid_src(RandomForestClassifier(), param_grid_rf)
```

```
Best Parameters:
{'criterion': 'entropy', 'max_depth': 10, 'min_samples_split': 4, 'n_estimators': 10}
```

```python
# Grid Search for XGBoost
param_grid_xg = {
    'n_estimators': [100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 10],
    'colsample_bytree': [0.7, 1],
    'gamma': [0.0, 0.1, 0.2]
}
grid_src(XGBClassifier(), param_grid_xg)
```

```
Best Parameters:
{'colsample_bytree': 1, 'gamma': 0.0, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 100}
```

### Running Models with Best Parameters

```python
# Helper function to train, evaluate, and report a model
def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_accuracy = roc_auc_score(y_train, train_preds)
    test_accuracy = roc_auc_score(y_test, test_preds)
    report = classification_report(y_test, test_preds)

    print('Model Scores')
    print(f"Training Accuracy: {(train_accuracy * 100):.4}%")
    print(f"Test Accuracy:     {(test_accuracy * 100):.4}%")
    print('Classification Report:\n', report)
```

```python
# Random Forest with best parameters
rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10, min_samples_split=3)
run_model(rf_model, X_train, y_train, X_test, y_test)
```

```
Model Scores
Training Accuracy: 85.55%
Test Accuracy:     84.17%

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     19940
           1       1.00      0.68      0.81        60

    accuracy                           1.00     20000
   macro avg       1.00      0.84      0.91     20000
weighted avg       1.00      1.00      1.00     20000
```

The accuracy is lower here because `max_depth=10` limits the tree depth. Without this cap, the model would keep splitting until leaves are pure, which produces higher accuracy but risks overfitting on large datasets. I kept this constraint and focused on improving results through other means.

```python
# XGBoost with best parameters
xgb_model = XGBClassifier(colsample_bytree=1, n_estimators=100, gamma=0.1, learning_rate=0.1, max_depth=5)
run_model(xgb_model, X_train, y_train, X_test, y_test)
```

```
Model Scores
Training Accuracy: 90.83%
Test Accuracy:     87.5%

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     19940
           1       1.00      0.75      0.86        60

    accuracy                           1.00     20000
   macro avg       1.00      0.88      0.93     20000
weighted avg       1.00      1.00      1.00     20000
```

XGBoost clearly performs better with its optimal parameters. The remaining challenge is the severe class imbalance in the target variable.

### 3.3. Dealing with Unbalanced Data

#### 3.3.1. Balancing Data via Oversampling with SMOTE

```python
from imblearn.over_sampling import SMOTE

print(target.value_counts())

# Resample using only the training set
X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train)
print(pd.Series(y_resampled).value_counts())
```

```
0    99722
1      278
Name: isFraud, dtype: int64

1    79782
0    79782
dtype: int64
```

```python
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=0)
```

Running both models with the SMOTE-balanced data:

```python
# Random Forest on SMOTE-balanced data
run_model(rf_model, X_train, y_train, X_test, y_test)
```

```
Model Scores
Training Accuracy: 99.12%
Test Accuracy:     99.01%
```

```python
# XGBoost on SMOTE-balanced data
run_model(xgb_model, X_train, y_train, X_test, y_test)
```

```
Model Scores
Training Accuracy: 99.59%
Test Accuracy:     99.47%
```

Performance increased dramatically for both models, but achieving near-perfect accuracy raised a concern: with so few real fraud records, SMOTE generated a large number of synthetic duplicates. It is likely that identical data points appear in both the training and test sets, causing the model to memorize rather than generalize. To address this, I moved to a more grounded approach.

#### 3.3.2. Subsampling Data from the Original Dataset

Instead of generating synthetic fraud records, I went back to the full original dataset and collected all real fraud instances. I then paired them with a random sample of 50,000 safe transactions to create a less skewed, entirely organic dataset.

```python
# Filter to fraud-relevant transaction types
data2 = data[data['type'].isin(['TRANSFER', 'CASH_OUT'])]
safe_2 = data2[data2['isFraud'] == 0]
fraud_2 = data2[data2['isFraud'] == 1]

# Take 50,000 random safe transactions
safe_sample = safe_2.sample(n=50000, random_state=1).sort_index().reset_index(drop=True)

# Combine with all fraud records
df3 = pd.concat([safe_sample, fraud_2]).reset_index(drop=True)

# Drop name columns and encode type
df3 = df3.drop(['nameOrig', 'nameDest'], axis=1)
df3.loc[df3.type == 'CASH_OUT', 'type'] = 1
df3.loc[df3.type == 'TRANSFER', 'type'] = 0
```

```python
df3.isFraud.value_counts()
```

```
0    50000
1     8213
Name: isFraud, dtype: int64
```

The new dataset is entirely organic with a more reasonable class distribution. It is still imbalanced, but significantly less so.

```python
features2 = df3.drop('isFraud', axis=1)
target2 = df3.isFraud
X_train2, X_test2, y_train2, y_test2 = train_test_split(features2, target2, test_size=0.2)
```

```python
# Random Forest on subsampled organic data
run_model(rf_model, X_train2, y_train2, X_test2, y_test2)
```

```
Model Scores
Training Accuracy: 93.8%
Test Accuracy:     93.45%
```

```python
# XGBoost on subsampled organic data
run_model(xgb_model, X_train2, y_train2, X_test2, y_test2)
```

```
Model Scores
Training Accuracy: 99.4%
Test Accuracy:     98.92%
```

These results look much more realistic. XGBoost maintains a strong lead over Random Forest on real-world data. The dataset is still somewhat imbalanced, so I applied SMOTE on top of this new subset to see if it further improves performance.

#### 3.3.3. Performing SMOTE on the New Data

```python
from imblearn.over_sampling import SMOTE

print(target2.value_counts())

X_resampled2, y_resampled2 = SMOTE().fit_sample(X_train2, y_train2)
print(pd.Series(y_resampled2).value_counts())

X_train2, X_test2, y_train2, y_test2 = train_test_split(features2, target2, test_size=0.2)
```

```
0    50000
1     8213
Name: isFraud, dtype: int64

1    39941
0    39941
dtype: int64
```

```python
# Random Forest on subsampled + SMOTE data
run_model(rf_model, X_train2, y_train2, X_test2, y_test2)
```

```
Model Scores
Training Accuracy: 93.77%
Test Accuracy:     92.47%
```

```python
# XGBoost on subsampled + SMOTE data
run_model(xgb_model, X_train2, y_train2, X_test2, y_test2)
```

```
Model Scores
Training Accuracy: 99.45%
Test Accuracy:     98.66%
```

XGBoost improved slightly while Random Forest accuracy decreased. This suggests that Random Forest is more sensitive to repeated synthetic data, whereas XGBoost handles the oversampling more gracefully.

## 4. Machine Learning Pipeline

Pipelines allow you to chain preprocessing and modeling steps into a single, clean workflow. In this pipeline, MinMaxScaler normalizes the feature values, PCA reduces dimensionality, and XGBoost performs the final classification.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
```

```python
# Build the pipeline
pipe = Pipeline([
    ('scl', MinMaxScaler()),
    ('pca', PCA(n_components=7)),
    ('xgb', XGBClassifier())
])

# Define the parameter grid for the XGBoost step
grid = [{'xgb__n_estimators': [100],
         'xgb__learning_rate': [0.05, 0.1],
         'xgb__max_depth': [3, 5, 10],
         'xgb__colsample_bytree': [0.7, 1],
         'xgb__gamma': [0.0, 0.1, 0.2]}]

# Run Grid Search over the pipeline
gridsearch = GridSearchCV(estimator=pipe, param_grid=grid, scoring='accuracy', cv=3)
gridsearch.fit(X_train, y_train)

print('Best accuracy: %.3f' % gridsearch.best_score_)
print('\nBest params:\n', gridsearch.best_params_)
```

```
Best accuracy: 0.995

Best params:
 {'xgb__colsample_bytree': 0.7, 'xgb__gamma': 0.1, 'xgb__learning_rate': 0.1, 'xgb__max_depth': 10, 'xgb__n_estimators': 100}
```

## 5. Feature Importance

To understand which features drive the model's decisions, I plotted the relative importance of each feature according to XGBoost.

```python
from xgboost import plot_importance

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax = plot_importance(xgb_model, height=0.5, color='orange', grid=False,
                     show_values=False, importance_type='cover', ax=ax)

for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

ax.set_xlabel('Relative Feature Importance for XGBoost', size=12)
ax.set_yticklabels(ax.get_yticklabels(), size=12)
ax.set_title('Feature Importance Order', size=16)
```

The two most influential features are `oldbalanceOrg` (the sender's balance before the transaction) and `newbalanceDest` (the recipient's balance after the transaction). This makes intuitive sense — fraudulent transfers tend to drain an account entirely and land in a receiving account with an unusually high post-transaction balance.

## 6. Conclusion

After five iterations of training, tuning, and data rebalancing, the final model achieved the following:

### 99% Accuracy with XGBoost Classifier and Balanced Data

### Most Influential Features

- The sender's balance before the transaction (`oldbalanceOrg`) and the recipient's balance after the transaction (`newbalanceDest`) are the strongest fraud indicators.

### Key EDA Findings

- Safe transaction volume drops significantly in the second half of the month, but fraud transaction frequency remains constant throughout.
- Fraud accounts for only 0.13% of all transactions by count, but represents approximately 1.05% of total transaction volume by amount.
- Safe transactions follow a clear 24-hour seasonality pattern. Fraud transactions do not — they occur at nearly the same rate every hour.
- Most fraud transactions involve smaller amounts, but there is a notable spike at exactly $1M.
- Fraud transaction amounts range from $119 to $10M. There are also 16 fraud-labeled records with a $0 amount, which were retained as potential noise signals.
- Fraud only occurs in TRANSFER and CASH_OUT transactions. DEBIT, PAYMENT, and CASH_IN are completely safe.

## 7. Future Work

- Reindexing the dataset with real timestamps would allow for proper time-series analysis. This could reveal seasonality patterns in both the frequency and amount of fraud transactions that are not visible in the current hourly step format.
- Time-series forecasting on fraud activity could help financial companies anticipate high-risk periods and apply extra monitoring proactively.
