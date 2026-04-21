"""
StudyPush - Daily Data Science Article Generator
Generates a random data science article/note and commits it to the repo.
"""

import random
import datetime
import os

# ─────────────────────────────────────────────────────────────
# 100 Data Science Topics (organized by category)
# ─────────────────────────────────────────────────────────────

TOPICS = {
    # ── Machine Learning Fundamentals ──
    "Linear Regression": {
        "category": "Machine Learning",
        "tags": ["regression", "supervised-learning", "statistics"],
        "content": """
## What is Linear Regression?

Linear Regression is one of the most fundamental algorithms in machine learning and statistics. It models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to the observed data.

### The Math Behind It

The simple linear regression equation is:

```
y = β₀ + β₁x + ε
```

Where:
- **y** = dependent variable (target)
- **x** = independent variable (feature)
- **β₀** = y-intercept
- **β₁** = slope coefficient
- **ε** = error term

### Key Assumptions

1. **Linearity** — The relationship between X and Y is linear
2. **Independence** — Observations are independent of each other
3. **Homoscedasticity** — Constant variance of residuals
4. **Normality** — Residuals are normally distributed

### Python Implementation

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)

print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"R² Score: {model.score(X, y):.2f}")
```

### When to Use

- Predicting continuous outcomes (house prices, sales forecasting)
- Understanding feature importance and direction of relationships
- As a baseline model before trying complex algorithms

### Common Pitfalls

- Multicollinearity among features
- Overfitting with too many features
- Ignoring outliers that can skew the regression line

### Further Reading

- *An Introduction to Statistical Learning* — James, Witten, Hastie, Tibshirani
- Scikit-learn documentation on Linear Models
"""
    },

    "Decision Trees": {
        "category": "Machine Learning",
        "tags": ["classification", "regression", "tree-models"],
        "content": """
## What are Decision Trees?

Decision Trees are versatile supervised learning algorithms that can perform both classification and regression tasks. They work by recursively splitting the data based on feature values to create a tree-like model of decisions.

### How It Works

1. **Select the best feature** to split on (using Gini impurity or Information Gain)
2. **Create a decision node** based on that feature
3. **Recursively split** child nodes until a stopping criterion is met
4. **Assign leaf nodes** with the majority class (classification) or mean value (regression)

### Splitting Criteria

**Gini Impurity:**
```
Gini(S) = 1 - Σ(pᵢ²)
```

**Information Gain (Entropy):**
```
Entropy(S) = -Σ pᵢ log₂(pᵢ)
IG(S, A) = Entropy(S) - Σ (|Sᵥ|/|S|) × Entropy(Sᵥ)
```

### Python Implementation

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

print(f"Accuracy: {clf.score(X_test, y_test):.2f}")
```

### Pros & Cons

| Pros | Cons |
|------|------|
| Easy to interpret | Prone to overfitting |
| No feature scaling needed | Unstable (small data changes → different tree) |
| Handles both numerical & categorical | Biased toward features with more levels |

### Key Hyperparameters

- `max_depth` — Limits tree depth to prevent overfitting
- `min_samples_split` — Minimum samples to split a node
- `min_samples_leaf` — Minimum samples in a leaf node
"""
    },

    "Random Forest": {
        "category": "Machine Learning",
        "tags": ["ensemble", "classification", "regression"],
        "content": """
## Random Forest

Random Forest is an ensemble learning method that builds multiple decision trees and merges their predictions. It uses **bagging** (Bootstrap Aggregating) and **feature randomness** to create an uncorrelated forest of trees.

### How It Works

1. Create `n` bootstrap samples from the training data
2. Build a decision tree for each sample, using a random subset of features at each split
3. Aggregate predictions:
   - **Classification**: Majority vote
   - **Regression**: Average

### Why It Works So Well

The magic is in **decorrelation**. By forcing each tree to consider different features, the trees make different errors. When averaged, these errors cancel out — reducing variance without increasing bias.

### Python Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X, y)

# Feature importance
for i, imp in enumerate(rf.feature_importances_):
    if imp > 0.05:
        print(f"Feature {i}: {imp:.3f}")
```

### Feature Importance

Random Forest provides built-in feature importance, making it great for feature selection. However, be aware of:
- **Bias toward high-cardinality features** — Use permutation importance instead
- **Correlated features** — Importance gets split among them

### Tuning Tips

- Start with `n_estimators=100`, increase until diminishing returns
- Use `max_features='sqrt'` for classification, `'auto'` for regression
- Set `min_samples_leaf=5` as a good starting point to prevent overfitting
"""
    },

    "Gradient Boosting (XGBoost)": {
        "category": "Machine Learning",
        "tags": ["boosting", "ensemble", "xgboost"],
        "content": """
## Gradient Boosting & XGBoost

Gradient Boosting builds an ensemble of weak learners (usually decision trees) **sequentially**, where each new tree corrects the errors of the previous ones.

### The Core Idea

```
F₀(x) = initial prediction (e.g., mean)
Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)
```

Where `hₘ(x)` is a tree fit to the **negative gradient** (residuals) of the loss function.

### XGBoost Advantages

- **Regularization** — L1 and L2 penalties to prevent overfitting
- **Handling missing values** — Built-in sparsity awareness
- **Parallel processing** — Despite being sequential, tree building is parallelized
- **Tree pruning** — Uses max_depth and then prunes, unlike traditional greedy approach

### Python Implementation

```python
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

print(f"RMSE: {model.score(X_test, y_test):.3f}")
```

### XGBoost vs LightGBM vs CatBoost

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Speed | Fast | Fastest | Moderate |
| Categorical features | Manual encoding | Native support | Best native support |
| Missing values | Native handling | Native handling | Native handling |
| Default performance | Good | Good | Often best out-of-box |
"""
    },

    "K-Nearest Neighbors (KNN)": {
        "category": "Machine Learning",
        "tags": ["classification", "instance-based", "non-parametric"],
        "content": """
## K-Nearest Neighbors (KNN)

KNN is a simple, instance-based learning algorithm. It classifies a data point based on how its neighbors are classified. It's a **lazy learner** — it doesn't build a model, it memorizes the training data.

### How It Works

1. Choose the number of neighbors **K**
2. Calculate the distance from the new point to all training points
3. Select the K nearest neighbors
4. For classification: majority vote. For regression: average.

### Distance Metrics

```
Euclidean:  d = √(Σ(xᵢ - yᵢ)²)
Manhattan:  d = Σ|xᵢ - yᵢ|
Minkowski:  d = (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
```

### Choosing K

- **Small K** (e.g., 1-3): Low bias, high variance (overfitting risk)
- **Large K** (e.g., 20+): High bias, low variance (underfitting risk)
- **Rule of thumb**: K = √n (where n is the number of training samples)
- Always use **odd K** for binary classification to avoid ties

### Python Example

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# CRITICAL: Always scale features for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train_scaled, y_train)
print(f"Accuracy: {knn.score(X_test_scaled, y_test):.2f}")
```

### When to Use KNN

- Small to medium datasets
- When decision boundaries are irregular
- When you need a quick baseline
- **Avoid** with high-dimensional data (curse of dimensionality)
"""
    },

    "Support Vector Machines (SVM)": {
        "category": "Machine Learning",
        "tags": ["classification", "kernel-methods", "supervised-learning"],
        "content": """
## Support Vector Machines

SVMs find the **optimal hyperplane** that maximizes the margin between classes. They are powerful for both linear and non-linear classification tasks.

### Key Concepts

- **Support Vectors** — The closest data points to the decision boundary
- **Margin** — The distance between the hyperplane and nearest support vectors
- **Kernel Trick** — Maps data to higher dimensions where it becomes linearly separable

### Common Kernels

```python
# Linear:      K(x, y) = x · y
# Polynomial:  K(x, y) = (γx · y + r)^d
# RBF:         K(x, y) = exp(-γ||x - y||²)
# Sigmoid:     K(x, y) = tanh(γx · y + r)
```

### Python Implementation

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
])

svm_pipeline.fit(X_train, y_train)
print(f"Accuracy: {svm_pipeline.score(X_test, y_test):.2f}")
```

### Tuning SVMs

- **C parameter**: Controls regularization. High C = less regularization
- **gamma**: Defines influence radius. High gamma = points must be very close
- Use `GridSearchCV` to find optimal C and gamma

### Pros & Cons

- ✅ Effective in high-dimensional spaces
- ✅ Memory efficient (only stores support vectors)
- ❌ Slow on large datasets (O(n²) to O(n³))
- ❌ Doesn't provide probability estimates directly
"""
    },

    "Cross-Validation": {
        "category": "Model Evaluation",
        "tags": ["validation", "model-selection", "overfitting"],
        "content": """
## Cross-Validation

Cross-validation is a technique for assessing how well a model will generalize to an independent dataset. It helps detect **overfitting** and gives a more reliable performance estimate than a single train-test split.

### K-Fold Cross-Validation

1. Split data into K equal-sized folds
2. For each fold: use it as test set, train on remaining K-1 folds
3. Average the K performance scores

### Types of Cross-Validation

```python
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    TimeSeriesSplit,
    cross_val_score
)

# Standard K-Fold
scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True))

# Stratified K-Fold (preserves class distribution)
scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5))

# Time Series Split (respects temporal order)
scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))
```

### When to Use Which

| Method | Use Case |
|--------|----------|
| K-Fold (K=5 or 10) | General purpose, enough data |
| Stratified K-Fold | Imbalanced classification |
| Leave-One-Out | Very small datasets |
| Time Series Split | Temporal/sequential data |
| Repeated K-Fold | When you want lower variance estimates |

### Common Mistake

Never use cross-validation on **already preprocessed** data. Preprocessing (scaling, feature selection) must happen **inside** each fold to prevent data leakage.

```python
from sklearn.pipeline import Pipeline

# CORRECT: preprocessing inside CV
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
scores = cross_val_score(pipe, X, y, cv=5)
```
"""
    },

    "Feature Engineering": {
        "category": "Data Preprocessing",
        "tags": ["features", "preprocessing", "transformations"],
        "content": """
## Feature Engineering

Feature engineering is the process of creating new features or transforming existing ones to improve model performance. It's often the **most impactful** step in a data science pipeline.

### Common Techniques

#### 1. Numerical Transformations
```python
import numpy as np

# Log transform (for right-skewed data)
df['log_income'] = np.log1p(df['income'])

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100],
                          labels=['child', 'young', 'middle', 'senior', 'elderly'])

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)
```

#### 2. Categorical Encoding
```python
# One-hot encoding
pd.get_dummies(df, columns=['color'], drop_first=True)

# Target encoding (for high-cardinality)
means = df.groupby('city')['target'].mean()
df['city_encoded'] = df['city'].map(means)

# Frequency encoding
freq = df['category'].value_counts(normalize=True)
df['category_freq'] = df['category'].map(freq)
```

#### 3. Date/Time Features
```python
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

#### 4. Text Features
```python
df['text_length'] = df['review'].str.len()
df['word_count'] = df['review'].str.split().str.len()
df['has_exclamation'] = df['review'].str.contains('!').astype(int)
```

### Golden Rules

- Always create features **before** splitting train/test
- Use domain knowledge — the best features come from understanding the problem
- Remove features with near-zero variance
- Check for feature leakage (features that wouldn't be available at prediction time)
"""
    },

    "Handling Missing Data": {
        "category": "Data Preprocessing",
        "tags": ["missing-values", "imputation", "data-cleaning"],
        "content": """
## Handling Missing Data

Missing data is a reality in almost every dataset. How you handle it can significantly impact model performance.

### Types of Missing Data

1. **MCAR** (Missing Completely At Random) — Missingness is random
2. **MAR** (Missing At Random) — Missingness depends on observed data
3. **MNAR** (Missing Not At Random) — Missingness depends on unobserved data

### Detection

```python
import pandas as pd
import missingno as msno

# Summary
print(df.isnull().sum())
print(df.isnull().mean() * 100)  # percentage

# Visualization
msno.matrix(df)
msno.heatmap(df)  # correlation of missingness
```

### Imputation Strategies

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputation
imp_mean = SimpleImputer(strategy='mean')
imp_median = SimpleImputer(strategy='median')
imp_mode = SimpleImputer(strategy='most_frequent')

# KNN imputation (considers feature relationships)
imp_knn = KNNImputer(n_neighbors=5)
X_imputed = imp_knn.fit_transform(X)

# Iterative imputation (MICE)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp_mice = IterativeImputer(random_state=42)
X_imputed = imp_mice.fit_transform(X)
```

### Best Practices

| Scenario | Recommended Approach |
|----------|---------------------|
| < 5% missing | Drop rows or simple imputation |
| 5-30% missing | KNN or MICE imputation |
| > 30% missing | Consider dropping the feature |
| Categorical | Mode imputation or add "Unknown" category |
| Time series | Forward/backward fill |

### Adding a Missing Indicator

```python
# Sometimes missingness itself is informative!
df['salary_missing'] = df['salary'].isnull().astype(int)
df['salary'] = df['salary'].fillna(df['salary'].median())
```
"""
    },

    "Pandas Essentials": {
        "category": "Python Tools",
        "tags": ["pandas", "dataframes", "data-manipulation"],
        "content": """
## Pandas Essentials for Data Science

Pandas is the backbone of data manipulation in Python. Here are the most important operations you should know.

### Reading Data

```python
import pandas as pd

df = pd.read_csv('data.csv', parse_dates=['date_col'])
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_json('data.json')
df = pd.read_sql('SELECT * FROM table', connection)
```

### Essential Operations

```python
# Quick look
df.head(), df.info(), df.describe()
df.shape, df.dtypes, df.columns

# Selection
df['col']                          # single column
df[['col1', 'col2']]             # multiple columns
df.loc[0:5, 'col1':'col3']       # label-based
df.iloc[0:5, 0:3]                # integer-based
df.query("age > 30 and city == 'NYC'")  # query syntax

# Filtering
mask = (df['age'] > 30) & (df['salary'] > 50000)
filtered = df[mask]

# Sorting
df.sort_values(['col1', 'col2'], ascending=[True, False])
```

### GroupBy — The Power Tool

```python
# Basic aggregation
df.groupby('category')['sales'].agg(['mean', 'sum', 'count'])

# Multiple aggregations
df.groupby('dept').agg(
    avg_salary=('salary', 'mean'),
    headcount=('employee_id', 'nunique'),
    total_sales=('sales', 'sum')
)

# Transform (returns same-sized output)
df['salary_zscore'] = df.groupby('dept')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

### Merging & Joining

```python
# Merge (like SQL JOIN)
result = pd.merge(df1, df2, on='key', how='left')
result = pd.merge(df1, df2, left_on='id', right_on='user_id', how='inner')

# Concat (stacking)
combined = pd.concat([df1, df2], axis=0, ignore_index=True)
```

### Performance Tips

- Use `category` dtype for low-cardinality strings
- Use `read_csv(usecols=[...])` to load only needed columns
- Use `.values` or `.to_numpy()` when feeding to sklearn
"""
    },

    "Neural Networks Basics": {
        "category": "Deep Learning",
        "tags": ["neural-networks", "deep-learning", "backpropagation"],
        "content": """
## Neural Networks — The Basics

Neural networks are computing systems inspired by biological neural networks. They learn by adjusting connection weights through a process called **backpropagation**.

### Architecture

```
Input Layer → Hidden Layer(s) → Output Layer
     x₁  ─┐
     x₂  ──┼── h₁ ──┐
     x₃  ──┤        ├── ŷ
            └── h₂ ──┘
```

### Key Components

1. **Neurons** — Compute weighted sum + bias, then apply activation
2. **Weights** — Learnable parameters connecting neurons
3. **Bias** — Offset term for each neuron
4. **Activation Functions** — Introduce non-linearity

### Common Activation Functions

```python
# ReLU (most common for hidden layers)
f(x) = max(0, x)

# Sigmoid (binary classification output)
f(x) = 1 / (1 + e^(-x))

# Softmax (multi-class classification output)
f(xᵢ) = e^(xᵢ) / Σe^(xⱼ)

# Tanh (centered version of sigmoid)
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

### PyTorch Example

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

model = SimpleNN(input_dim=10, hidden_dim=64, output_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### Training Loop Essentials

1. Forward pass → compute predictions
2. Calculate loss
3. Backward pass → compute gradients
4. Update weights using optimizer
5. Repeat for all epochs
"""
    },

    "Exploratory Data Analysis (EDA)": {
        "category": "Data Analysis",
        "tags": ["visualization", "statistics", "data-exploration"],
        "content": """
## Exploratory Data Analysis (EDA)

EDA is the process of examining and visualizing data to discover patterns, spot anomalies, and check assumptions before modeling.

### Step 1: Understand the Data

```python
import pandas as pd

df.shape                    # rows, columns
df.dtypes                   # data types
df.describe()               # summary stats
df.nunique()                # unique values per column
df.isnull().sum()           # missing values
df.duplicated().sum()       # duplicate rows
```

### Step 2: Univariate Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution of numerical features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    ax = axes[i // 3, i % 3]
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f'{col} — skew: {df[col].skew():.2f}')

# Categorical features
for col in categorical_cols:
    print(df[col].value_counts(normalize=True).head(10))
```

### Step 3: Bivariate Analysis

```python
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)

# Target vs features
for col in numerical_cols:
    sns.boxplot(x='target', y=col, data=df)
    plt.show()
```

### Step 4: Outlier Detection

```python
# IQR method
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['col'] < Q1 - 1.5*IQR) | (df['col'] > Q3 + 1.5*IQR)]

# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df[numerical_cols]))
outliers = df[(z_scores > 3).any(axis=1)]
```

### EDA Checklist

- [ ] Check data types and fix any mismatches
- [ ] Handle missing values
- [ ] Check for duplicates
- [ ] Examine distributions (skewness, kurtosis)
- [ ] Look at correlations between features
- [ ] Check target variable distribution (balanced?)
- [ ] Identify outliers
- [ ] Create meaningful visualizations
- [ ] Document findings and hypotheses
"""
    },

    "Logistic Regression": {
        "category": "Machine Learning",
        "tags": ["classification", "supervised-learning", "probability"],
        "content": """
## Logistic Regression

Despite its name, Logistic Regression is a **classification** algorithm. It predicts the probability that an instance belongs to a particular class using the **logistic (sigmoid) function**.

### The Sigmoid Function

```
P(y=1|x) = 1 / (1 + e^(-(β₀ + β₁x₁ + ... + βₙxₙ)))
```

The output is always between 0 and 1, making it perfect for probability estimation.

### Decision Boundary

By default, the threshold is 0.5:
- P ≥ 0.5 → Class 1
- P < 0.5 → Class 0

You can adjust this threshold based on your business need (e.g., lower threshold for medical diagnosis to reduce false negatives).

### Python Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
```

### Interpreting Coefficients

```python
import numpy as np

# Odds ratios
odds_ratios = np.exp(model.coef_[0])
for feature, odds in zip(feature_names, odds_ratios):
    print(f"{feature}: {odds:.3f}")
    # odds > 1: increases probability of class 1
    # odds < 1: decreases probability of class 1
```

### Regularization

- **L1 (Lasso)**: Performs feature selection (drives coefficients to zero)
- **L2 (Ridge)**: Shrinks coefficients but keeps all features
- **ElasticNet**: Combines L1 and L2
"""
    },

    "Clustering with K-Means": {
        "category": "Unsupervised Learning",
        "tags": ["clustering", "unsupervised", "k-means"],
        "content": """
## K-Means Clustering

K-Means is the most popular unsupervised learning algorithm for partitioning data into K distinct clusters based on feature similarity.

### Algorithm Steps

1. Randomly initialize K centroids
2. Assign each point to the nearest centroid
3. Recalculate centroids as the mean of assigned points
4. Repeat steps 2-3 until convergence

### Finding Optimal K

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Elbow Method
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Silhouette Score
from sklearn.metrics import silhouette_score
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k}: Silhouette Score = {score:.3f}")
```

### Important Preprocessing

```python
from sklearn.preprocessing import StandardScaler

# ALWAYS scale features before K-Means!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Limitations & Alternatives

- K-Means assumes **spherical clusters** of similar size
- Sensitive to initialization → use `n_init=10` or `init='k-means++'`
- For non-spherical clusters: try **DBSCAN** or **Gaussian Mixture Models**
- For hierarchical structure: try **Agglomerative Clustering**
"""
    },

    "Dimensionality Reduction with PCA": {
        "category": "Unsupervised Learning",
        "tags": ["PCA", "dimensionality-reduction", "feature-extraction"],
        "content": """
## Principal Component Analysis (PCA)

PCA transforms a high-dimensional dataset into a lower-dimensional one while preserving the maximum amount of **variance** (information).

### How PCA Works

1. Standardize the features
2. Compute the covariance matrix
3. Calculate eigenvectors and eigenvalues
4. Sort eigenvectors by decreasing eigenvalues
5. Select top K eigenvectors → these are your principal components
6. Project data onto the new K-dimensional space

### Python Implementation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 1: Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=0.95)  # retain 95% variance
X_pca = pca.fit_transform(X_scaled)

print(f"Original dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {X_pca.shape[1]}")
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

### Visualization

```python
# Scree plot — how many components to keep?
import matplotlib.pyplot as plt

pca_full = PCA().fit(X_scaled)
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

plt.plot(cumulative_var)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.show()
```

### When to Use PCA

- High-dimensional data (reduce curse of dimensionality)
- Data visualization (project to 2D/3D)
- Noise reduction
- As preprocessing before other ML algorithms

### Important Notes

- PCA is a **linear** method — for non-linear data, try t-SNE or UMAP
- Components are **not interpretable** — they're linear combinations of original features
- Always **standardize** data before PCA
"""
    },

    "Natural Language Processing (NLP) Basics": {
        "category": "NLP",
        "tags": ["text-processing", "NLP", "tokenization"],
        "content": """
## NLP Basics

Natural Language Processing enables machines to understand, interpret, and generate human language.

### Text Preprocessing Pipeline

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)
```

### Text Vectorization

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Bag of Words
bow = CountVectorizer(max_features=5000)
X_bow = bow.fit_transform(corpus)

# TF-IDF (preferred)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(corpus)
```

### Modern Approach: Word Embeddings

```python
# Using pre-trained embeddings
from gensim.models import Word2Vec

# Train custom embeddings
model = Word2Vec(sentences, vector_size=100, window=5, min_count=2)
vector = model.wv['machine']  # get word vector
similar = model.wv.most_similar('learning', topn=5)
```

### Common NLP Tasks

| Task | Description | Common Approach |
|------|-------------|-----------------|
| Sentiment Analysis | Positive/negative classification | BERT, fine-tuned transformers |
| NER | Extract entities (names, dates) | spaCy, Hugging Face |
| Text Classification | Categorize documents | TF-IDF + Logistic Regression |
| Summarization | Condense long text | T5, BART |
| Translation | Convert between languages | Seq2seq, Transformers |
"""
    },

    "Time Series Forecasting": {
        "category": "Time Series",
        "tags": ["forecasting", "ARIMA", "temporal-data"],
        "content": """
## Time Series Forecasting

Time series analysis involves techniques for analyzing time-ordered data points to extract meaningful statistics and predict future values.

### Key Components

- **Trend** — Long-term increase or decrease
- **Seasonality** — Regular periodic patterns
- **Residual** — Random noise after removing trend and seasonality

### Stationarity Check

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
# p < 0.05 → series is stationary
```

### ARIMA Model

```python
from statsmodels.tsa.arima.model import ARIMA

# ARIMA(p, d, q)
# p = autoregressive terms
# d = differencing order
# q = moving average terms

model = ARIMA(train_data, order=(2, 1, 2))
fitted = model.fit()
forecast = fitted.forecast(steps=30)

print(fitted.summary())
```

### Prophet (by Meta)

```python
from prophet import Prophet

df = pd.DataFrame({'ds': dates, 'y': values})

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
model.fit(df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
model.plot(forecast)
model.plot_components(forecast)
```

### Evaluation Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
```
"""
    },

    "SQL for Data Science": {
        "category": "Data Engineering",
        "tags": ["SQL", "databases", "querying"],
        "content": """
## SQL for Data Science

SQL is an essential skill for any data scientist. Here are the most important patterns you'll use daily.

### Essential Query Patterns

```sql
-- Window Functions (the power tool)
SELECT 
    employee_id,
    department,
    salary,
    AVG(salary) OVER (PARTITION BY department) AS dept_avg,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS salary_rank,
    salary - LAG(salary) OVER (ORDER BY hire_date) AS salary_change
FROM employees;

-- Common Table Expressions (CTEs)
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS total_sales
    FROM orders
    GROUP BY 1
),
growth AS (
    SELECT 
        month,
        total_sales,
        LAG(total_sales) OVER (ORDER BY month) AS prev_month,
        ROUND((total_sales - LAG(total_sales) OVER (ORDER BY month)) 
              / LAG(total_sales) OVER (ORDER BY month) * 100, 2) AS growth_pct
    FROM monthly_sales
)
SELECT * FROM growth;
```

### Useful Analytical Queries

```sql
-- Cohort Analysis
SELECT 
    first_purchase_month,
    DATEDIFF(month, first_purchase_month, order_month) AS months_since,
    COUNT(DISTINCT user_id) AS retained_users
FROM (
    SELECT 
        user_id,
        DATE_TRUNC('month', order_date) AS order_month,
        MIN(DATE_TRUNC('month', order_date)) OVER (PARTITION BY user_id) AS first_purchase_month
    FROM orders
) t
GROUP BY 1, 2;

-- Running Total
SELECT 
    date,
    daily_revenue,
    SUM(daily_revenue) OVER (ORDER BY date) AS cumulative_revenue
FROM daily_metrics;

-- Percentile / Distribution
SELECT 
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY salary) AS p95_salary
FROM employees;
```

### Performance Tips

- Always filter early with `WHERE` before `JOIN`
- Use `EXISTS` instead of `IN` for subqueries
- Create indexes on frequently filtered/joined columns
- Use `EXPLAIN ANALYZE` to understand query plans
"""
    },

    "A/B Testing": {
        "category": "Statistics",
        "tags": ["hypothesis-testing", "experiments", "statistics"],
        "content": """
## A/B Testing for Data Scientists

A/B testing is the gold standard for measuring the causal impact of changes. Understanding the statistics behind it is crucial.

### The Framework

1. **Hypothesis**: "Changing X will improve metric Y"
2. **Control (A)**: Current version
3. **Treatment (B)**: New version
4. **Metric**: What you're measuring (conversion rate, revenue, etc.)

### Sample Size Calculation

```python
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

# Parameters
baseline_rate = 0.10  # current conversion rate
minimum_effect = 0.02  # want to detect 2% absolute increase
alpha = 0.05           # significance level
power = 0.80           # probability of detecting real effect

analysis = NormalIndPower()
sample_size = analysis.solve_power(
    effect_size=(minimum_effect / baseline_rate) / 
                 np.sqrt(baseline_rate * (1 - baseline_rate)),
    alpha=alpha,
    power=power,
    alternative='two-sided'
)
print(f"Required sample size per group: {int(np.ceil(sample_size))}")
```

### Running the Test

```python
from scipy import stats

# For proportions (conversion rates)
control_conversions = 500
control_total = 5000
treatment_conversions = 550
treatment_total = 5000

count = np.array([control_conversions, treatment_conversions])
nobs = np.array([control_total, treatment_total])

z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Confidence interval for the difference
from statsmodels.stats.proportion import confint_proportions_2indep
ci_low, ci_high = confint_proportions_2indep(
    count[1], nobs[1], count[0], nobs[0], method='wald'
)
print(f"95% CI for difference: [{ci_low:.4f}, {ci_high:.4f}]")
```

### Common Mistakes

- **Peeking**: Checking results before reaching required sample size
- **Multiple testing**: Running many tests inflates false positive rate
- **Simpson's paradox**: Aggregate results can mislead — always segment
- **Novelty effect**: Short-term behavior change that fades
"""
    },

    "Bias-Variance Tradeoff": {
        "category": "Machine Learning Theory",
        "tags": ["theory", "overfitting", "underfitting"],
        "content": """
## The Bias-Variance Tradeoff

Understanding this tradeoff is fundamental to building good ML models.

### The Decomposition

```
Total Error = Bias² + Variance + Irreducible Error
```

- **Bias**: Error from wrong assumptions (underfitting)
- **Variance**: Error from sensitivity to training data (overfitting)
- **Irreducible error**: Noise in the data itself

### Visual Intuition

| | Low Variance | High Variance |
|---|---|---|
| **Low Bias** | 🎯 Perfect | Overfitting |
| **High Bias** | Underfitting | Worst case |

### How to Diagnose

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

# Plot
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
```

### Interpretation

- **High bias** (underfitting): Both training and validation scores are low
- **High variance** (overfitting): Training score is high, validation score is much lower
- **Good fit**: Both scores are high and close together

### Solutions

**Reduce Bias (fix underfitting):**
- Add more features
- Use a more complex model
- Reduce regularization

**Reduce Variance (fix overfitting):**
- Get more training data
- Use regularization (L1, L2, dropout)
- Simplify the model
- Use ensemble methods (bagging)
"""
    },

    "Data Visualization Best Practices": {
        "category": "Data Visualization",
        "tags": ["visualization", "matplotlib", "seaborn"],
        "content": """
## Data Visualization Best Practices

Good visualizations communicate insights clearly. Here's how to choose and create effective charts.

### Choosing the Right Chart

| Data Relationship | Best Chart Type |
|-------------------|----------------|
| Distribution | Histogram, KDE, Box plot |
| Comparison | Bar chart, Grouped bar |
| Trend over time | Line chart |
| Correlation | Scatter plot |
| Composition | Pie chart, Stacked bar |
| Part-to-whole | Treemap, Waterfall |

### Matplotlib + Seaborn Templates

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Publication-quality figure
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x='category', y='value', ax=ax)
ax.set_title('Sales by Category', fontsize=14, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('Revenue ($)')

# Add value labels
for p in ax.patches:
    ax.annotate(f'${p.get_height():,.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('chart.png', dpi=150, bbox_inches='tight')
```

### Plotly for Interactive Charts

```python
import plotly.express as px

fig = px.scatter(df, x='feature1', y='feature2', 
                 color='category', size='value',
                 hover_data=['name'],
                 title='Feature Relationships')
fig.update_layout(template='plotly_white')
fig.show()
```

### Design Principles

1. **Remove chartjunk** — no unnecessary gridlines, borders, or 3D effects
2. **Use color purposefully** — highlight key data, not everything
3. **Label directly** — put labels on the data, not in a legend
4. **Start y-axis at zero** for bar charts (but not always for line charts)
5. **Tell a story** — every chart should answer a specific question
"""
    },
}

# ─────────────────────────────────────────────────────────────
# Article Generator
# ─────────────────────────────────────────────────────────────

def generate_article():
    """Pick a random topic and generate a markdown article."""
    topic_name = random.choice(list(TOPICS.keys()))
    topic = TOPICS[topic_name]
    
    today = datetime.date.today()
    date_str = today.strftime("%Y-%m-%d")
    day_name = today.strftime("%A")
    
    article = f"""# 📘 {topic_name}

> **Category:** {topic["category"]}  
> **Date:** {date_str} ({day_name})  
> **Tags:** {", ".join(f"`{tag}`" for tag in topic["tags"])}

---
{topic["content"]}
---

*This article was auto-generated as part of the StudyPush daily data science learning challenge.*  
*Keep pushing, keep learning! 🚀*
"""
    return topic_name, topic["category"], date_str, article


def main():
    topic_name, category, date_str, article = generate_article()
    
    # Create directory structure
    category_dir = os.path.join("articles", category.lower().replace(" ", "-"))
    os.makedirs(category_dir, exist_ok=True)
    
    # Create filename
    safe_name = topic_name.lower().replace(" ", "-").replace("(", "").replace(")", "")
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '-')
    filename = f"{date_str}_{safe_name}.md"
    filepath = os.path.join(category_dir, filename)
    
    # Write the article
    with open(filepath, "w") as f:
        f.write(article)
    
    print(f"✅ Generated: {filepath}")
    print(f"📚 Topic: {topic_name}")
    print(f"📂 Category: {category}")
    
    # Update the README with latest article
    update_readme(topic_name, category, date_str, filepath)
    
    return filepath


def update_readme(topic_name, category, date_str, filepath):
    """Update README.md with the latest article entry."""
    readme_path = "README.md"
    
    new_entry = f"| {date_str} | [{topic_name}]({filepath}) | {category} |"
    
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            content = f.read()
        
        # Insert new entry after the table header
        table_header = "| Date | Topic | Category |"
        separator = "|------|-------|----------|"
        
        if table_header in content:
            insert_pos = content.index(separator) + len(separator)
            content = content[:insert_pos] + "\n" + new_entry + content[insert_pos:]
        else:
            content += f"\n\n## 📊 Article Log\n\n{table_header}\n{separator}\n{new_entry}\n"
    else:
        content = f"""# 📚 StudyPush — Daily Data Science Articles

> Automated daily commits with random data science articles to keep the learning streak alive!

## 🚀 How It Works

A GitHub Actions workflow runs daily at **9:00 AM UTC** and:
1. Picks a random data science topic
2. Generates a detailed markdown article with code examples
3. Commits and pushes it to this repository

## 📂 Topics Covered

- Machine Learning (Linear Regression, Decision Trees, SVM, etc.)
- Deep Learning (Neural Networks, CNNs, RNNs)
- Data Preprocessing (Feature Engineering, Missing Data)
- Statistics (A/B Testing, Hypothesis Testing)
- Python Tools (Pandas, NumPy, Scikit-learn)
- NLP, Time Series, SQL, and more!

## 📊 Article Log

{table_header}
{separator}
{new_entry}
"""
    
    with open(readme_path, "w") as f:
        f.write(content)
    
    print(f"📝 README updated")


if __name__ == "__main__":
    main()
