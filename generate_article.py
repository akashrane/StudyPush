"""
StudyPush - Daily Data Science Article Generator
Generates a random data science article/note and commits it to the repo.
"""

import random
import datetime
import os

TOPICS = {
    "Linear Regression": {
        "category": "Machine Learning",
        "tags": ["regression", "supervised-learning", "statistics"],
        "content": """
## What is Linear Regression?

Linear Regression models the relationship between a dependent variable and one or more independent variables by fitting a linear equation.

### The Math

```
y = β₀ + β₁x + ε
```

- **y** = target, **x** = feature, **β₀** = intercept, **β₁** = slope, **ε** = error

### Key Assumptions

1. **Linearity** — X and Y have a linear relationship
2. **Independence** — Observations are independent
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
- Predicting continuous outcomes (house prices, sales)
- Understanding feature importance
- As a baseline before trying complex models

### Common Pitfalls
- Multicollinearity among features
- Overfitting with too many features
- Ignoring outliers
"""
    },

    "Decision Trees": {
        "category": "Machine Learning",
        "tags": ["classification", "regression", "tree-models"],
        "content": """
## Decision Trees

Decision Trees split data recursively based on feature values to create a tree-like decision model.

### Splitting Criteria

**Gini Impurity:** `Gini(S) = 1 - Σ(pᵢ²)`

**Information Gain (Entropy):** `Entropy(S) = -Σ pᵢ log₂(pᵢ)`

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

### Key Hyperparameters
- `max_depth` — Limits tree depth
- `min_samples_split` — Minimum samples to split a node
- `min_samples_leaf` — Minimum samples in a leaf

### Pros & Cons
| Pros | Cons |
|------|------|
| Easy to interpret | Prone to overfitting |
| No feature scaling needed | Unstable with small changes |
| Handles both types | Biased toward high-cardinality |
"""
    },

    "Random Forest": {
        "category": "Machine Learning",
        "tags": ["ensemble", "classification", "regression"],
        "content": """
## Random Forest

An ensemble of decision trees using **bagging** and **feature randomness** to reduce overfitting.

### How It Works
1. Create `n` bootstrap samples
2. Build a tree per sample using random feature subsets
3. Aggregate: majority vote (classification) or average (regression)

### Python Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X, y)

for i, imp in enumerate(rf.feature_importances_):
    if imp > 0.05:
        print(f"Feature {i}: {imp:.3f}")
```

### Tuning Tips
- Start with `n_estimators=100`
- `max_features='sqrt'` for classification
- `min_samples_leaf=5` to prevent overfitting
"""
    },

    "XGBoost": {
        "category": "Machine Learning",
        "tags": ["boosting", "ensemble", "gradient-boosting"],
        "content": """
## Gradient Boosting & XGBoost

Builds trees **sequentially** — each corrects errors of the previous.

### Core Formula
```
Fₘ(x) = Fₘ₋₁(x) + η × hₘ(x)
```

### XGBoost Advantages
- L1/L2 regularization
- Built-in missing value handling
- Parallel tree building
- Smart pruning

### Python Implementation

```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=200, max_depth=5,
    learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
```

### XGBoost vs LightGBM vs CatBoost
| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Speed | Fast | Fastest | Moderate |
| Categorical | Manual | Native | Best native |
| Out-of-box | Good | Good | Often best |
"""
    },

    "K-Nearest Neighbors": {
        "category": "Machine Learning",
        "tags": ["classification", "instance-based", "non-parametric"],
        "content": """
## K-Nearest Neighbors (KNN)

A lazy learner that classifies based on the K closest training examples.

### Distance Metrics
```
Euclidean:  d = √(Σ(xᵢ - yᵢ)²)
Manhattan:  d = Σ|xᵢ - yᵢ|
```

### Choosing K
- Small K (1-3): overfitting risk
- Large K (20+): underfitting risk
- Rule of thumb: K = √n

### Python Example

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
print(f"Accuracy: {knn.score(X_test_scaled, y_test):.2f}")
```

**Important:** Always scale features before KNN!
"""
    },

    "Support Vector Machines": {
        "category": "Machine Learning",
        "tags": ["classification", "kernel-methods", "supervised-learning"],
        "content": """
## Support Vector Machines (SVM)

SVMs find the optimal hyperplane that maximizes margin between classes.

### Key Concepts
- **Support Vectors** — Closest points to the boundary
- **Margin** — Distance between hyperplane and support vectors
- **Kernel Trick** — Maps to higher dimensions for non-linear separation

### Common Kernels
- Linear: `K(x,y) = x·y`
- RBF: `K(x,y) = exp(-γ||x-y||²)`
- Polynomial: `K(x,y) = (γx·y + r)^d`

### Python Implementation

```python
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))
])
svm_pipe.fit(X_train, y_train)
print(f"Accuracy: {svm_pipe.score(X_test, y_test):.2f}")
```
"""
    },

    "Cross-Validation": {
        "category": "Model Evaluation",
        "tags": ["validation", "model-selection", "overfitting"],
        "content": """
## Cross-Validation

Assesses model generalization by splitting data into multiple train/test folds.

### Types

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score
)

# Standard K-Fold
scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True))

# Stratified (preserves class distribution)
scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5))

# Time Series (respects temporal order)
scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))
```

### Common Mistake — Data Leakage!

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

Creating and transforming features to improve model performance.

### Numerical Transforms
```python
import numpy as np
df['log_income'] = np.log1p(df['income'])
df['age_group'] = pd.cut(df['age'], bins=[0,18,35,50,65,100])
```

### Categorical Encoding
```python
pd.get_dummies(df, columns=['color'], drop_first=True)

# Target encoding
means = df.groupby('city')['target'].mean()
df['city_encoded'] = df['city'].map(means)
```

### Date/Time Features
```python
df['hour'] = df['timestamp'].dt.hour
df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5,6]).astype(int)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
```

### Golden Rules
- Create features **before** splitting
- Use domain knowledge
- Check for feature leakage
"""
    },

    "Handling Missing Data": {
        "category": "Data Preprocessing",
        "tags": ["missing-values", "imputation", "data-cleaning"],
        "content": """
## Handling Missing Data

### Types
1. **MCAR** — Missing completely at random
2. **MAR** — Depends on observed data
3. **MNAR** — Depends on unobserved data

### Imputation Strategies

```python
from sklearn.impute import SimpleImputer, KNNImputer

imp_mean = SimpleImputer(strategy='mean')
imp_knn = KNNImputer(n_neighbors=5)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp_mice = IterativeImputer(random_state=42)
```

### Best Practices
| Missing % | Approach |
|-----------|----------|
| < 5% | Drop rows or simple imputation |
| 5-30% | KNN or MICE |
| > 30% | Consider dropping the feature |

### Pro Tip: Missing Indicator
```python
df['salary_missing'] = df['salary'].isnull().astype(int)
df['salary'] = df['salary'].fillna(df['salary'].median())
```
"""
    },

    "Pandas Essentials": {
        "category": "Python Tools",
        "tags": ["pandas", "dataframes", "data-manipulation"],
        "content": """
## Pandas Essentials

### Core Operations
```python
import pandas as pd

df.head(), df.info(), df.describe()
df.query("age > 30 and city == 'NYC'")
df.sort_values(['col1', 'col2'], ascending=[True, False])
```

### GroupBy Power
```python
df.groupby('dept').agg(
    avg_salary=('salary', 'mean'),
    headcount=('employee_id', 'nunique'),
    total_sales=('sales', 'sum')
)

df['salary_zscore'] = df.groupby('dept')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

### Merging
```python
result = pd.merge(df1, df2, on='key', how='left')
combined = pd.concat([df1, df2], axis=0, ignore_index=True)
```

### Performance Tips
- Use `category` dtype for low-cardinality strings
- Use `read_csv(usecols=[...])` to load only needed columns
"""
    },

    "Neural Networks Basics": {
        "category": "Deep Learning",
        "tags": ["neural-networks", "deep-learning", "pytorch"],
        "content": """
## Neural Networks Basics

### Architecture
```
Input Layer → Hidden Layer(s) → Output Layer
```

### Activation Functions
- **ReLU**: `f(x) = max(0, x)` — most common for hidden layers
- **Sigmoid**: Binary classification output
- **Softmax**: Multi-class output

### PyTorch Example

```python
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)

model = SimpleNN(10, 64, 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
"""
    },

    "Exploratory Data Analysis": {
        "category": "Data Analysis",
        "tags": ["EDA", "visualization", "statistics"],
        "content": """
## EDA Checklist

### Step 1: Understand
```python
df.shape, df.dtypes, df.describe()
df.isnull().sum()
df.duplicated().sum()
```

### Step 2: Distributions
```python
import seaborn as sns
for col in numerical_cols:
    sns.histplot(df[col], kde=True)
```

### Step 3: Correlations
```python
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
```

### Step 4: Outliers
```python
Q1, Q3 = df['col'].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = df[(df['col'] < Q1-1.5*IQR) | (df['col'] > Q3+1.5*IQR)]
```
"""
    },

    "Logistic Regression": {
        "category": "Machine Learning",
        "tags": ["classification", "probability", "supervised-learning"],
        "content": """
## Logistic Regression

A classification algorithm using the sigmoid function to predict probabilities.

### Sigmoid
```
P(y=1|x) = 1 / (1 + e^(-(β₀ + β₁x₁ + ... + βₙxₙ)))
```

### Python Example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.3f}")
print(classification_report(y_test, model.predict(X_test)))
```

### Interpreting Coefficients
```python
import numpy as np
odds_ratios = np.exp(model.coef_[0])
# > 1: increases probability, < 1: decreases
```
"""
    },

    "K-Means Clustering": {
        "category": "Unsupervised Learning",
        "tags": ["clustering", "unsupervised", "k-means"],
        "content": """
## K-Means Clustering

### Algorithm
1. Initialize K centroids randomly
2. Assign points to nearest centroid
3. Recalculate centroids
4. Repeat until convergence

### Finding Optimal K
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    print(f"K={k}: Silhouette = {silhouette_score(X_scaled, labels):.3f}")
```

**Always scale features before K-Means!**
"""
    },

    "PCA": {
        "category": "Unsupervised Learning",
        "tags": ["dimensionality-reduction", "PCA", "feature-extraction"],
        "content": """
## Principal Component Analysis

Reduces dimensions while preserving maximum variance.

### Python Implementation
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=0.95)  # retain 95% variance
X_pca = pca.fit_transform(X_scaled)

print(f"Reduced: {X.shape[1]} → {X_pca.shape[1]} dimensions")
```

### When to Use
- High-dimensional data
- Visualization (project to 2D/3D)
- Noise reduction
- Always **standardize** first!
"""
    },

    "NLP Basics": {
        "category": "NLP",
        "tags": ["text-processing", "NLP", "tokenization"],
        "content": """
## NLP Basics

### Text Preprocessing
```python
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\\\\s]', '', text.lower())
    tokens = [t for t in text.split() if t not in stopwords.words('english')]
    return ' '.join(WordNetLemmatizer().lemmatize(t) for t in tokens)
```

### Vectorization
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(corpus)
```
"""
    },

    "Time Series Forecasting": {
        "category": "Time Series",
        "tags": ["forecasting", "ARIMA", "temporal-data"],
        "content": """
## Time Series Forecasting

### Stationarity Check
```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(series)
print(f"p-value: {result[1]:.4f}")  # < 0.05 = stationary
```

### ARIMA
```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(2, 1, 2))
forecast = model.fit().forecast(steps=30)
```

### Prophet
```python
from prophet import Prophet
model = Prophet(yearly_seasonality=True)
model.fit(pd.DataFrame({'ds': dates, 'y': values}))
forecast = model.predict(model.make_future_dataframe(periods=365))
```
"""
    },

    "SQL for Data Science": {
        "category": "Data Engineering",
        "tags": ["SQL", "databases", "querying"],
        "content": """
## SQL for Data Science

### Window Functions
```sql
SELECT employee_id, department, salary,
    AVG(salary) OVER (PARTITION BY department) AS dept_avg,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
FROM employees;
```

### CTEs
```sql
WITH monthly AS (
    SELECT DATE_TRUNC('month', order_date) AS month,
           SUM(amount) AS total
    FROM orders GROUP BY 1
)
SELECT month, total,
    LAG(total) OVER (ORDER BY month) AS prev
FROM monthly;
```

### Performance Tips
- Filter early with `WHERE` before `JOIN`
- Use `EXISTS` over `IN` for subqueries
- Index frequently filtered columns
"""
    },

    "A/B Testing": {
        "category": "Statistics",
        "tags": ["hypothesis-testing", "experiments", "statistics"],
        "content": """
## A/B Testing

### Framework
1. **Hypothesis**: Changing X improves metric Y
2. **Control (A)** vs **Treatment (B)**
3. Calculate required sample size
4. Run test → analyze results

### Sample Size
```python
from statsmodels.stats.power import NormalIndPower
sample_size = NormalIndPower().solve_power(
    effect_size=0.2, alpha=0.05, power=0.80
)
```

### Statistical Test
```python
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

z, p = proportions_ztest(
    count=np.array([500, 550]),
    nobs=np.array([5000, 5000])
)
print(f"p-value: {p:.4f}")
```

### Common Mistakes
- Peeking before reaching sample size
- Multiple testing without correction
- Ignoring novelty effects
"""
    },

    "Bias-Variance Tradeoff": {
        "category": "ML Theory",
        "tags": ["theory", "overfitting", "underfitting"],
        "content": """
## Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Error
```

### Diagnosis
```python
from sklearn.model_selection import learning_curve
sizes, train_sc, val_sc = learning_curve(model, X, y, cv=5)
```

- **High bias**: Both scores low → underfitting
- **High variance**: Train high, val low → overfitting

### Solutions
| Problem | Fix |
|---------|-----|
| Underfitting | More features, complex model, less regularization |
| Overfitting | More data, regularization, simpler model, ensembles |
"""
    },

    "Data Visualization": {
        "category": "Data Visualization",
        "tags": ["matplotlib", "seaborn", "plotly"],
        "content": """
## Data Visualization Best Practices

### Chart Selection
| Relationship | Chart |
|-------------|-------|
| Distribution | Histogram, Box plot |
| Comparison | Bar chart |
| Trend | Line chart |
| Correlation | Scatter plot |

### Seaborn Template
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df, x='category', y='value', ax=ax)
ax.set_title('Sales by Category', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('chart.png', dpi=150)
```

### Principles
1. Remove chartjunk
2. Use color purposefully
3. Label directly on data
4. Start bar chart y-axis at zero
"""
    },
}


def generate_article():
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

*Auto-generated by StudyPush — daily data science learning 🚀*
"""
    return topic_name, topic["category"], date_str, article


def update_readme(topic_name, category, date_str, filepath):
    readme_path = "README.md"
    new_entry = f"| {date_str} | [{topic_name}]({filepath}) | {category} |"

    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        table_header = "| Date | Topic | Category |"
        separator = "|------|-------|----------|"
        if table_header in content:
            pos = content.index(separator) + len(separator)
            content = content[:pos] + "\n" + new_entry + content[pos:]
        else:
            content += f"\n\n## 📊 Article Log\n\n{table_header}\n{separator}\n{new_entry}\n"
    else:
        content = f"""# 📚 StudyPush — Daily Data Science Articles

> Automated daily commits with data science articles to keep the streak alive!

## 📊 Article Log

{table_header}
{separator}
{new_entry}
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("📝 README updated")


def main():
    topic_name, category, date_str, article = generate_article()
    cat_dir = os.path.join("articles", category.lower().replace(" ", "-"))
    os.makedirs(cat_dir, exist_ok=True)
    safe = ''.join(c for c in topic_name.lower().replace(" ", "-") if c.isalnum() or c == '-')
    filepath = os.path.join(cat_dir, f"{date_str}_{safe}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(article)
    print(f"✅ Generated: {filepath}")
    print(f"📚 Topic: {topic_name}")
    update_readme(topic_name, category, date_str, filepath)
    return filepath


if __name__ == "__main__":
    main()
