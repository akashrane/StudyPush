# 📚 StudyPush — Daily Data Science Articles

> Automated daily commits with random data science articles to keep the learning streak alive!

## 🚀 How It Works

A GitHub Actions workflow runs daily at **9:00 AM UTC** and:
1. Picks a random data science topic from a curated pool of 20+ topics
2. Generates a detailed markdown article with code examples
3. Commits and pushes it to this repository automatically

## 📂 Topics Covered

- **Machine Learning** — Linear Regression, Decision Trees, Random Forest, XGBoost, SVM, KNN, Logistic Regression
- **Deep Learning** — Neural Networks, Backpropagation, Activation Functions
- **Unsupervised Learning** — K-Means Clustering, PCA, Dimensionality Reduction
- **Data Preprocessing** — Feature Engineering, Handling Missing Data
- **Statistics** — A/B Testing, Bias-Variance Tradeoff, Cross-Validation
- **Python Tools** — Pandas, NumPy, Scikit-learn
- **NLP** — Text Preprocessing, TF-IDF, Word Embeddings
- **Time Series** — ARIMA, Prophet, Forecasting
- **Data Engineering** — SQL Window Functions, CTEs, Query Optimization
- **Data Visualization** — Matplotlib, Seaborn, Plotly Best Practices
- **Data Analysis** — EDA, Outlier Detection, Distribution Analysis

## 🛠️ Setup Instructions

1. Fork or clone this repo
2. Enable GitHub Actions in your repo settings
3. The workflow will run automatically every day at 9 AM UTC
4. To trigger manually: go to **Actions** tab → select the workflow → click **Run workflow**

## ⚙️ Customization

- **Change schedule**: Edit the `cron` expression in `.github/workflows/daily-push.yml`
- **Add topics**: Add new entries to the `TOPICS` dictionary in `generate_article.py`
- **Change timezone**: Modify the cron time (e.g., `0 14 * * *` for 2 PM UTC = 9 AM EST)

## 📊 Article Log

| Date | Topic | Category |
|------|-------|----------|
| 2026-04-21 | [Clustering with K-Means](articles/unsupervised-learning/2026-04-21_clustering-with-k-means.md) | Unsupervised Learning |
