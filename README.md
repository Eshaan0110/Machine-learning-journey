# Machine Learning Journey

A hands-on collection of ML algorithm implementations built while learning core concepts from scratch. Each notebook is self-contained and focused on one algorithm, with both standalone experiments and structured lab versions.

## Algorithms Covered

| Topic | Files |
|---|---|
| Simple Linear Regression | `SLR.ipynb`, `Lab1/SLR.ipynb` |
| Multiple & Polynomial Regression | `Multiple&Polynomial.ipynb`, `Lab1/Multiple&Polynomial.ipynb` |
| Decision Trees (ID3) | `ID3.ipynb`, `Lab2/ID3.ipynb` |
| Decision Trees (C4.5) | `c4.5.ipynb`, `Lab2/c4_5.ipynb` |
| Decision Trees (CART) | `cartUpdated.ipynb`, `Lab2/cart.ipynb` |
| K-Nearest Neighbours | `KNN and SVM.ipynb`, `Lab4/KNN.ipynb` |
| Support Vector Machines | `KNN and SVM.ipynb` |
| Naive Bayes | `Naive Bayes.ipynb` |
| K-Means Clustering | `Clustering.ipynb`, `KMeans.ipynb` |
| Neural Networks | `Neural Network.ipynb` |
| Q-Learning | `Q learning.ipynb` |
| Time Series (Prophet) | `prophet-learning/` |

Root notebooks are quick experiments. The `Lab1/`, `Lab2/`, and `Lab4/` folders contain more structured lab versions of the same algorithms.

## Project: Prophet Time Series Forecasting

`prophet-learning/` is a standalone project that forecasts website traffic using Meta's Prophet library. It produces two plots:

- `results/forecast.png` — predicted values with uncertainty intervals
- `results/components.png` — trend and seasonality breakdown

See [`prophet-learning/Readme.MD`](prophet-learning/Readme.MD) for full details.

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

## Running the Prophet Project

```bash
cd prophet-learning/src
python forecast.py
```
