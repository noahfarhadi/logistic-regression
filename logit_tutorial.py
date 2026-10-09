"""
Logistic regression in Python: from scratch, with statsmodels, and with scikit-learn.

Run in Spyder cell by cell (Ctrl+Enter on each "# %%" block) or as a whole file (F5).
Every setting lives in CONFIG. With csv_path = None the script simulates data with known
coefficients, so you can check that each method recovers them.
"""

# %% 0. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, confusion_matrix)

# %% 1. Settings
CONFIG = {
    "csv_path": None,            # path to your own CSV file; None = simulate data
    "target": "y",               # name of the 0/1 dependent variable
    "features": None,            # list of independent variables; None = all other numeric columns
    "n_obs": 1000,               # simulated data: number of observations
    "true_coefs": [-0.5, 1.2, -0.8, 0.5],   # simulated data: intercept first, then one value per variable
    "test_share": 0.20,          # share of rows kept for testing (the 80-20 split)
    "threshold": 0.50,           # predicted probability at or above this value is classed as 1
    "standardize": False,        # True = rescale each variable to mean 0 and sd 1 (learned on training rows)
    "robust_se": False,          # True = heteroskedasticity-robust (HC0) standard errors in statsmodels
    "newton_max_iter": 50,       # from-scratch fit: maximum number of Newton steps
    "newton_tol": 1e-10,         # from-scratch fit: stop when the largest step is below this
    "seed": 42,
}


# %% 2. The two formulas behind the model
# Linear score:   z = b0 + b1*x1 + ... + bk*xk
# Probability:    p = 1 / (1 + exp(-z))            (the sigmoid)
# Log-likelihood: sum( y*log(p) + (1-y)*log(1-p) ) (the quantity the estimation maximizes)
def sigmoid(z):
    # Step 1: clip the score so exp() cannot overflow for extreme values.
    z = np.clip(z, -500, 500)
    # Step 2: map any real number to a probability between 0 and 1.
    return 1.0 / (1.0 + np.exp(-z))


# %% 3. Data
def make_synthetic(cfg):
    # Step 1: draw independent standard-normal variables.
    rng = np.random.default_rng(cfg["seed"])
    k = len(cfg["true_coefs"]) - 1
    X = rng.normal(size=(cfg["n_obs"], k))
    # Step 2: build the true linear score and turn it into probabilities.
    p = sigmoid(cfg["true_coefs"][0] + X @ np.asarray(cfg["true_coefs"][1:]))
    # Step 3: draw each outcome as a 0/1 coin flip with that probability.
    y = rng.binomial(1, p)
    cols = [f"x{j + 1}" for j in range(k)]
    df = pd.DataFrame(X, columns=cols)
    df[cfg["target"]] = y
    truth = pd.Series(cfg["true_coefs"], index=["const"] + cols)
    return df, truth


def load_data(cfg):
    # Step 1: read the user's file, or simulate when no file is given.
    if cfg["csv_path"] is None:
        df, truth = make_synthetic(cfg)
    else:
        df, truth = pd.read_csv(cfg["csv_path"]), None
    # Step 2: take the listed variables, or every other numeric column.
    features = cfg["features"] or [c for c in df.select_dtypes("number").columns if c != cfg["target"]]
    # Step 3: drop rows with missing values in the variables used.
    df = df[features + [cfg["target"]]].dropna()
    return df[features], df[cfg["target"]].astype(int), truth


X, y, truth = load_data(CONFIG)
print(f"Rows: {len(X)}   Variables: {list(X.columns)}   Share of y = 1: {y.mean():.3f}")

# %% 4. Training and test sample
# Step 1: split once, keeping the share of 1s the same in both parts (stratify).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=CONFIG["test_share"], stratify=y, random_state=CONFIG["seed"])

# Step 2: optional standardizing. Mean and sd come from the training rows only,
# so no information from the test rows enters the model.
if CONFIG["standardize"]:
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X.columns)


# %% 5. Estimation from scratch (Newton's method, also called iteratively reweighted least squares)
# Gradient of the log-likelihood:  X'(y - p)
# Hessian (negative):              X'WX   with W = diag(p*(1-p))
# Update:                          b_new = b + (X'WX)^-1 X'(y - p)
def fit_logit_newton(X_df, y_s, max_iter, tol):
    # Step 1: add a column of ones for the intercept and start from all-zero coefficients.
    A = np.column_stack([np.ones(len(X_df)), X_df.to_numpy(float)])
    t = y_s.to_numpy(float)
    b = np.zeros(A.shape[1])
    for i in range(1, max_iter + 1):
        # Step 2: current probabilities, weights, gradient and information matrix.
        p = sigmoid(A @ b)
        w = p * (1 - p)
        grad = A.T @ (t - p)
        info = A.T @ (A * w[:, None])
        # Step 3: Newton step; stop when the coefficients no longer move.
        step = np.linalg.solve(info, grad)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    # Step 4: classical standard errors come from the inverse of the information matrix at the solution.
    p = sigmoid(A @ b)
    info = A.T @ (A * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(info)))
    names = ["const"] + list(X_df.columns)
    return pd.Series(b, index=names), pd.Series(se, index=names), i


b_scratch, se_scratch, n_steps = fit_logit_newton(
    X_train, y_train, CONFIG["newton_max_iter"], CONFIG["newton_tol"])
print(f"\nFrom scratch: converged in {n_steps} Newton steps")

# %% 6. Estimation with statsmodels (full inference output)
# Step 1: add the intercept, fit by maximum likelihood, choose classical or robust standard errors.
cov_type = "HC0" if CONFIG["robust_se"] else "nonrobust"
sm_fit = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=0, cov_type=cov_type)
# Step 2: the summary shows coefficients, standard errors, z, p-values and the confidence interval.
print("\n", sm_fit.summary())
# Step 3: odds ratios are exp(coefficient): the factor by which the odds of y = 1 change per unit of x.
print("\nOdds ratios:\n", np.exp(sm_fit.params).round(4))

# %% 7. Estimation with scikit-learn (the tool for prediction work)
# C = infinity switches the penalty off, which gives the same estimator as above.
# A finite C (for example 1.0) adds an L2 penalty that shrinks the coefficients.
skl = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10000).fit(X_train, y_train)
b_skl = pd.Series(np.r_[skl.intercept_, skl.coef_.ravel()], index=["const"] + list(X.columns))

# %% 8. Compare the three sets of coefficients
table = pd.DataFrame({"scratch": b_scratch, "statsmodels": sm_fit.params, "sklearn": b_skl})
if truth is not None and not CONFIG["standardize"]:
    table["true value"] = truth
print("\nCoefficients on the training sample:\n", table.round(4))
print("Largest difference between the three methods:",
      float((table[["scratch", "statsmodels", "sklearn"]].max(axis=1)
             - table[["scratch", "statsmodels", "sklearn"]].min(axis=1)).max()))


# %% 9. Judge the model on the test sample
def class_measures(y_true, prob, threshold):
    # Step 1: turn probabilities into 0/1 predictions with the chosen threshold.
    pred = (prob >= threshold).astype(int)
    # Step 2: count-based measures use the predictions; AUC and log loss use the probabilities.
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "AUC": roc_auc_score(y_true, prob),
        "log loss": log_loss(y_true, prob),
    }, confusion_matrix(y_true, pred, labels=[0, 1])


prob_test = skl.predict_proba(X_test)[:, 1]
measures, cm = class_measures(y_test, prob_test, CONFIG["threshold"])
print("\nTest sample measures:\n", pd.Series(measures).round(4))
print("\nConfusion matrix (rows = actual 0/1, columns = predicted 0/1):\n", cm)

# %% 10. Plot the sigmoid
# Step 1: linear score of every test row, and a smooth grid of scores for the curve.
z_test = skl.decision_function(X_test)
grid = np.linspace(z_test.min() - 1, z_test.max() + 1, 300)
# Step 2: the curve is the sigmoid; the points are the actual 0/1 outcomes of the test rows.
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(grid, sigmoid(grid), color="tab:blue", label="sigmoid")
ax.scatter(z_test, y_test, s=14, alpha=0.4, color="tab:orange", label="test observations")
ax.axhline(CONFIG["threshold"], color="grey", linestyle="--", linewidth=1, label="threshold")
ax.set_xlabel("linear score z")
ax.set_ylabel("probability of y = 1")
ax.legend(loc="center right")
fig.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Version log
# ---------------------------------------------------------------------------
# v1.0 (2026-10-09)  New file. Added: CONFIG; sigmoid; make_synthetic and load_data; stratified train/test split
#   with optional standardizing learned on training rows; fit_logit_newton (coefficients and classical standard
#   errors); statsmodels Logit with optional HC0 errors and odds ratios; scikit-learn LogisticRegression without
#   penalty; coefficient comparison table; class_measures and confusion matrix; sigmoid plot.
