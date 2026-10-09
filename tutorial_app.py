"""
Logistic regression in Python, as a web page (Streamlit).

Run locally:   streamlit run tutorial_app.py
The code shown on the page is read from the functions that the page executes, so the text
and the results always match. The Spyder version of the same material is logit_tutorial.py.
"""

import inspect

import altair as alt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------
CONFIG = {
    "page_title": "Logistic regression in Python",
    "eyebrow": "FINC 4970 Machine Learning in Finance",
    "target": "y",
    "n_obs": {"default": 1000, "min": 200, "max": 5000, "step": 100},
    "true_coefs": [-0.5, 1.2, -0.8, 0.5],      # intercept first, then one value per variable
    "test_share": {"default": 20, "min": 10, "max": 50, "step": 5},   # percent of rows kept for testing
    "threshold": {"default": 0.50, "min": 0.05, "max": 0.95, "step": 0.05},
    "seed": 42,
    "newton_max_iter": 50,
    "newton_tol": 1e-10,
    "sklearn_tol": 1e-10,
    "sklearn_max_iter": 10000,
    "sigmoid_points": 300,
    "preview_rows": 8,
    "decimals": 4,
}

st.set_page_config(page_title=CONFIG["page_title"], layout="centered")


# ---------------------------------------------------------------------------
# 2. The functions that are both executed and shown on the page
# ---------------------------------------------------------------------------
def sigmoid(z):
    # Step 1: clip the score so exp() cannot overflow for extreme values.
    z = np.clip(z, -500, 500)
    # Step 2: map any real number to a probability between 0 and 1.
    return 1.0 / (1.0 + np.exp(-z))


def make_synthetic(n_obs, true_coefs, seed, target):
    # Step 1: draw independent standard-normal variables.
    rng = np.random.default_rng(seed)
    k = len(true_coefs) - 1
    X = rng.normal(size=(n_obs, k))
    # Step 2: build the true linear score and turn it into probabilities.
    p = sigmoid(true_coefs[0] + X @ np.asarray(true_coefs[1:]))
    # Step 3: draw each outcome as a 0/1 coin flip with that probability.
    y = rng.binomial(1, p)
    cols = [f"x{j + 1}" for j in range(k)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name=target)


def split_and_scale(X, y, test_share, seed, standardize):
    # Step 1: split once; stratify keeps the share of 1s the same in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_share, stratify=y, random_state=seed)
    # Step 2: optional standardizing. Mean and sd come from the training rows only,
    # so no information from the test rows enters the model.
    if standardize:
        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X.columns)
    return X_train, X_test, y_train, y_test


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


def fit_statsmodels(X_train, y_train, robust):
    # Step 1: choose classical or heteroskedasticity-robust (HC0) standard errors.
    cov_type = "HC0" if robust else "nonrobust"
    # Step 2: add the intercept and fit by maximum likelihood.
    return sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=0, cov_type=cov_type)


def fit_sklearn(X_train, y_train, tol, max_iter):
    # Step 1: C = infinity switches the penalty off, which gives the same estimator as above.
    # A finite C (for example 1.0) adds an L2 penalty that shrinks the coefficients.
    model = LogisticRegression(C=np.inf, tol=tol, max_iter=max_iter).fit(X_train, y_train)
    # Step 2: collect intercept and slopes in one named series.
    coefs = pd.Series(np.r_[model.intercept_, model.coef_.ravel()], index=["const"] + list(X_train.columns))
    return model, coefs


def class_measures(y_true, prob, threshold):
    # Step 1: turn probabilities into 0/1 predictions with the chosen threshold.
    pred = (prob >= threshold).astype(int)
    # Step 2: count-based measures use the predictions; AUC and log loss use the probabilities.
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "AUC": roc_auc_score(y_true, prob),
        "Log loss": log_loss(y_true, prob),
    }, confusion_matrix(y_true, pred, labels=[0, 1])


def sigmoid_chart(z_test, y_test, threshold, n_grid):
    # Step 1: smooth grid of scores for the curve, and the test rows as points at 0 or 1.
    grid = np.linspace(z_test.min() - 1, z_test.max() + 1, n_grid)
    curve = pd.DataFrame({"z": grid, "p": sigmoid(grid)})
    points = pd.DataFrame({"z": z_test, "p": y_test.to_numpy()})
    # Step 2: three layers on one pair of axes: curve, points and the threshold line.
    ydom = alt.Scale(domain=[-0.05, 1.05])
    xax = alt.X("z:Q", title="linear score z")
    yax = alt.Y("p:Q", title="probability of y = 1", scale=ydom)
    line = alt.Chart(curve).mark_line(color="#0B6E75", strokeWidth=2.5).encode(x=xax, y=yax)
    dots = alt.Chart(points).mark_circle(color="#C2410C", opacity=0.35, size=40).encode(x=xax, y=yax)
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(
        color="grey", strokeDash=[5, 4]).encode(y=alt.Y("t:Q", scale=ydom))
    return (line + dots + rule).properties(height=320)


# ---------------------------------------------------------------------------
# 3. Small helpers for the page
# ---------------------------------------------------------------------------
def show_source(*functions):
    # Display the source of the functions that the page runs; long lines wrap where Streamlit supports it.
    text = "\n\n".join(inspect.getsource(f).rstrip() for f in functions)
    try:
        st.code(text, language="python", wrap_lines=True)
    except TypeError:
        st.code(text, language="python")


def show_chart(chart):
    # Newer Streamlit versions use width="stretch"; older ones use use_container_width.
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def rounded(obj):
    return obj.round(CONFIG["decimals"])


# ---------------------------------------------------------------------------
# 4. Page
# ---------------------------------------------------------------------------
# Sidebar: every control is read from CONFIG.
with st.sidebar:
    st.header("Settings")
    n_obs = st.slider("Observations", CONFIG["n_obs"]["min"], CONFIG["n_obs"]["max"],
                      CONFIG["n_obs"]["default"], CONFIG["n_obs"]["step"])
    st.caption("True coefficients used to simulate the data")
    true_coefs = [st.number_input("Intercept" if j == 0 else f"x{j}", value=float(v), step=0.1, format="%.2f")
                  for j, v in enumerate(CONFIG["true_coefs"])]
    test_pct = st.slider("Test sample (%)", CONFIG["test_share"]["min"], CONFIG["test_share"]["max"],
                         CONFIG["test_share"]["default"], CONFIG["test_share"]["step"])
    threshold = st.slider("Classification threshold", CONFIG["threshold"]["min"], CONFIG["threshold"]["max"],
                          CONFIG["threshold"]["default"], CONFIG["threshold"]["step"])
    standardize = st.checkbox("Standardize the variables", value=False)
    robust = st.checkbox("Robust (HC0) standard errors in statsmodels", value=False)
    seed = st.number_input("Random seed", value=CONFIG["seed"], step=1)

st.caption(CONFIG["eyebrow"])
st.title(CONFIG["page_title"])
st.write("This page fits a logistic regression three ways on simulated data with known coefficients: "
         "from scratch with numpy, with statsmodels, and with scikit-learn. The code under each heading is the "
         "code that produces the results below it. Change the settings in the left panel and every number updates.")

# Compute everything once, in the order the page explains it.
X, y = make_synthetic(n_obs, true_coefs, int(seed), CONFIG["target"])
X_train, X_test, y_train, y_test = split_and_scale(X, y, test_pct / 100, int(seed), standardize)
b_scratch, se_scratch, n_steps = fit_logit_newton(X_train, y_train, CONFIG["newton_max_iter"], CONFIG["newton_tol"])
sm_fit = fit_statsmodels(X_train, y_train, robust)
skl, b_skl = fit_sklearn(X_train, y_train, CONFIG["sklearn_tol"], CONFIG["sklearn_max_iter"])

# --- 1. The model
st.header("1. The model")
st.write("The model turns a linear score into a probability. The coefficients are chosen to maximize the "
         "log-likelihood. There is no closed-form solution, so the estimation is iterative.")
st.latex(r"z = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k \qquad p = \frac{1}{1 + e^{-z}}")
st.latex(r"\ell(\beta) = \sum_i \left[\, y_i \ln p_i + (1 - y_i)\ln(1 - p_i) \,\right]")
show_source(sigmoid)

# --- 2. Data and split
st.header("2. Data and the 80-20 split")
st.write("The outcome is a 0/1 coin flip whose probability comes from the true coefficients in the left panel. "
         "The split is stratified, so training and test sample contain the same share of 1s. "
         "If you standardize, mean and standard deviation come from the training rows only.")
show_source(make_synthetic, split_and_scale)
preview = X.copy()
preview[CONFIG["target"]] = y
st.dataframe(rounded(preview.head(CONFIG["preview_rows"])))
c1, c2, c3 = st.columns(3)
c1.metric("Training rows", f"{len(X_train):,}")
c2.metric("Test rows", f"{len(X_test):,}")
c3.metric("Share of y = 1", f"{y.mean():.1%}")

# --- 3. From scratch
st.header("3. Estimation from scratch")
st.write("Newton's method, also called iteratively reweighted least squares, updates the coefficients with "
         "the gradient and the curvature of the log-likelihood until they stop moving.")
st.latex(r"\beta_{\text{new}} = \beta + (X'WX)^{-1} X'(y - p), \qquad W = \mathrm{diag}\big(p(1-p)\big)")
show_source(fit_logit_newton)
st.write(f"Converged after {n_steps} steps.")
st.dataframe(rounded(pd.DataFrame({"Coefficient": b_scratch, "Std. error": se_scratch,
                                   "z": b_scratch / se_scratch})))

# --- 4. statsmodels
st.header("4. statsmodels")
st.write("statsmodels reports the full inference table: coefficients, standard errors, z statistics, p-values "
         "and confidence intervals. The odds ratio is exp(coefficient), the factor by which the odds of y = 1 "
         "change when the variable rises by one unit.")
show_source(fit_statsmodels)
st.code(sm_fit.summary().as_text())
st.dataframe(rounded(pd.DataFrame({"Odds ratio": np.exp(sm_fit.params)})))

# --- 5. scikit-learn
st.header("5. scikit-learn")
st.write("scikit-learn is built for prediction. Without a penalty it estimates the same model as the two "
         "methods above, and it gives predicted probabilities for new rows.")
show_source(fit_sklearn)

# --- 6. Comparison
st.header("6. The three sets of coefficients")
table = pd.DataFrame({"From scratch": b_scratch, "statsmodels": sm_fit.params, "scikit-learn": b_skl})
if not standardize:
    table["True value"] = pd.Series(true_coefs, index=table.index)
st.dataframe(rounded(table))
spread = float((table[["From scratch", "statsmodels", "scikit-learn"]].max(axis=1)
                - table[["From scratch", "statsmodels", "scikit-learn"]].min(axis=1)).max())
st.write(f"The largest difference between the three methods is {spread:.1e}. "
         + ("The true values differ from the estimates because the sample is finite."
            if not standardize else
            "With standardized variables the coefficients are on a different scale, so the true values are not shown."))

# --- 7. Test sample
st.header("7. Judging the model on the test sample")
st.write("Predicted probabilities at or above the threshold are classed as 1. Precision is the share of "
         "predicted 1s that are correct, recall is the share of actual 1s that are found. AUC and log loss "
         "use the probabilities and do not depend on the threshold.")
show_source(class_measures)
prob_test = skl.predict_proba(X_test)[:, 1]
measures, cm = class_measures(y_test, prob_test, threshold)
cols = st.columns(3)
for i, (name, value) in enumerate(measures.items()):
    cols[i % 3].metric(name, f"{value:.3f}")
st.caption("Confusion matrix: rows are the actual values, columns the predicted values.")
st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))

# --- 8. Sigmoid
st.header("8. The sigmoid")
st.write("Each point is a test row placed at its linear score, at height 0 or 1 according to its actual outcome. "
         "The curve is the predicted probability. The dashed line is the threshold.")
show_source(sigmoid_chart)
show_chart(sigmoid_chart(skl.decision_function(X_test), y_test, threshold, CONFIG["sigmoid_points"]))

st.caption("To use your own data, open the Logistic regression lab, which accepts CSV and Excel files.")

# ---------------------------------------------------------------------------
# Version log
# ---------------------------------------------------------------------------
# v1.0 (2026-10-09)  New file. Browser version of logit_tutorial.py. Added: CONFIG; sigmoid, make_synthetic,
#   split_and_scale, fit_logit_newton, fit_statsmodels, fit_sklearn, class_measures, sigmoid_chart (Altair, so no
#   extra package is needed); show_source reads the code shown on the page from the executed functions;
#   sidebar controls for sample size, true coefficients, test share, threshold, standardizing, robust errors
#   and seed.
