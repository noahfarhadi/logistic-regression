"""
logit_app.py

Logistic regression lab for students, as one Streamlit app. Run with:

    streamlit run logit_app.py

What a student does: uploads a CSV or Excel file, picks the dependent variable (two values) and the
independent variables, chooses how to prepare them (standardize, reduce skewness, robust standard
errors), and reads the results on the home page: a logistic regression on the full dataset, a
train/test model (80/20 by default, adjustable), and the sigmoid curve of either model.

All texts, defaults and widths are in CONFIG (section 1). Needs: streamlit, pandas, numpy, scipy,
scikit-learn, statsmodels, openpyxl (see requirements.txt).
"""
import io
import math
import re
import warnings
from html import escape

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from scipy.stats import yeojohnson
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1. Settings (edit here)
# ---------------------------------------------------------------------------
CONFIG = {
    "page_title": "Logistic Regression Lab",
    "eyebrow": "FINC 4970 Machine Learning in Finance",
    "app_title": "Logistic regression lab",
    "app_subtitle": ("Upload a dataset, choose the variables, prepare them, and compare a model fitted on "
                     "the full dataset with a model fitted on a training sample and judged on a test sample."),
    # Width of the whole application frame (sidebar plus home page) and of the sidebar, in pixels.
    "frame_width_px": 1120,
    "sidebar_width_px": 320,
    "file_types": ["csv", "xlsx"],
    # Train/test split: values are shares of the data used for testing, in percent.
    "default_test_pct": 20,
    "test_pct_min": 10,
    "test_pct_max": 50,
    "test_pct_step": 5,
    "default_seed": 42,
    "default_threshold": 0.50,
    "threshold_min": 0.05,
    "threshold_max": 0.95,
    "threshold_step": 0.05,
    # A variable counts as skewed when the absolute value of its skewness is above this number.
    "default_skew_cutoff": 1.0,
    "default_penalty_c": 1.0,
    "solver_tol": 1e-10,
    "max_iter": 10000,
    "ci_level": 0.95,
    # A text column with more distinct values than this is not offered as a predictor.
    "max_dummy_levels": 12,
    "min_rows": 20,
    "min_per_class": 5,
    "preview_rows": 8,
    "max_points_plotted": 1500,
    # Streamlit draws its own sliders and checkboxes in red; this turns them towards the teal of the page.
    "widget_hue_deg": 185,
    "plot_width": 720,
    "plot_height": 360,
    "plot_width_narrow": 380,
    "plot_height_narrow": 330,
    "max_points_narrow": 700,
    # Vertical spread of the 0/1 points in the sigmoid chart (share of the 0-1 axis), and points on the curve.
    "jitter_width": 0.08,
    "curve_points": 240,
    "roc_width": 400,
    "roc_height": 400,
}

TRANSFORMS = ["None", "Log", "Square root", "Yeo-Johnson"]
TRANSFORM_TAGS = {"None": "", "Log": "log", "Square root": "sqrt", "Yeo-Johnson": "YJ"}
MISSING_RULES = ["Drop rows with missing values", "Fill gaps (median for numbers, most frequent for text)"]
SE_LABELS = {False: "classical standard errors", True: "robust (HC0) standard errors"}

# ---------------------------------------------------------------------------
# 2. Styling
# ---------------------------------------------------------------------------
# Colours are variables; a second set is used when the visitor's system is in dark mode. The page paints
# its own background, so it looks the same whatever theme Streamlit itself uses. The application frame
# has a fixed maximum width and is centred; the area outside it has a slightly different background.
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');
:root{--outer:#E5E9F0;--paper:#F4F6F9;--panel:#FFFFFF;--ink:#14213D;--body:#2B3A55;--muted:#4A586F;--line:#D3DAE4;
--accent:#0B6E75;--c1:#0B6E75;--c0:#C2410C;--soft:#E3F1F2;}
@media (prefers-color-scheme: dark){:root{--outer:#070B13;--paper:#0D1420;--panel:#142033;--ink:#E8ECF4;
--body:#C9D2E2;--muted:#A5B0C4;--line:#2A3850;--accent:#5CC8CF;--c1:#5CC8CF;--c0:#F4A26B;--soft:#16303A;}}
.stApp{background:var(--outer) !important;}
[data-testid="stAppViewContainer"]{max-width:__FRAME__px;margin:0 auto;background:var(--paper) !important;
box-shadow:0 0 0 1px var(--line);}
[data-testid="stMain"],section.main{background:var(--paper) !important;}
header[data-testid="stHeader"]{background:transparent !important;}
[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{display:none !important;}
section[data-testid="stSidebar"]{width:__SIDEBAR__px !important;min-width:__SIDEBAR__px !important;
max-width:100vw;background:var(--panel) !important;border-right:1px solid var(--line);}
section[data-testid="stSidebar"][aria-expanded="false"]{display:none !important;}
section[data-testid="stSidebar"] > div{width:100% !important;}
[data-testid="stSidebarContent"]{padding-top:.5rem;}
[data-testid="stSidebarUserContent"]{padding-top:1rem;}
.block-container{max-width:100% !important;padding:2.2rem 1.6rem 3rem !important;}
[data-baseweb="tag"],[data-tag]{background:var(--accent) !important;}
[data-baseweb="tag"] *,[data-tag] *{color:var(--panel) !important;}
[data-baseweb="tab-highlight"],[class*="SelectionIndicator"]{background:var(--accent) !important;}
[role="tab"][aria-selected="true"],[role="tab"][aria-selected="true"] *{color:var(--accent) !important;}
[data-testid="stSlider"],[data-testid="stCheckbox"],[data-testid="stRadio"]{filter:hue-rotate(__HUE__deg);}
.lg{color:var(--ink);line-height:1.5;}
.lg-title,.lg-h2,.lg-tv{font-family:'Source Serif 4',Georgia,'Times New Roman',serif;}
.lg-eyebrow{font-size:.92rem;color:var(--muted);}
.lg-title{font-size:2.1rem;font-weight:600;letter-spacing:-0.01em;line-height:1.15;margin:.25rem 0 0;}
.lg-subtitle{margin-top:.7rem;max-width:42rem;font-size:1.04rem;color:var(--body);}
.lg-step{display:flex;align-items:center;gap:.6rem;margin:1.1rem 0 .4rem;font-weight:600;font-size:1.02rem;
color:var(--ink);}
.lg-n{display:inline-flex;align-items:center;justify-content:center;width:1.55rem;height:1.55rem;border-radius:50%;
background:var(--accent);color:var(--panel);font-size:.85rem;flex:none;}
.lg-cap{font-size:.86rem;color:var(--muted);line-height:1.45;}
.lg-sec{margin-top:2rem;}
.lg-h2{font-size:1.5rem;font-weight:600;line-height:1.2;margin:0;color:var(--ink);}
.lg-sub{margin-top:.35rem;font-size:.95rem;color:var(--muted);}
.lg-chips{display:flex;flex-wrap:wrap;gap:.45rem;margin:.9rem 0 .2rem;}
.lg-chip{background:var(--soft);color:var(--ink);border-radius:999px;padding:.2rem .75rem;font-size:.86rem;}
.lg-tiles{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem;margin-top:1rem;}
.lg-n2{grid-template-columns:repeat(2,minmax(0,1fr));}.lg-n3{grid-template-columns:repeat(3,minmax(0,1fr));}
.lg-n4{grid-template-columns:repeat(4,minmax(0,1fr));}.lg-n5{grid-template-columns:repeat(5,minmax(0,1fr));}
.lg-n6{grid-template-columns:repeat(6,minmax(0,1fr));}.lg-captop{margin-top:.6rem;}
.lg-tile{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.7rem .8rem;min-width:0;}
.lg-tl{font-size:.8rem;color:var(--muted);}
.lg-tv{font-size:1.4rem;font-weight:600;line-height:1.25;color:var(--ink);}
.lg-ts{font-size:.78rem;color:var(--muted);}
.lg-wrap{overflow-x:auto;margin-top:1rem;background:var(--panel);border:1px solid var(--line);border-radius:6px;}
.lg-table{border-collapse:collapse;width:100%;min-width:max-content;font-size:.9rem;color:var(--ink);}
.lg-table th{text-align:right;font-weight:600;padding:.5rem .6rem;border-bottom:1px solid var(--line);
background:var(--soft);white-space:nowrap;}
.lg-table td{text-align:right;padding:.42rem .6rem;border-bottom:1px solid var(--line);white-space:nowrap;}
.lg-table th:first-child,.lg-table td:first-child{text-align:left;white-space:normal;word-break:break-word;}
.lg-table tr:last-child td{border-bottom:none;}
.lg-two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem;margin-top:1rem;align-items:start;}
.lg-two .lg-wrap,.lg-two .lg-plot{margin-top:0;}
.lg-two .lg-card{margin-top:1rem;}
.lg-card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.9rem 1rem;min-width:0;}
.lg-card .lg-ch{font-weight:600;font-size:.95rem;margin-bottom:.5rem;}
.lg-cm{display:grid;grid-template-columns:auto repeat(2,minmax(0,1fr));gap:.4rem;align-items:center;font-size:.88rem;}
.lg-cmh{color:var(--muted);text-align:center;}
.lg-cmr{color:var(--muted);}
.lg-cmc{background:var(--soft);border-radius:5px;padding:.7rem .3rem;text-align:center;font-family:'Source Serif 4',Georgia,serif;
font-size:1.35rem;font-weight:600;}
.lg-cmc small{display:block;font-family:inherit;font-size:.75rem;font-weight:400;color:var(--muted);}
.lg-plot{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.6rem .6rem .3rem;margin-top:.8rem;}
.lg-svg{width:100%;height:auto;display:block;}
.lg-svg text{fill:var(--muted);font-size:12px;font-family:'Source Sans 3','Source Sans Pro',sans-serif;}
.lg-svg .lg-axtitle{fill:var(--body);font-size:12.5px;}
.lg-svg .lg-grid{stroke:var(--line);stroke-width:1;}
.lg-svg .lg-axis{stroke:var(--muted);stroke-width:1;}
.lg-svg .lg-curve{fill:none;stroke:var(--accent);stroke-width:2.6;}
.lg-svg .lg-roc{fill:none;stroke:var(--c1);stroke-width:2.4;}
.lg-svg .lg-roc2{fill:none;stroke:var(--c0);stroke-width:1.8;stroke-dasharray:5 3;}
.lg-svg .lg-diag{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4;}
.lg-svg .lg-thr{stroke:var(--ink);stroke-width:1.2;stroke-dasharray:5 4;opacity:.7;}
.lg-svg .lg-p1{fill:var(--c1);fill-opacity:.55;}
.lg-svg .lg-p0{fill:var(--c0);fill-opacity:.55;}
.lg-svg .lg-h1{fill:none;stroke:var(--c1);stroke-width:1.4;}
.lg-svg .lg-h0{fill:none;stroke:var(--c0);stroke-width:1.4;}
.lg-legend{display:flex;flex-wrap:wrap;gap:.3rem 1.1rem;padding:.2rem .4rem .4rem;font-size:.85rem;color:var(--body);}
.lg-key{display:inline-block;width:.75rem;height:.75rem;border-radius:50%;margin-right:.35rem;vertical-align:-1px;}
.lg-k1{background:var(--c1);}.lg-k0{background:var(--c0);}
.lg-kh1{border:2px solid var(--c1);}.lg-kh0{border:2px solid var(--c0);}
.lg-kl{display:inline-block;width:1.1rem;height:0;border-top:3px solid var(--accent);margin-right:.35rem;vertical-align:3px;}
.lg-kd{display:inline-block;width:1.1rem;height:0;border-top:2px dashed var(--ink);margin-right:.35rem;vertical-align:3px;}
.lg-note{margin-top:.8rem;font-size:.9rem;color:var(--body);}
.lg-eq{margin-top:.8rem;font-family:'Source Serif 4',Georgia,serif;font-size:1.05rem;color:var(--ink);}
.lg-narrow{display:none;}
.lg-rocbox{max-width:460px;margin:0 auto;}
.lg-steps{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem;margin-top:1.2rem;}
@media (max-width: 1000px){.lg-n5,.lg-n6{grid-template-columns:repeat(3,minmax(0,1fr));}}
@media (max-width: 900px) and (min-width: 769px){.lg-wide{display:none;}.lg-narrow{display:block;}}
@media (max-width: 520px){.lg-wide{display:none;}.lg-narrow{display:block;}}
@media (max-width: 900px){.lg-two{grid-template-columns:minmax(0,1fr);}}
@media (max-width: 760px){.lg-two,.lg-steps{grid-template-columns:minmax(0,1fr);}
.lg-tiles.lg-n2,.lg-tiles.lg-n3,.lg-tiles.lg-n4,.lg-tiles.lg-n5,.lg-tiles.lg-n6{grid-template-columns:repeat(2,minmax(0,1fr));}
.block-container{padding:1.6rem 1rem 2.5rem !important;}.lg-title{font-size:1.7rem;}}
"""


def inject_css():
    css = CSS.replace("__FRAME__", str(CONFIG["frame_width_px"])).replace("__SIDEBAR__", str(CONFIG["sidebar_width_px"])).replace("__HUE__", str(CONFIG["widget_hue_deg"]))
    css = re.sub(r"\s+", " ", css)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. Small helpers (text, numbers, HTML pieces)
# ---------------------------------------------------------------------------
def esc(value):
    """Escape text for HTML. The dollar sign is also escaped so that Streamlit does not read it as maths."""
    return escape(str(value)).replace("$", "&#36;")


def md_safe(value):
    """Make a column name safe inside a Streamlit widget label (which is read as markdown)."""
    return str(value).replace("$", "\\$").replace("*", "\\*")


def label_text(v):
    """Text for a category value: whole floats such as 1.0 are shown as 1."""
    return f"{v:g}" if isinstance(v, (float, np.floating)) else str(v)


def fnum(x, d=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:,.{d}f}"


def fp(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def tiles_html(items):
    """items: list of (label, value, small text). One bordered tile each."""
    cells = "".join(
        f'<div class="lg-tile"><div class="lg-tl">{esc(a)}</div><div class="lg-tv">{esc(b)}</div>'
        f'<div class="lg-ts">{esc(c)}</div></div>' for a, b, c in items)
    return f'<div class="lg"><div class="lg-tiles lg-n{len(items)}">{cells}</div></div>'


def table_html(headers, rows):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="lg"><div class="lg-wrap"><table class="lg-table"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div></div>'


def section_html(title, sub=None):
    s = f'<div class="lg-sub">{esc(sub)}</div>' if sub else ""
    return f'<div class="lg lg-sec"><div class="lg-h2" role="heading" aria-level="2">{esc(title)}</div>{s}</div>'


def show(html_text):
    st.markdown(html_text, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 4. Reading and describing the uploaded data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def excel_sheets(raw):
    return pd.ExcelFile(io.BytesIO(raw)).sheet_names


@st.cache_data(show_spinner=False)
def read_table(raw, name, sheet):
    """Read a CSV (delimiter detected, two encodings tried) or an Excel sheet into a DataFrame."""
    if name.lower().endswith(".csv"):
        last = None
        for enc in ("utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
            except UnicodeDecodeError as err:
                last = err
        raise last
    return pd.read_excel(io.BytesIO(raw), sheet_name=sheet)


def tidy(df):
    """Trim column names, drop empty rows and columns, and turn text columns that hold only numbers into numbers."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == object:
            as_num = pd.to_numeric(df[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan}), errors="coerce")
            if as_num.notna().sum() == df[c].notna().sum() and as_num.notna().any():
                df[c] = as_num
    return df


def is_numeric(s):
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def target_candidates(df):
    return [c for c in df.columns if df[c].dropna().nunique() == 2]


def feature_candidates(df, target):
    """Columns that can serve as predictors: numeric columns with variation, and text columns with few levels."""
    ok = []
    for c in df.columns:
        if c == target or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        n = df[c].dropna().nunique()
        if n < 2:
            continue
        if is_numeric(df[c]) or n <= CONFIG["max_dummy_levels"]:
            ok.append(c)
    return ok


def preview_value(v):
    """Text for one cell of the preview table: blanks for missing values and no trailing .0 on whole numbers."""
    if pd.isna(v):
        return ""
    return f"{v:g}" if isinstance(v, (float, np.floating)) else v


def column_overview(df):
    rows = []
    for c in df.columns:
        s = df[c]
        kind = "number" if is_numeric(s) else ("date" if pd.api.types.is_datetime64_any_dtype(s) else "text")
        rows.append([c, kind, f"{int(s.isna().sum()):,}", f"{int(s.nunique()):,}"])
    return rows


# ---------------------------------------------------------------------------
# 5. Building the design matrix
# ---------------------------------------------------------------------------
def build_design(df, target, positive, features, missing_rule):
    """Return a dict with y (0/1), X (all predictors as numbers), the names of the continuous and the
    0/1 predictors, and a list of notes about what was done (rows dropped, dummies created)."""
    notes = []
    data = df[[target] + list(features)].copy()
    n_raw = len(data)
    data = data[data[target].notna()]
    if missing_rule == MISSING_RULES[0]:
        data = data.dropna()
    else:
        for c in features:
            if data[c].isna().any():
                data[c] = data[c].fillna(data[c].median() if is_numeric(data[c]) else data[c].mode().iloc[0])
    data = data.reset_index(drop=True)
    if n_raw - len(data) > 0:
        notes.append(f"{n_raw - len(data):,} of {n_raw:,} rows were removed because of missing values.")
    y = (data[target] == positive).astype(int)

    parts, continuous, binary = [], [], []
    for c in features:
        s = data[c]
        if is_numeric(s):
            s = s.astype(float)
            if s.nunique() < 2:
                notes.append(f"{c} has no variation after cleaning and was left out.")
                continue
            parts.append(s.rename(c))
            (binary if s.nunique() == 2 else continuous).append(c)
        else:
            d = pd.get_dummies(s.astype(str), prefix=c, drop_first=True, dtype=float)
            if d.shape[1] == 0:
                notes.append(f"{c} has only one level after cleaning and was left out.")
                continue
            notes.append(f"{c} was converted to {d.shape[1]} dummy variable(s); the first level is the reference.")
            parts.extend([d[k] for k in d.columns])
            binary.extend(list(d.columns))
    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=data.index)
    return {"y": y, "X": X, "continuous": continuous, "binary": binary, "notes": notes, "n_raw": n_raw}


# ---------------------------------------------------------------------------
# 6. Preparation: skewness correction and standardization
# ---------------------------------------------------------------------------
class Prep:
    """Learns the transformation of each continuous predictor from one data set (fit) and applies it to
    that or another data set (transform). 0/1 predictors are never transformed or standardized."""

    def __init__(self, continuous, standardize, method, cutoff, overrides):
        self.continuous, self.standardize = list(continuous), standardize
        self.method, self.cutoff, self.overrides = method, cutoff, overrides
        self.plan = {}

    def fit(self, X):
        for c in self.continuous:
            x = X[c].to_numpy(float)
            skew_before = float(pd.Series(x).skew())
            chosen = self.overrides.get(c, "Automatic")
            if chosen == "Automatic":
                chosen = self.method if (np.isfinite(skew_before) and abs(skew_before) > self.cutoff) else "None"
            p = {"method": chosen, "skew_before": skew_before, "shift": 0.0, "lam": None, "mean": 0.0, "sd": 1.0}
            lo = float(x.min())
            if chosen == "Log":
                p["shift"] = 0.0 if lo > 0 else 1.0 - lo
            elif chosen == "Square root":
                p["shift"] = 0.0 if lo >= 0 else -lo
            elif chosen == "Yeo-Johnson":
                p["lam"] = float(yeojohnson(x)[1])
            t = self._apply(x, p)
            if self.standardize:
                p["mean"] = float(np.mean(t))
                sd = float(np.std(t))
                p["sd"] = sd if sd > 0 else 1.0
            p["skew_after"] = float(pd.Series(t).skew())
            self.plan[c] = p
        return self

    @staticmethod
    def _apply(x, p):
        m = p["method"]
        if m == "Log":
            return np.log(np.clip(x + p["shift"], 1e-12, None))
        if m == "Square root":
            return np.sqrt(np.clip(x + p["shift"], 0.0, None))
        if m == "Yeo-Johnson":
            return yeojohnson(x, lmbda=p["lam"])
        return x.copy()

    def label(self, c):
        tags = []
        if c in self.plan:
            if TRANSFORM_TAGS[self.plan[c]["method"]]:
                tags.append(TRANSFORM_TAGS[self.plan[c]["method"]])
            if self.standardize:
                tags.append("z")
        return f"{c} ({', '.join(tags)})" if tags else c

    def transform(self, X):
        out = X.copy()
        for c in self.continuous:
            p = self.plan[c]
            t = self._apply(X[c].to_numpy(float), p)
            if self.standardize:
                t = (t - p["mean"]) / p["sd"]
            out[c] = t
        return out.rename(columns={c: self.label(c) for c in self.continuous})


# ---------------------------------------------------------------------------
# 7. Models and measures
# ---------------------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def class_measures(y, p, threshold):
    """Confusion counts, accuracy, precision, recall, F1, AUC and log loss for probabilities p."""
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp_, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    n = len(y)
    prec = tp / (tp + fp_) if (tp + fp_) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0) else float("nan")
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    q = np.clip(p, 1e-15, 1 - 1e-15)
    ll = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    return {"tp": tp, "tn": tn, "fp": fp_, "fn": fn, "n": n, "accuracy": (tp + tn) / n,
            "precision": prec, "recall": rec, "f1": f1, "auc": auc, "logloss": ll}


def check_rank(Xp):
    """Stop with a readable message when predictors carry exactly the same information (perfect collinearity)."""
    if Xp.shape[1] > 1 and np.linalg.matrix_rank(Xp.to_numpy(float)) < Xp.shape[1]:
        raise ValueError("Some predictors are perfectly collinear: one is an exact combination of the others. "
                         "Remove one of the predictors that carry the same information.")


def fit_full_logit(Xp, y, robust, ci_level):
    """Logistic regression on all rows with statsmodels. Returns a dict, or raises ValueError with a message."""
    check_rank(Xp)
    Xc = sm.add_constant(Xp.astype(float), has_constant="add")
    notes = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sm.Logit(np.asarray(y, dtype=float), Xc).fit(disp=0, maxiter=200, cov_type="HC0" if robust else "nonrobust")
        text = " ".join(str(w.message).lower() for w in caught)
        if "separation" in text:
            notes.append("The data show (near) perfect separation: some combination of predictors predicts the "
                         "outcome without error, so coefficients and standard errors are not reliable.")
        if not res.mle_retvals.get("converged", True):
            notes.append("The optimizer did not converge. Treat the estimates with caution.")
    except np.linalg.LinAlgError:
        raise ValueError("The predictors are perfectly correlated (singular matrix). Remove one of the "
                         "predictors that carry the same information.")
    except Exception as err:  # statsmodels raises several error types for separation and failed fits
        raise ValueError(f"The model could not be fitted: {err}")
    ci = res.conf_int(alpha=1 - ci_level)
    names = list(Xc.columns)
    with np.errstate(over="ignore"):
        table = pd.DataFrame({
            "name": names, "coef": res.params.values, "se": res.bse.values, "z": res.tvalues.values,
            "p": res.pvalues.values, "or": np.exp(res.params.values),
            "or_lo": np.exp(ci.iloc[:, 0].values), "or_hi": np.exp(ci.iloc[:, 1].values)})
    prob = np.asarray(res.predict(Xc))
    return {"res": res, "table": table, "prob": prob, "b0": float(res.params.iloc[0]),
            "b": res.params.values[1:].astype(float), "notes": notes,
            "llr": float(res.llr), "llr_p": float(res.llr_pvalue), "r2": float(res.prsquared),
            "aic": float(res.aic), "bic": float(res.bic)}


def fit_split_model(X, y, prep_args, test_share, seed, stratify, penalty, c_value):
    """Split once, learn the preparation on the training rows only, fit scikit-learn logistic regression."""
    idx = np.arange(len(y))
    try:
        tr, te = train_test_split(idx, test_size=test_share, random_state=int(seed),
                                  stratify=y if stratify else None)
    except ValueError as err:
        raise ValueError(f"The data could not be split: {err}")
    for name, part in (("training", tr), ("test", te)):
        counts = np.bincount(y.iloc[part], minlength=2)
        if counts.min() < CONFIG["min_per_class"]:
            raise ValueError(f"The {name} sample has fewer than {CONFIG['min_per_class']} observations in one class. "
                             "Use a different seed, switch on stratified sampling, or change the split.")
    prep = Prep(*prep_args).fit(X.iloc[tr])
    Xtr, Xte = prep.transform(X.iloc[tr]), prep.transform(X.iloc[te])
    if not (np.isfinite(Xtr.to_numpy()).all() and np.isfinite(Xte.to_numpy()).all()):
        raise ValueError("A transformation produced invalid values in the training or test sample. "
                         "Choose a different transformation for the skewed variables.")
    check_rank(Xtr)
    ytr, yte = y.iloc[tr].to_numpy(), y.iloc[te].to_numpy()
    clf = LogisticRegression(C=(c_value if penalty != "None" else np.inf), solver="lbfgs", max_iter=CONFIG["max_iter"],
                             tol=CONFIG["solver_tol"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(Xtr.to_numpy(), ytr)
    ptr, pte = clf.predict_proba(Xtr.to_numpy())[:, 1], clf.predict_proba(Xte.to_numpy())[:, 1]
    return {"prep": prep, "clf": clf, "Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte, "ptr": ptr, "pte": pte,
            "b0": float(clf.intercept_[0]), "b": clf.coef_[0].astype(float), "names": list(Xtr.columns),
            "n_train": len(tr), "n_test": len(te)}


# ---------------------------------------------------------------------------
# 8. Charts (inline SVG, so they follow the light and dark colours of the page)
# ---------------------------------------------------------------------------
def nice_ticks(lo, hi, n=5):
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    start = math.ceil(lo / step - 1e-9) * step
    ticks, v = [], start
    while v <= hi + 1e-9:
        ticks.append(round(v, 10))
        v += step
    return ticks


class Plot:
    """A minimal SVG scatter and line plot with axes, grid and tick labels."""

    def __init__(self, xlim, ylim, xlabel, ylabel, yticks=None, size=None, margins=(58, 16, 12, 50), n_xticks=7):
        self.w, self.h = size or (CONFIG["plot_width"], CONFIG["plot_height"])
        self.ml, self.mr, self.mt, self.mb = margins
        self.n_xticks = n_xticks
        self.x0, self.x1 = xlim
        self.y0, self.y1 = ylim
        self.xlabel, self.ylabel, self.yticks = xlabel, ylabel, yticks
        self.items = []

    def X(self, x):
        return self.ml + (x - self.x0) / (self.x1 - self.x0) * (self.w - self.ml - self.mr)

    def Y(self, y):
        return self.mt + (self.h - self.mt - self.mb) - (y - self.y0) / (self.y1 - self.y0) * (self.h - self.mt - self.mb)

    def line(self, xs, ys, cls):
        pts = " ".join(f"{self.X(a):.1f},{self.Y(b):.1f}" for a, b in zip(xs, ys))
        self.items.append(f'<polyline class="{cls}" points="{pts}"/>')

    def points(self, xs, ys, cls, r=3.2):
        self.items.append("".join(f'<circle class="{cls}" cx="{self.X(a):.1f}" cy="{self.Y(b):.1f}" r="{r}"/>'
                                  for a, b in zip(xs, ys)))

    def hline(self, y, cls):
        self.items.append(f'<line class="{cls}" x1="{self.ml}" x2="{self.w - self.mr}" y1="{self.Y(y):.1f}" y2="{self.Y(y):.1f}"/>')

    def vline(self, x, cls):
        self.items.append(f'<line class="{cls}" y1="{self.mt}" y2="{self.h - self.mb}" x1="{self.X(x):.1f}" x2="{self.X(x):.1f}"/>')

    def segment(self, x0, y0, x1, y1, cls):
        self.items.append(f'<line class="{cls}" x1="{self.X(x0):.1f}" y1="{self.Y(y0):.1f}" x2="{self.X(x1):.1f}" y2="{self.Y(y1):.1f}"/>')

    def render(self, title):
        grid = []
        for t in nice_ticks(self.x0, self.x1, self.n_xticks):
            grid.append(f'<line class="lg-grid" x1="{self.X(t):.1f}" x2="{self.X(t):.1f}" y1="{self.mt}" y2="{self.h - self.mb}"/>'
                        f'<text x="{self.X(t):.1f}" y="{self.h - self.mb + 18}" text-anchor="middle">{t:g}</text>')
        for t in (self.yticks or nice_ticks(self.y0, self.y1, 5)):
            grid.append(f'<line class="lg-grid" x1="{self.ml}" x2="{self.w - self.mr}" y1="{self.Y(t):.1f}" y2="{self.Y(t):.1f}"/>'
                        f'<text x="{self.ml - 8}" y="{self.Y(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
        axes = (f'<line class="lg-axis" x1="{self.ml}" x2="{self.w - self.mr}" y1="{self.h - self.mb}" y2="{self.h - self.mb}"/>'
                f'<line class="lg-axis" x1="{self.ml}" x2="{self.ml}" y1="{self.mt}" y2="{self.h - self.mb}"/>')
        labels = (f'<text class="lg-axtitle" x="{(self.ml + self.w - self.mr) / 2:.0f}" y="{self.h - 8}" text-anchor="middle">{esc(self.xlabel)}</text>'
                  f'<text class="lg-axtitle" transform="translate(15 {(self.mt + self.h - self.mb) / 2:.0f}) rotate(-90)" text-anchor="middle">{esc(self.ylabel)}</text>')
        return (f'<svg class="lg-svg" viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{esc(title)}">'
                f'<title>{esc(title)}</title>{"".join(grid)}{axes}{"".join(self.items)}{labels}</svg>')


def plot_box(svg, legend_html):
    return f'<div class="lg"><div class="lg-plot">{svg}<div class="lg-legend">{legend_html}</div></div></div>'


def plot_box_two(wide, narrow):
    """wide and narrow are (svg, legend) pairs. The narrow drawing replaces the wide one on small screens, so
    the text in the chart stays readable instead of shrinking with the page."""
    return ('<div class="lg"><div class="lg-plot">'
            f'<div class="lg-wide">{wide[0]}<div class="lg-legend">{wide[1]}</div></div>'
            f'<div class="lg-narrow">{narrow[0]}<div class="lg-legend">{narrow[1]}</div></div></div></div>')


def sigmoid_chart(view, which, threshold, seed):
    """Sigmoid of the chosen model. which = 'score' (z = b0 + sum b*x) or the name of one prepared predictor
    (the other predictors are held at their average in the sample the model was fitted on)."""
    Xp, y, group = view["Xp"], view["y"], view["group"]
    b0, b = view["b0"], view["b"]
    names = list(Xp.columns)
    if which == "score":
        xs = b0 + Xp.to_numpy() @ b
        curve_of = lambda g: sigmoid(g)
        xlabel = "Linear score z = b0 + b1 x1 + b2 x2 + ..."
        thr_x = math.log(threshold / (1 - threshold))
    else:
        j = names.index(which)
        base = b0 + float(np.dot(np.delete(view["means"], j), np.delete(b, j)))
        xs = Xp[which].to_numpy()
        curve_of = lambda g: sigmoid(base + b[j] * g)
        xlabel = which
        thr_x = ((math.log(threshold / (1 - threshold)) - base) / b[j]) if b[j] != 0 else None
    n = len(xs)
    jitter = (np.random.default_rng(int(seed)).random(n) - 0.5) * CONFIG["jitter_width"]
    lo, hi = float(np.min(xs)), float(np.max(xs))
    pad = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - pad, hi + pad
    grid = np.linspace(lo, hi, CONFIG["curve_points"])
    two = bool((group == "test").any())

    def draw(size, margins, n_xticks, cap):
        rng = np.random.default_rng(int(seed) + 1)
        keep = np.arange(n) if n <= cap else np.sort(rng.choice(n, cap, replace=False))
        inside = np.zeros(n, bool)
        inside[keep] = True
        pl = Plot((lo, hi), (-0.08, 1.08), xlabel, "Probability of the event", yticks=[0, 0.25, 0.5, 0.75, 1],
                  size=size, margins=margins, n_xticks=n_xticks)
        pl.hline(threshold, "lg-thr")
        if thr_x is not None and lo <= thr_x <= hi:
            pl.vline(thr_x, "lg-thr")
        pl.line(grid, curve_of(grid), "lg-curve")
        for cls_y in (0, 1):
            for grp, shape in (("train", "p"), ("all", "p"), ("test", "h")):
                m = inside & (y == cls_y) & (group == grp)
                if m.any():
                    pl.points(xs[m], cls_y + jitter[m], f"lg-{shape}{cls_y}", r=3.2 if shape == "p" else 3.6)
        legend = ('<span><i class="lg-kl"></i>Sigmoid curve</span><span><i class="lg-kd"></i>Threshold</span>'
                  '<span><i class="lg-key lg-k1"></i>Event (1)' + (", training" if two else "") + '</span>'
                  '<span><i class="lg-key lg-k0"></i>No event (0)' + (", training" if two else "") + '</span>')
        if two:
            legend += ('<span><i class="lg-key lg-kh1"></i>Event (1), test</span>'
                       '<span><i class="lg-key lg-kh0"></i>No event (0), test</span>')
        if n > len(keep):
            legend += f"<span>Showing {len(keep):,} of {n:,} observations</span>"
        return pl.render("Sigmoid curve and observations"), legend

    wide = draw((CONFIG["plot_width"], CONFIG["plot_height"]), (58, 16, 12, 50), 7, CONFIG["max_points_plotted"])
    narrow = draw((CONFIG["plot_width_narrow"], CONFIG["plot_height_narrow"]), (54, 12, 10, 46), 5,
                  CONFIG["max_points_narrow"])
    return plot_box_two(wide, narrow)


def roc_chart(ytr, ptr, yte, pte, auc_tr, auc_te):
    pl = Plot((0, 1), (0, 1), "False positive rate", "True positive rate", size=(CONFIG["roc_width"], CONFIG["roc_height"]))
    pl.segment(0, 0, 1, 1, "lg-diag")
    f, t, _ = roc_curve(ytr, ptr)
    pl.line(f, t, "lg-roc2")
    f, t, _ = roc_curve(yte, pte)
    pl.line(f, t, "lg-roc")
    legend = (f'<span><i class="lg-kl"></i>Test, AUC {fnum(auc_te)}</span>'
              f'<span><i class="lg-kd"></i>Training, AUC {fnum(auc_tr)}</span>')
    return plot_box(pl.render("ROC curve"), legend)


# ---------------------------------------------------------------------------
# 9. Page parts
# ---------------------------------------------------------------------------
def step(n, title):
    st.markdown(f'<div class="lg lg-step"><span class="lg-n">{n}</span>{esc(title)}</div>', unsafe_allow_html=True)


def header():
    show(f'<div class="lg"><div class="lg-eyebrow">{esc(CONFIG["eyebrow"])}</div>'
         f'<div class="lg-title" role="heading" aria-level="1">{esc(CONFIG["app_title"])}</div>'
         f'<div class="lg-subtitle">{esc(CONFIG["app_subtitle"])}</div></div>')


def welcome():
    steps = [("1", "Upload", "A CSV or Excel file with one row per observation and a column that has exactly two values."),
             ("2", "Choose variables", "Pick the dependent variable and the event, then the independent variables."),
             ("3", "Prepare", "Standardize, reduce skewness and choose standard errors that allow for heteroskedasticity."),
             ("4", "Validate", "Set the share of data kept for testing and the classification threshold.")]
    cards = "".join(f'<div class="lg-card"><div class="lg-ch">{n}. {esc(t)}</div><div class="lg-cap">{esc(d)}</div></div>'
                    for n, t, d in steps)
    show(section_html("Start with a dataset", "The steps are in the panel on the left. Results appear here as soon as the choices are complete.")
         + f'<div class="lg"><div class="lg-steps">{cards}</div></div>')


def strip_chips(items):
    show('<div class="lg"><div class="lg-chips">' + "".join(f'<span class="lg-chip">{esc(i)}</span>' for i in items) + "</div></div>")


def coef_table(table, label_for_or):
    rows = [[r["name"] if r["name"] != "const" else "Constant", fnum(r["coef"], 4), fnum(r["se"], 4), fnum(r["z"], 2),
             fp(r["p"]), fnum(r["or"], 3), fnum(r["or_lo"], 3), fnum(r["or_hi"], 3)] for _, r in table.iterrows()]
    return table_html(["Variable", "Coef.", "Std. error", "z", "p-value", label_for_or, "OR low", "OR high"], rows)


def measures_table(mtr, mte):
    rows = [[lab, fnum(mtr[k]), fnum(mte[k])] for lab, k in
            (("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 score", "f1"),
             ("AUC", "auc"), ("Log loss", "logloss"))]
    return table_html(["Measure", "Training", "Test"], rows)


def confusion_html(m):
    def cell(v, tot):
        return f'<div class="lg-cmc">{v:,}<small>{(v / tot * 100 if tot else 0):.0f}% of row</small></div>'
    r0, r1 = m["tn"] + m["fp"], m["fn"] + m["tp"]
    return ('<div class="lg"><div class="lg-card"><div class="lg-ch">Confusion matrix, test sample</div><div class="lg-cm">'
            '<div></div><div class="lg-cmh">Predicted 0</div><div class="lg-cmh">Predicted 1</div>'
            f'<div class="lg-cmr">Actual 0</div>{cell(m["tn"], r0)}{cell(m["fp"], r0)}'
            f'<div class="lg-cmr">Actual 1</div>{cell(m["fn"], r1)}{cell(m["tp"], r1)}</div></div></div>')


# ---------------------------------------------------------------------------
# 10. The application
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=CONFIG["page_title"], layout="wide", initial_sidebar_state="expanded")
    inject_css()
    header()

    # Step 1 in the sidebar: the upload. Nothing else is shown until a file has been read.
    with st.sidebar:
        step(1, "Data")
        up = st.file_uploader("CSV or Excel file", type=CONFIG["file_types"], key="upload")
    if up is None:
        with st.sidebar:
            st.markdown('<div class="lg lg-cap">The next steps appear after you upload a file.</div>', unsafe_allow_html=True)
        welcome()
        return

    raw = up.getvalue()
    sig = f"{up.name}-{len(raw)}"
    sheet = 0
    with st.sidebar:
        if up.name.lower().endswith(".xlsx"):
            sheets = excel_sheets(raw)
            sheet = st.selectbox("Sheet", sheets, key=f"sheet::{sig}") if len(sheets) > 1 else sheets[0]
    sig += f"-{sheet}"
    key = lambda name: f"{name}::{sig}"
    try:
        df = tidy(read_table(raw, up.name, sheet))
    except Exception as err:
        st.error(f"The file could not be read: {err}")
        return
    if df.shape[0] < CONFIG["min_rows"] or df.shape[1] < 2:
        st.error(f"The file needs at least {CONFIG['min_rows']} rows and two columns.")
        return
    candidates = target_candidates(df)
    if not candidates:
        st.error("No column has exactly two distinct values, so there is no possible dependent variable. "
                 "Logistic regression needs a two-valued outcome.")
        return

    # Steps 2 to 4 in the sidebar: variables, preparation, validation.
    with st.sidebar:
        st.markdown(f'<div class="lg lg-cap">{df.shape[0]:,} rows, {df.shape[1]:,} columns.</div>', unsafe_allow_html=True)
        step(2, "Variables")
        target = st.selectbox("Dependent variable (two values)", candidates, index=None,
                              placeholder="Choose a column", key=key("target"))
        positive = None
        if target is not None:
            levels = sorted(df[target].dropna().unique().tolist(), key=str)
            default = levels.index(1) if 1 in levels else len(levels) - 1
            positive = st.selectbox("Event coded as 1", levels, index=default, key=key("positive"), format_func=label_text)
        options = feature_candidates(df, target) if target is not None else []
        features = st.multiselect("Independent variables", options, key=key("features"),
                                  placeholder="Choose one or more columns",
                                  help="Number columns are used as they are. Text columns with few levels become dummy variables.")
        left_out = [c for c in df.columns if c != target and c not in options]
        if target is not None and left_out:
            shown = ", ".join(left_out[:6]) + (f" and {len(left_out) - 6} more" if len(left_out) > 6 else "")
            st.markdown(f'<div class="lg lg-cap">Not offered (no variation, dates, or text with more than '
                        f'{CONFIG["max_dummy_levels"]} levels): {esc(shown)}</div>', unsafe_allow_html=True)
        missing_rule = st.selectbox("Missing values", MISSING_RULES, key=key("missing"))

        step(3, "Preparation")
        standardize = st.checkbox("Standardize (mean 0, standard deviation 1)", value=True, key=key("std"),
                                  help="Applies to number columns with more than two values. 0/1 variables are left as they are.")
        method = st.selectbox("Skewness correction", TRANSFORMS, key=key("skew"),
                              help="Log and square root reduce right skewness and the pull of very large values. "
                                   "Yeo-Johnson also works with zero, negative values and left skew. The rule is applied to "
                                   "variables whose absolute skewness is above the cutoff; you can set each variable "
                                   "separately on the Data and preparation tab.")
        cutoff = st.number_input("Apply when |skewness| is above", min_value=0.0, value=float(CONFIG["default_skew_cutoff"]),
                                 step=0.1, key=key("cutoff"))
        robust = st.checkbox("Robust standard errors (HC0)", value=False, key=key("robust"),
                             help="Huber-White heteroskedasticity-consistent standard errors for the full-dataset model. "
                                  "The coefficients do not change; the standard errors, z values, p-values and intervals do.")

        step(4, "Validation")
        test_pct = st.slider("Share of data used for testing (%)", CONFIG["test_pct_min"], CONFIG["test_pct_max"],
                             CONFIG["default_test_pct"], CONFIG["test_pct_step"], key=key("test_pct"))
        st.markdown(f'<div class="lg lg-cap">Training {100 - test_pct}% / test {test_pct}%</div>', unsafe_allow_html=True)
        seed = st.number_input("Random seed", min_value=0, value=int(CONFIG["default_seed"]), step=1, key=key("seed"))
        stratify = st.checkbox("Keep the event share equal in both samples", value=True, key=key("strat"))
        threshold = st.slider("Classification threshold", float(CONFIG["threshold_min"]), float(CONFIG["threshold_max"]),
                              float(CONFIG["default_threshold"]), float(CONFIG["threshold_step"]), key=key("thr"),
                              help="An observation is classified as an event when its predicted probability is at least this value.")
        penalty = st.selectbox("Train/test model: penalty", ["None", "L2 (ridge)"], key=key("penalty"),
                               help="L2 shrinks the coefficients. It works best when the variables are standardized.")
        c_value = float(CONFIG["default_penalty_c"])
        if penalty != "None":
            c_value = st.number_input("Inverse penalty strength C", min_value=0.0001, value=c_value, step=0.1,
                                      format="%.4f", key=key("c"), help="Smaller values shrink the coefficients more.")

    if target is None or not features:
        show(section_html("Choose the variables", "Select the dependent variable and at least one independent variable in step 2."))
        tab_data, = st.tabs(["Data and preparation"])
        with tab_data:
            data_overview(df)
        return

    design = build_design(df, target, positive, features, missing_rule)
    X, y, continuous = design["X"], design["y"], design["continuous"]
    if X.shape[1] == 0:
        st.error("None of the chosen independent variables can be used. " + " ".join(design["notes"]))
        return
    counts = np.bincount(y, minlength=2)
    if len(y) < CONFIG["min_rows"] or counts.min() < CONFIG["min_per_class"]:
        st.error(f"After cleaning, {len(y):,} rows are left, with {counts[1]:,} events and {counts[0]:,} non-events. "
                 f"Each class needs at least {CONFIG['min_per_class']} observations and the data at least {CONFIG['min_rows']} rows.")
        return

    tab_home, tab_data = st.tabs(["Home", "Data and preparation"])

    # The per-variable transformation choices sit on the second tab, but the models need them first,
    # so that part of the tab is drawn before the models are fitted.
    overrides = {}
    with tab_data:
        if continuous:
            show(section_html("Transformation by variable", "Automatic follows the skewness rule in step 3. Choose a transformation to override it."))
            skews = {c: float(X[c].skew()) for c in continuous}
            cols = st.columns(3)
            for i, c in enumerate(continuous):
                with cols[i % 3]:
                    overrides[c] = st.selectbox(md_safe(f"{c} (skewness {skews[c]:+.2f})"), ["Automatic"] + TRANSFORMS,
                                                key=key(f"ovr::{c}"))
    prep_args = (continuous, standardize, method, cutoff, overrides)
    prep_full = Prep(*prep_args).fit(X)
    Xp = prep_full.transform(X)
    bad_full = not np.isfinite(Xp.to_numpy()).all()
    full, full_err, split, split_err = None, None, None, None
    if bad_full:
        full_err = split_err = ("A transformation produced invalid values. Choose a different transformation for the "
                                "skewed variables on the Data and preparation tab.")
    else:
        try:
            full = fit_full_logit(Xp, y, robust, CONFIG["ci_level"])
        except ValueError as err:
            full_err = str(err)
        try:
            split = fit_split_model(X, y, prep_args, test_pct / 100.0, seed, stratify, penalty, c_value)
        except ValueError as err:
            split_err = str(err)

    with tab_data:
        data_overview(df, design, prep_full, Xp)

    with tab_home:
        render_home(df, target, positive, design, prep_full, Xp, full, full_err, split, split_err,
                    robust, threshold, test_pct, seed, stratify, penalty, c_value, key)


def data_overview(df, design=None, prep=None, Xp=None):
    """Data tab: notes on cleaning, skewness before and after, class balance, preview and column list."""
    if design is not None:
        if design["notes"]:
            for note in design["notes"]:
                st.markdown(f'<div class="lg lg-cap">{esc(note)}</div>', unsafe_allow_html=True)
        if prep.plan:
            rows = []
            for c, p in prep.plan.items():
                rows.append([c, fnum(p["skew_before"], 2), p["method"], fnum(p["skew_after"], 2),
                             "yes" if prep.standardize else "no"])
            show(section_html("Skewness before and after", "Skewness of each continuous predictor before and after the transformation (full dataset). Standardizing does not change skewness."))
            show(table_html(["Variable", "Skewness before", "Transformation", "Skewness after", "Standardized"], rows))
        yv = design["y"]
        show(section_html("Outcome balance"))
        show(tiles_html([("Observations used", f"{len(yv):,}", f"of {design['n_raw']:,} rows"),
                         ("Events (1)", f"{int(yv.sum()):,}", f"{yv.mean() * 100:.1f}% of rows"),
                         ("Non-events (0)", f"{int((1 - yv).sum()):,}", f"{(1 - yv.mean()) * 100:.1f}% of rows"),
                         ("Predictors in the model", f"{Xp.shape[1]:,}", "after dummy coding")]))
    show(section_html("Preview", f"First {CONFIG['preview_rows']} rows of the uploaded file."))
    head = df.head(CONFIG["preview_rows"])
    show(table_html(list(head.columns), [[preview_value(v) for v in r] for r in head.itertuples(index=False)]))
    show(section_html("Columns"))
    show(table_html(["Column", "Type", "Missing", "Distinct values"], column_overview(df)))


def render_home(df, target, positive, design, prep_full, Xp, full, full_err, split, split_err,
                robust, threshold, test_pct, seed, stratify, penalty, c_value, key):
    y = design["y"]
    transformed = [c for c, p in prep_full.plan.items() if p["method"] != "None"]
    strip_chips([f"Dependent variable: {target}", f"Event (1): {label_text(positive)}", f"{len(y):,} observations",
                 f"{Xp.shape[1]} predictor" + ("" if Xp.shape[1] == 1 else "s"), "standardized" if prep_full.standardize else "not standardized",
                 (f"{len(transformed)} transformed" if transformed else "no transformation"), SE_LABELS[robust]])
    for note in design["notes"]:
        st.markdown(f'<div class="lg lg-cap">{esc(note)}</div>', unsafe_allow_html=True)

    # Section A: logistic regression on all rows.
    show(section_html("Logistic regression on the full dataset",
                      f"All {len(y):,} observations. Fitted by maximum likelihood with {SE_LABELS[robust]}. "
                      f"OR low and OR high are the {int(round(CONFIG['ci_level'] * 100))}% confidence interval of the odds ratio."))
    if full is None:
        st.error(full_err)
    else:
        for note in full["notes"]:
            st.warning(note)
        m = class_measures(y.to_numpy(), full["prob"], threshold)
        show(tiles_html([("Observations", f"{len(y):,}", f"{y.mean() * 100:.1f}% events"),
                         ("McFadden R²", fnum(full["r2"]), "pseudo R-squared"),
                         ("LR test p-value", fp(full["llr_p"]), f"chi-square {fnum(full['llr'], 1)}"),
                         ("Accuracy", fnum(m["accuracy"]), f"threshold {threshold:.2f}"),
                         ("AUC", fnum(m["auc"]), "in-sample"),
                         ("AIC", fnum(full["aic"], 1), f"BIC {fnum(full['bic'], 1)}")]))
        show(coef_table(full["table"], "Odds ratio"))
        unit = "one standard deviation of the prepared variable" if prep_full.standardize else "one unit of the prepared variable"
        show(f'<div class="lg lg-note">An odds ratio above 1 raises the odds of the event by that factor for {esc(unit)}; below 1 lowers them. '
             f'Variable names show the preparation: log, sqrt, YJ (Yeo-Johnson) and z (standardized).</div>')

    # Section B: training sample and test sample.
    show(section_html(f"Train/test model: {100 - test_pct}% training, {test_pct}% test",
                      "The preparation and the coefficients are learned on the training sample only. The test sample "
                      "shows how the model does on observations it has not seen."))
    if split is None:
        st.error(split_err)
    else:
        mtr = class_measures(split["ytr"], split["ptr"], threshold)
        mte = class_measures(split["yte"], split["pte"], threshold)
        shape = "stratified" if stratify else "random"
        pen = "no penalty" if penalty == "None" else f"L2 penalty, C = {c_value:g}"
        show(f'<div class="lg lg-cap lg-captop">Training {split["n_train"]:,} rows, test {split["n_test"]:,} rows; '
             f'{shape} split, seed {int(seed)}; {pen}.</div>')
        show(tiles_html([("Test accuracy", fnum(mte["accuracy"]), f"training {fnum(mtr['accuracy'])}"),
                         ("Test AUC", fnum(mte["auc"]), f"training {fnum(mtr['auc'])}"),
                         ("Test precision", fnum(mte["precision"]), f"training {fnum(mtr['precision'])}"),
                         ("Test recall", fnum(mte["recall"]), f"training {fnum(mtr['recall'])}")]))
        show('<div class="lg lg-two"><div>' + measures_table(mtr, mte) + confusion_html(mte) + '</div>'
             '<div class="lg-rocbox">' + roc_chart(split["ytr"], split["ptr"], split["yte"], split["pte"], mtr["auc"], mte["auc"])
             + '</div></div>')
        # Both models list the predictors in the same order, so the rows are matched by position. The label
        # of the training model is shown only when the automatic skewness rule chose a different transformation.
        names_full = list(full["table"]["name"]) if full is not None else ["const"] + split["names"]
        names_split = ["const"] + split["names"]
        coef_full = list(full["table"]["coef"]) if full is not None else [float("nan")] * len(names_split)
        coef_split = [split["b0"]] + list(split["b"])
        differ = names_full != names_split
        headers = ["Variable", "Full dataset", "Training sample"] + (["Training sample label"] if differ else [])
        cmp_rows = []
        for a, b_, cf, cs in zip(names_full, names_split, coef_full, coef_split):
            row = ["Constant" if a == "const" else a, fnum(cf, 4), fnum(cs, 4)]
            if differ:
                row.append("" if a == b_ else b_)
            cmp_rows.append(row)
        show(section_html("Coefficients: full dataset and training sample",
                          "Each row is the same variable in both models. Coefficients are on the prepared scale."))
        show(table_html(headers, cmp_rows))
        csv = pd.DataFrame(cmp_rows, columns=headers).to_csv(index=False)
        st.download_button("Download coefficient table (CSV)", csv, file_name="logit_coefficients.csv", mime="text/csv",
                           key=key("dl"))

    # Section C: the sigmoid.
    show(section_html("The sigmoid", "The predicted probability is the sigmoid of the linear score: P(event) = 1 / (1 + e^(-z))."))
    views = {}
    if full is not None:
        views["Full dataset"] = {"Xp": Xp, "y": y.to_numpy(), "group": np.array(["all"] * len(y)), "b0": full["b0"],
                                 "b": full["b"], "means": Xp.mean().to_numpy()}
    if split is not None:
        both = pd.concat([split["Xtr"], split["Xte"]], ignore_index=True)
        views["Train/test model"] = {"Xp": both, "y": np.concatenate([split["ytr"], split["yte"]]),
                                     "group": np.array(["train"] * split["n_train"] + ["test"] * split["n_test"]),
                                     "b0": split["b0"], "b": split["b"], "means": split["Xtr"].mean().to_numpy()}
    if not views:
        st.info("The sigmoid appears when at least one model could be fitted.")
        return
    c1, c2 = st.columns(2)
    with c1:
        which_model = st.radio("Model", list(views), horizontal=True, key=key("sig_model"))
    view = views[which_model]
    with c2:
        which = st.selectbox("Horizontal axis", ["Linear score z (all variables)"] + list(view["Xp"].columns),
                             key=key("sig_x"), format_func=md_safe)
    choice = "score" if which.startswith("Linear score") else which
    if choice != "score" and choice not in view["Xp"].columns:
        choice = "score"
    show(sigmoid_chart(view, choice, threshold, seed))
    what = ("The vertical dashed line marks the score at which the probability equals the threshold."
            if choice == "score" else
            "The horizontal axis shows the variable as the model uses it (after transformation and standardizing), with the "
            "other predictors held at their average. The vertical dashed line marks the value at which the probability "
            "equals the threshold.")
    show(f'<div class="lg lg-note">The horizontal dashed line is the threshold ({threshold:.2f}). {esc(what)} '
         'Points are placed at 0 or 1 according to the actual outcome, with a small vertical spread so that they do not overlap.</div>')


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Version log
# ---------------------------------------------------------------------------
# v1.0 (2026-10-09)  New file. Added: CONFIG (texts, frame and sidebar width, split, threshold, skewness
#   cutoff, limits), styling with a fixed-width centred frame and light/dark colours, upload of CSV and
#   Excel files, choice of the dependent variable and event, choice of independent variables with dummy
#   coding of text columns, missing-value rule, Prep class (log, square root, Yeo-Johnson, standardizing,
#   per-variable override), robust (HC0) standard errors, full-dataset logit (statsmodels), train/test
#   model (scikit-learn, optional L2 penalty, stratified split, preparation learned on training rows),
#   measures (accuracy, precision, recall, F1, AUC, log loss), confusion matrix, ROC chart, sigmoid
#   chart (linear score or one variable), coefficient comparison with CSV download. Charts are drawn
#   twice (wide and narrow) and CSS shows the one that fits the screen.
