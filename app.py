"""
PCOS Detection - Streamlit dashboard (single file)

Run with:   streamlit run app.py

Part 1 is the modelling pipeline (same parameters, seeds, grids and plots as the notebook).
Part 2 is the dashboard UI.
"""
from __future__ import annotations

# =========================================================================== #
#                       PART 1  -  PIPELINE (notebook code)
# =========================================================================== #
import io
import threading
import warnings
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from imblearn.combine import SMOTEENN  # noqa: E402
from imblearn.over_sampling import ADASYN, SMOTE  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    AdaBoostClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import RFE  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

warnings.filterwarnings("ignore")

TARGET = "PCOS (Y/N)"
_PLOT_LOCK = threading.RLock()  # matplotlib's pyplot state is global


# --------------------------------------------------------------------------- #
# Plot plumbing
# --------------------------------------------------------------------------- #
@contextmanager
def plot_ctx(styled: bool):
    """Thread-safe matplotlib context.

    ``styled=True`` reproduces the notebook state *after* ``sns.set(font_scale=0.6)``
    had been executed (cell 37 onwards).  rcParams are restored on exit.
    """
    with _PLOT_LOCK, plt.rc_context():
        if styled:
            sns.set(font_scale=0.6)
        yield


def _to_png(fig, dpi: int = 130) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render(builder, styled: bool) -> bytes:
    """Run a notebook-style plotting function and return the figure as PNG bytes."""
    with plot_ctx(styled):
        return _to_png(builder())


# --------------------------------------------------------------------------- #
# Utility functions (notebook cell 8)
# --------------------------------------------------------------------------- #
def printdf(data) -> pd.DataFrame:
    """for returing readable dataframes"""
    return pd.DataFrame(data)


def plot_missing_vals(data: pd.DataFrame):
    """for plotting missing values"""
    fig = plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    sns.heatmap(data.isnull(), cbar=True, vmin=0, vmax=1)
    plt.title("Missing Values Plot")
    return fig


def remove_outliers(data, feature):
    try:
        Q1 = np.percentile(data[feature], 25, method="midpoint")
        Q3 = np.percentile(data[feature], 75, method="midpoint")
    except TypeError:  # very old NumPy
        Q1 = np.percentile(data[feature], 25, interpolation="midpoint")
        Q3 = np.percentile(data[feature], 75, interpolation="midpoint")
    IQR = Q3 - Q1
    upper = data[feature] >= (Q3 + 1.5 * IQR)
    lower = data[feature] <= (Q1 - 1.5 * IQR)

    return data[~(upper | lower)]


def info_frame(df: pd.DataFrame) -> pd.DataFrame:
    """A tidy version of ``df.info()``."""
    return pd.DataFrame(
        {
            "Column": [str(c).strip() for c in df.columns],
            "Non-Null": df.notna().sum().to_numpy(),
            "Dtype": df.dtypes.astype(str).to_numpy(),
        }
    )


# --------------------------------------------------------------------------- #
# Stage 1 - loading, cleaning, EDA, preprocessing, resampling
# --------------------------------------------------------------------------- #
# Canonical column names (exactly as the notebook sees them after cleaning)
CANONICAL_COLUMNS = [
    "Sl. No", "Patient File No.", "PCOS (Y/N)", "Age (yrs)", "Weight (Kg)", "Height(Cm)", "BMI",
    "Blood Group", "Pulse rate(bpm)", "RR (breaths/min)", "Hb(g/dl)", "Cycle(R/I)", "Cycle length(days)",
    "Marraige Status (Yrs)", "Pregnant(Y/N)", "No. of aborptions", "I   beta-HCG(mIU/mL)",
    "II    beta-HCG(mIU/mL)", "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH", "Hip(inch)", "Waist(inch)",
    "Waist:Hip Ratio", "TSH (mIU/L)", "AMH(ng/mL)", "PRL(ng/mL)", "Vit D3 (ng/mL)", "PRG(ng/mL)",
    "RBS(mg/dl)", "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)",
    "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)", "BP _Systolic (mmHg)", "BP _Diastolic (mmHg)",
    "Follicle No. (L)", "Follicle No. (R)", "Avg. F size (L) (mm)", "Avg. F size (R) (mm)", "Endometrium (mm)",
]
_IDENTIFIERS = {"Sl. No", "Patient File No."}


def _key(name) -> str:
    """Loose comparison key: lower-case, no whitespace, no underscores."""
    return "".join(str(name).lower().split()).replace("_", "")


_ALIASES = {  # spelling variants seen across versions of the dataset file
    _key("Marriage Status (Yrs)"): "Marraige Status (Yrs)",
    _key("Marraige Status(Yrs)"): "Marraige Status (Yrs)",
    _key("No. of abortions"): "No. of aborptions",
    _key("PCOS(Y/N)"): "PCOS (Y/N)",
}
_CANON_BY_KEY = {_key(c): c for c in CANONICAL_COLUMNS} | _ALIASES


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns that match a notebook column (ignoring spaces / case / typos)."""
    import difflib

    mapping, used = {}, set()
    for c in df.columns:  # 1) exact match on the loose key, or a known alias
        canon = _CANON_BY_KEY.get(_key(c))
        if canon is not None and canon not in used:
            mapping[c] = canon
            used.add(canon)
    keys = {_key(c): c for c in CANONICAL_COLUMNS}
    for c in df.columns:  # 2) near-identical spellings (e.g. 'Marriage' vs 'Marraige')
        if c in mapping:
            continue
        close = difflib.get_close_matches(_key(c), list(keys), n=1, cutoff=0.88)
        if close and keys[close[0]] not in used:
            mapping[c] = keys[close[0]]
            used.add(keys[close[0]])
    return df.rename(columns=mapping)


# Columns the pipeline refers to by name.  Everything else is handled generically.
REQUIRED_COLUMNS = [
    "PCOS (Y/N)", "Weight (Kg)", "Hip(inch)", "Waist:Hip Ratio",
    "Pulse rate(bpm)", "BP _Systolic (mmHg)", "BP _Diastolic (mmHg)",
]


def validate_columns(df: pd.DataFrame) -> None:
    import difflib

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        found = [str(c).strip() for c in df.columns]
        hints = []
        for m in missing:
            close = difflib.get_close_matches(m, found, n=1, cutoff=0.6)
            hints.append(f"{m!r}" + (f" (closest in your file: {close[0]!r})" if close else ""))
        raise ValueError(
            "The file does not look like the PCOS workbook. Missing columns: " + "; ".join(hints)
            + f". Columns found ({len(found)}): " + ", ".join(found)
        )


def read_dataset(source) -> pd.DataFrame:
    """Cell 11: ``pd.read_excel(..., sheet_name=1)``.

    The notebook reads the second sheet.  To be forgiving we try sheet 1 first and then the other
    sheets, and use the first one that actually contains the target column.  Every sheet is parsed
    defensively (a text-only "description" sheet must never break loading).  Column names are then
    matched loosely (see ``standardize_columns``).
    """
    xl = pd.ExcelFile(source)
    names = xl.sheet_names
    order = ([1] if len(names) > 1 else []) + [i for i in range(len(names)) if i != 1]
    chosen = None
    for i in order:
        try:
            head = xl.parse(names[i], nrows=2)
        except Exception:  # noqa: BLE001
            continue
        if any(_key(c) == _key(TARGET) for c in head.columns):
            chosen = i
            break
    if chosen is None:
        raise ValueError(f"No sheet in the workbook contains a '{TARGET}' column (sheets found: {names}).")
    df = standardize_columns(xl.parse(names[chosen]))
    validate_columns(df)
    return df


def prepare_data(raw: pd.DataFrame) -> dict:
    """Runs the notebook's data pipeline; if anything fails the message says at which step."""
    stage = ["reading the file"]
    try:
        return _prepare_data(raw, stage)
    except ValueError as e:
        raise ValueError(f"{e}  [step: {stage[0]}]") from e
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{type(e).__name__}: {e}  [step: {stage[0]}]") from e


def _mode0(series: pd.Series):
    m = series.mode()
    return m.iloc[0] if len(m) else 0.0


def _prepare_data(raw: pd.DataFrame, stage: list) -> dict:
    out: dict = {}
    data = raw.copy()
    out["missing_optional"] = [
        c for c in CANONICAL_COLUMNS if c not in raw.columns and c not in _IDENTIFIERS and c not in REQUIRED_COLUMNS
    ]

    # ---- cell 14: remove the unnamed column --------------------------------
    stage[0] = 'cell 14: remove the unnamed column'
    data.drop(
        data.columns[data.columns.str.contains("Unnamed", case=False)],
        axis=1,
        inplace=True,
    )
    out["after_unnamed"] = data.copy()
    out["dtypes_before"] = info_frame(data)

    # ---- cell 19: drop Sl. No and Patient File No. (the first two columns) --
    stage[0] = 'cell 19: drop Sl. No and Patient File No. (the first two columns)'
    ids = [c for c in data.columns if c in _IDENTIFIERS]
    for c in list(data.columns[:3]):  # a leading 1..N counter column (e.g. an exported index) is also an identifier
        if c not in ids and c not in CANONICAL_COLUMNS and pd.api.types.is_integer_dtype(data[c]) \
                and data[c].notna().all() and data[c].nunique() == len(data):
            ids.append(c)
    data.drop(ids, axis=1, inplace=True)

    # ---- cell 21: object -> numeric ----------------------------------------
    stage[0] = 'cell 21: object -> numeric'
    converted = []
    for feature in [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])]:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        converted.append(str(feature).strip())
    data = data.rename(columns=lambda x: x.strip())
    out["converted_columns"] = converted
    # Tidy-up that is a no-op on the standard file: blank rows, unreadable (all-empty) columns, rows without a label.
    n_before = len(data)
    data = data.dropna(how="all")
    empty_cols = [c for c in data.columns if data[c].isna().all()]
    data = data.drop(columns=empty_cols)
    for c in REQUIRED_COLUMNS:
        if c not in data.columns:
            raise ValueError(f"Column {c!r} has no readable numeric values in this file.")
    data = data.dropna(subset=[TARGET])
    if len(data) != n_before:
        data = data.reset_index(drop=True)
    out["tidy_removed_rows"] = n_before - len(data)
    out["tidy_empty_columns"] = [str(c).strip() for c in empty_cols]
    out["dtypes_after"] = info_frame(data)

    # ---- cells 24 / 26: missing values -------------------------------------
    stage[0] = 'cells 24 / 26: missing values'
    nulls = data.isnull().sum()
    out["null_counts"] = nulls[nulls > 0]
    out["missing_before_png"] = render(lambda: plot_missing_vals(data), styled=False)

    fills, skipped = [], []
    for col, how in [
        ("Marraige Status (Yrs)", "mean"),
        ("Fast food (Y/N)", "mode"),
        ("II    beta-HCG(mIU/mL)", "mean"),
        ("AMH(ng/mL)", "mean"),
    ]:
        if col not in data.columns:  # column absent in this version of the file -> nothing to fill
            skipped.append(col.strip())
            continue
        value = data[col].mean() if how == "mean" else _mode0(data[col])
        n_missing = int(data[col].isnull().sum())
        data[col] = data[col].fillna(value)
        fills.append({"Column": col, "Strategy": how, "Fill value": float(value), "Rows filled": n_missing})
    out["fill_skipped"] = skipped
    # Safeguard (no-op on the standard file): anything still missing, e.g. a renamed column, gets the
    # same treatment as the notebook uses - mean for continuous columns, mode for binary ones.
    for col in data.columns[data.isnull().any()]:
        vals = data[col].dropna().unique()
        binary = len(vals) > 0 and set(vals) <= {0, 1}
        value = _mode0(data[col]) if binary else data[col].mean()
        n_missing = int(data[col].isnull().sum())
        data[col] = data[col].fillna(value)
        fills.append({"Column": col, "Strategy": "mode (safeguard)" if binary else "mean (safeguard)",
                      "Fill value": float(value), "Rows filled": n_missing})
    out["fills"] = pd.DataFrame(fills, columns=["Column", "Strategy", "Fill value", "Rows filled"])
    out["missing_after_png"] = render(lambda: plot_missing_vals(data), styled=False)

    out["describe"] = data.describe()
    out["data_filled"] = data.copy()

    # ---- cells 29 / 30: target, features, binary features ------------------
    stage[0] = 'cells 29 / 30: target, features, binary features'
    target = TARGET
    all_features = list(data.columns[1:])
    features_df = data[all_features]
    binary_features = features_df.columns[(features_df.max() == 1) & (features_df.min() == 0)]
    out["binary_features"] = list(binary_features)
    out["class_counts_raw"] = {int(k): int(v) for k, v in data[target].value_counts().items()}

    # ---- cells 35 / 37: correlation matrix + heatmap -----------------------
    stage[0] = 'cells 35 / 37: correlation matrix + heatmap'
    corr_matrix = data.corr().abs()
    out["corr_with_target"] = corr_matrix[target].sort_values()

    def _corr_heatmap():
        fig = plt.figure(figsize=(10, 8))
        sns.set(font_scale=0.6)
        indexes = corr_matrix.nlargest(17, target)[target].index
        cm = pd.DataFrame(np.corrcoef(data[indexes].values.T))
        sns.heatmap(
            data=cm,
            annot=True,
            yticklabels=indexes.values,
            xticklabels=indexes.values,
            cbar=True,
            cmap="YlGnBu",
            square=True,
        )
        plt.title(f"correlation heatmap with {target}")
        return fig

    out["corr_heatmap_png"] = render(_corr_heatmap, styled=False)  # sns.set happens inside, as in the notebook

    # ---- cell 39: highly-correlated feature pairs --------------------------
    stage[0] = 'cell 39: highly-correlated feature pairs'
    correlation_threshold = 0.8  # threshold = 80%
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated_pairs = []
    for column in upper_triangle.columns:
        correlated_columns = upper_triangle.index[upper_triangle[column] >= correlation_threshold].tolist()
        for correlated_column in correlated_columns:
            highly_correlated_pairs.append((column, correlated_column))
    out["correlation_threshold"] = correlation_threshold
    out["highly_correlated_pairs"] = highly_correlated_pairs

    # ---- cell 41: drop redundant features ----------------------------------
    stage[0] = 'cell 41: drop redundant features'
    data.drop(["Weight (Kg)"], axis=1, inplace=True)
    data.drop(["Hip(inch)"], axis=1, inplace=True)

    # ---- cell 43: split binary / non-binary --------------------------------
    stage[0] = 'cell 43: split binary / non-binary'
    all_features = list(data.columns[1:])
    features_df = data[all_features]
    binary_features = features_df.columns[(features_df.max() == 1) & (features_df.min() == 0)]
    nonbinary_data = features_df.drop(binary_features, axis=1)
    out["pre_outlier"] = data.copy()
    out["nonbinary"] = nonbinary_data.columns.to_list()

    # ---- cell 45: selective outlier removal --------------------------------
    stage[0] = 'cell 45: selective outlier removal'
    old_shape = data.shape
    features_to_filter = [
        "BP _Systolic (mmHg)",
        "Pulse rate(bpm)",
        "Waist:Hip Ratio",
        "BP _Systolic (mmHg)",
        "BP _Diastolic (mmHg)",
    ]
    log = []
    for feature in features_to_filter:
        # Exactly the notebook's rule - unless that rule would delete most of the data (this happens when a column is
        # nearly constant: IQR = 0 makes every row equal to the quartile count as an "outlier").  Then it is skipped.
        before = len(data)
        note = "removed outliers"
        if before == 0 or data[feature].notna().sum() == 0:
            note = "skipped (no data)"
        else:
            candidate = remove_outliers(data, feature=feature)
            if len(candidate) < 0.5 * before:
                note = f"skipped (rule would remove {before - len(candidate)} of {before} rows)"
            else:
                data = candidate
        log.append({"Feature filtered": feature, "Rows remaining": int(data[feature].shape[0]), "Status": note})
    out["outlier_old_shape"] = old_shape
    out["outlier_log"] = pd.DataFrame(log)
    out["data"] = data  # post-outlier data (used for box/strip + violin plots)

    # ---- cell 47: MinMax scaling -------------------------------------------
    stage[0] = 'cell 47: MinMax scaling'
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data)
    scaled_data.columns = data.columns
    out["scaled_data"] = scaled_data
    out["all_features"] = all_features
    out["class_counts_filtered"] = {int(k): int(v) for k, v in data[target].value_counts().items()}

    # ---- cells 52-54: resampling (ADASYN, SMOTE, SMOTE-ENN) ----------------
    stage[0] = 'cells 52-54: resampling (ADASYN, SMOTE, SMOTE-ENN)'
    X = scaled_data.drop("PCOS (Y/N)", axis=1)
    y = scaled_data["PCOS (Y/N)"]

    def _counts(s):
        return {int(k): int(v) for k, v in s.value_counts().sort_index().items()}

    resampling = {}
    stage[0] = "resampling"
    _cls = y.value_counts()
    if len(_cls) < 2 or int(_cls.min()) < 12 or len(y) < 40:
        raise ValueError(f"Not enough data after cleaning for resampling (rows: {len(y)}, per-class counts: "
                         f"{ {int(k): int(v) for k, v in _cls.items()} }).")

    # ADASYN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
    resampling["ADASYN"] = (_counts(y_train), _counts(y_resampled))

    # SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    resampling["SMOTE"] = (_counts(y_train), _counts(y_resampled))

    # SMOTE-ENN  (the one carried forward in the notebook)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    resampling["SMOTE-ENN"] = (_counts(y_train), _counts(y_resampled))

    out["resampling"] = resampling
    out["X_resampled"] = X_resampled
    out["y_resampled"] = y_resampled
    out["n_train"] = len(X_train)
    out["n_test"] = len(X_test)
    return out


# --------------------------------------------------------------------------- #
# Per-feature plots (cells 32, 43, 48, 57, 58) - code is unchanged
# --------------------------------------------------------------------------- #
def feature_plot(kind: str, prep: dict, feature: str) -> bytes:
    target = TARGET

    if kind == "count":  # cell 32  (before sns.set -> default style)

        def build():
            fig = plt.figure(figsize=(5, 5))
            sns.countplot(data=prep["data_filled"], x=feature, hue=target)
            plt.xlabel(feature)
            plt.ylabel("Count")
            plt.legend()
            plt.title(f"Countplot for {feature}")
            return fig

        return render(build, styled=False)

    if kind == "box_raw":  # cell 43

        def build():
            fig = plt.figure(figsize=(5, 3))
            sns.boxplot(data=prep["pre_outlier"], x=target, y=feature, hue=target)
            plt.xlabel(target)
            plt.ylabel(feature)
            plt.title(f"Boxplot for {feature}")
            return fig

    elif kind == "box_scaled":  # cell 48

        def build():
            fig = plt.figure(figsize=(5, 3))
            sns.boxplot(data=prep["scaled_data"], x=target, y=feature, hue=target, palette="colorblind")
            plt.xlabel(target)
            plt.ylabel(feature)
            plt.title(f"Boxplot for {feature}")
            return fig

    elif kind == "box_strip":  # cell 57

        def build():
            fig = plt.figure(figsize=(5, 3))
            data = prep["data"]
            sns.boxplot(data=data, x=target, y=feature, hue=target)
            sns.stripplot(data=data, y=feature, x=target, jitter=True, size=3, hue=target)
            plt.xlabel(target)
            plt.ylabel(feature)
            plt.title(f"Boxplot and Stripplot for {feature}")
            return fig

    elif kind == "violin":  # cell 58

        def build():
            fig = plt.figure(figsize=(5, 3))
            data = prep["data"]
            sns.violinplot(data=data, x=target, y=feature, hue=target)
            sns.stripplot(data=data, y=feature, x=target, jitter=True, size=3, hue=target, palette="pastel")
            plt.xlabel(target)
            plt.ylabel(feature)
            plt.title(f"Violinplot and Stripplot for {feature}")
            return fig

    else:
        raise ValueError(kind)

    return render(build, styled=True)  # all of these ran after sns.set(font_scale=0.6)


# --------------------------------------------------------------------------- #
# Stage 2 - split + feature selection (cells 62-70)
# --------------------------------------------------------------------------- #
def run_feature_selection(prep: dict) -> dict:
    scaled_data = prep["scaled_data"]
    all_features = prep["all_features"]
    X_resampled, y_resampled = prep["X_resampled"], prep["y_resampled"]

    # cell 62: splitting the data  (no seed in the notebook -> none here)
    features_df = scaled_data[all_features]
    x_train_rfe, x_test_rfe, y_train_rfe, y_test_rfe = train_test_split(
        X_resampled, y_resampled, stratify=y_resampled, test_size=0.20
    )

    # cell 64: Feature selection using RFE
    model_lr = LogisticRegression(solver="liblinear", random_state=0)
    model_rfc = RandomForestClassifier()
    rfe_lr = RFE(model_lr, n_features_to_select=15)
    rfe_rfc = RFE(model_rfc, n_features_to_select=15)

    # Logistic regression
    x_train_rfe_lr = rfe_lr.fit_transform(x_train_rfe, y_train_rfe)
    features_lr_indices = [i for i, value in enumerate(rfe_lr.support_) if value]
    features_lr = features_df.columns[features_lr_indices]
    x_train_rfe_lr = printdf(x_train_rfe_lr)
    x_train_rfe_lr.columns = features_lr

    def _rank_lr():
        fig = plt.figure()
        sns.barplot(x=rfe_lr.ranking_, y=features_df.columns)
        plt.title("15 Logistic Regression - Feature Ranking")
        plt.ylabel("feature")
        return fig

    rank_lr_png = render(_rank_lr, styled=True)

    # random forest classifier
    x_train_rfe_rfc = rfe_rfc.fit_transform(x_train_rfe, y_train_rfe)
    features_rfc_indices = [i for i, value in enumerate(rfe_rfc.support_) if value]
    features_rfc = features_df.columns[features_rfc_indices]
    x_train_rfe_rfc = printdf(x_train_rfe_rfc)
    x_train_rfe_rfc.columns = features_rfc

    def _rank_rfc():
        fig = plt.figure()
        sns.barplot(x=rfe_rfc.ranking_, y=features_df.columns)
        plt.title("Random Forest Classifier - Feature Ranking")
        plt.ylabel("feature")
        return fig

    rank_rfc_png = render(_rank_rfc, styled=True)

    # cell 66: Feature contribution scores - Logistic Regression
    lr_model = rfe_lr.estimator_
    lr_model.fit(x_train_rfe_lr, y_train_rfe)
    lr_scores = lr_model.coef_[0]

    def _score_lr():
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=lr_scores, y=features_lr)
        plt.title("Logistic Regression - Feature Scores")
        plt.xlabel("Score")
        plt.ylabel("Feature")
        return fig

    score_lr_png = render(_score_lr, styled=True)

    # cell 68: Feature contribution scores - Random Forest
    rfc_model = rfe_rfc.estimator_
    rfc_model.fit(x_train_rfe_rfc, y_train_rfe)
    rfc_scores = rfc_model.feature_importances_

    def _score_rfc():
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=rfc_scores, y=features_rfc)
        plt.title("Random Forest Classifier - Feature Scores")
        plt.xlabel("Score")
        plt.ylabel("Feature")
        return fig

    score_rfc_png = render(_score_rfc, styled=True)

    # cells 69 / 70: quick accuracy check of both selectors
    model_lr.fit(x_train_rfe_lr, y_train_rfe)
    model_rfc.fit(x_train_rfe_rfc, y_train_rfe)
    y_pred_lr = model_lr.predict(x_test_rfe[features_lr])
    y_pred_rfc = model_rfc.predict(x_test_rfe[features_rfc])
    acc_lr = accuracy_score(y_test_rfe, y_pred_lr)
    acc_rfc = accuracy_score(y_test_rfe, y_pred_rfc)

    return {
        "features_lr": list(features_lr),
        "features_rfc": list(features_rfc),
        "rank_lr_png": rank_lr_png,
        "rank_rfc_png": rank_rfc_png,
        "score_lr_png": score_lr_png,
        "score_rfc_png": score_rfc_png,
        "lr_scores": dict(zip(features_lr, map(float, lr_scores))),
        "rfc_scores": dict(zip(features_rfc, map(float, rfc_scores))),
        "acc_lr": float(acc_lr),
        "acc_rfc": float(acc_rfc),
        "shape_lr": x_train_rfe_lr.shape,
        "shape_rfc": x_train_rfe_rfc.shape,
        "n_train": len(x_train_rfe),
        "n_test": len(x_test_rfe),
        # carried into the modelling stage
        "_x_train_rfe_lr": x_train_rfe_lr,
        "_x_train_rfe_rfc": x_train_rfe_rfc,
        "_y_train_rfe": y_train_rfe,
        "_x_test_rfe": x_test_rfe,
        "_y_test_rfe": y_test_rfe,
    }


# --------------------------------------------------------------------------- #
# Stage 3 - model stacking, optimisation and evaluation (cells 73-90)
# --------------------------------------------------------------------------- #
def getModels() -> dict:
    """returns the models required for use"""
    models = {
        "rfc": RandomForestClassifier(),
        "knn": KNeighborsClassifier(),
        # 'svm': SVC(), #  --- toggled off
        # 'lr': LogisticRegression(), #  --- toggled off
        "adb": AdaBoostClassifier(),
        # 'nb':GaussianNB(), #  --- toggled off
        "xgb": XGBClassifier(),
    }
    return models


def getBaseModels(models: dict) -> list:
    base_models = []
    for name, model in models.items():
        base_models.append((name, models[name]))
    return base_models


def getStackedML(models: dict, meta_model=RandomForestClassifier()) -> StackingClassifier:
    base_models = getBaseModels(models)
    model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    return model


# cell 76 - parameter grid (unchanged)
param_grid = {
    "rfc": {  # random forest classifier
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2"],
        "n_jobs": [-1],
    },
    "knn": {  # k nearest neighbours classifier
        "n_neighbors": [5, 10, 15],
        "p": [1, 2],
        # 'weights': ['uniform', 'distance'],
        "n_jobs": [-1],
    },
    "svm": {  # support vector machine classifier
        "C": [0.1, 1, 10],
        "kernel": ["poly", "sigmoid", "rbf"],
        "degree": [3, 4],
        "coef0": [0.0, 1.0],
        "random_state": [42],
    },
    "lr": {  # logistic regressor
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
    },
    "adb": {  # ada boost classifier
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.5],
        "random_state": [42],
    },
    "nb": {},  # gaussian naive bayes classifier
    "xgb": {  # extreme gradient boost classifer
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15],
        "learning_rate": [0.1, 0.01, 0.001],
        "random_state": [42],
    },
    "stacked": {  # stacking classifier
        "cv": [3, 5, 7],
    },
}


def getBestModels(param_grid: dict, training_features, training_target) -> dict:
    """GridSearchCV (5-fold) over every model toggled on in getModels()."""
    models = getModels()
    best_models = {}
    for name, model in models.items():
        if param_grid[name] != {}:
            grid_search = GridSearchCV(model, param_grid[name], cv=5, n_jobs=-1)
            grid_search.fit(training_features, training_target)
            best_model = grid_search.best_estimator_
            best_models[name] = best_model
    return best_models


def calcPerfMetrics(models: dict, training_features, training_target, testing_features, testing_target) -> dict:
    """Same maths as the notebook; returns results instead of printing them."""
    results = {}
    with plot_ctx(True):  # runs after sns.set(font_scale=0.6) in the notebook
        fig = plt.figure()
        for name, model in models.items():
            model.fit(training_features, training_target)
            predictions = model.predict(testing_features)
            rep_text = classification_report(testing_target, predictions)
            rep_dict = classification_report(testing_target, predictions, output_dict=True)
            cm = confusion_matrix(testing_target, predictions)
            fpr, tpr, thresholds = roc_curve(testing_target, predictions)
            plt.plot(fpr, tpr, label=name)
            auc = roc_auc_score(testing_target, predictions)
            results[name] = {
                "report_text": rep_text,
                "report": pd.DataFrame(rep_dict).T,
                "cm": cm,
                "auc": float(auc),
                "accuracy": float(rep_dict["accuracy"]),
                "precision": float(rep_dict["1.0"]["precision"]),
                "recall": float(rep_dict["1.0"]["recall"]),
                "f1": float(rep_dict["1.0"]["f1-score"]),
                "support": int(len(testing_target)),
            }
        plt.legend()
        roc_png = _to_png(fig)
    return results, roc_png


def _best_params(models: dict) -> dict:
    out = {}
    for name, model in models.items():
        keys = param_grid[name].keys()
        p = model.get_params()
        out[name] = {k: p[k] for k in keys if k in p}
    return out


def run_modeling(fs: dict, progress=None) -> dict:
    say = progress or (lambda msg: None)
    x_train_rfe_lr = fs["_x_train_rfe_lr"]
    x_train_rfe_rfc = fs["_x_train_rfe_rfc"]
    y_train_rfe = fs["_y_train_rfe"]
    x_test_rfe = fs["_x_test_rfe"]
    y_test_rfe = fs["_y_test_rfe"]
    features_lr = fs["features_lr"]
    features_rfc = fs["features_rfc"]

    say("Grid-searching RFC, KNN, AdaBoost and XGBoost on the **Logistic-Regression** feature set...")
    best_models_lr = getBestModels(param_grid, x_train_rfe_lr, y_train_rfe)
    say("Grid-searching the same models on the **Random-Forest** feature set...")
    best_models_rfc = getBestModels(param_grid, x_train_rfe_rfc, y_train_rfe)
    params_lr, params_rfc = _best_params(best_models_lr), _best_params(best_models_rfc)

    say("Stacking and evaluating (Logistic-Regression features)...")
    stacked_model = getStackedML(best_models_lr)  # getting the stacked model
    best_models_lr["stacked"] = stacked_model  # adding the stacked model for iterative analysis
    res_lr, roc_lr = calcPerfMetrics(
        best_models_lr, x_train_rfe_lr, y_train_rfe, x_test_rfe[features_lr], y_test_rfe
    )

    say("Stacking and evaluating (Random-Forest features)...")
    stacked_model_rfc = getStackedML(best_models_rfc)  # noqa: F841  (built but unused in the notebook, too)
    # NOTE: exactly as in notebook cell 90, the stacked model added here is `stacked_model`
    # (the one built from the Logistic-Regression-tuned base models), not `stacked_model_rfc`.
    best_models_rfc["stacked"] = stacked_model
    res_rfc, roc_rfc = calcPerfMetrics(
        best_models_rfc, x_train_rfe_rfc, y_train_rfe, x_test_rfe[features_rfc], y_test_rfe
    )

    return {
        "lr": {"best_params": params_lr, "results": res_lr, "roc_png": roc_lr},
        "rfc": {"best_params": params_rfc, "results": res_rfc, "roc_png": roc_rfc},
    }

# =========================================================================== #
#                               PART 2  -  THE APP (UI)
# =========================================================================== #
import hashlib  # noqa: E402
import html  # noqa: E402
import tempfile  # noqa: E402
import traceback  # noqa: E402
from io import BytesIO  # noqa: E402
from pathlib import Path  # noqa: E402

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="PCOS Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    ROOT = Path(__file__).parent
except NameError:  # pragma: no cover
    ROOT = Path.cwd()
DEFAULT_DATA = ROOT / "data" / "PCOS_data_without_infertility.xlsx"

# --------------------------------------------------------------------------- #
# Styling - dark dashboard, mint accent
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{ --bg:#0c110f; --panel:#141a17; --panel2:#1a211e; --line:#252e2a; --txt:#e8f0eb; --mut:#8a9a92; --acc:#7ee2a8; --acc2:#4cc98a; --warn:#f2b84b; }
html, body, .stApp { font-family:'Inter',sans-serif; }
.stApp { background: var(--bg); }
.block-container { padding-top:1.4rem; padding-bottom:4rem; max-width:1440px; }
header[data-testid="stHeader"] { background: transparent; }
h1,h2,h3,h4 { letter-spacing:-0.015em; color:var(--txt); }

/* sidebar */
[data-testid="stSidebar"] { background:#0f1512; border-right:1px solid var(--line); }
[data-testid="stSidebarNavLink"] { border-radius:12px; padding:.55rem .8rem; margin:2px 0; color:var(--mut); }
[data-testid="stSidebarNavLink"]:hover { background:rgba(126,226,168,.07); color:var(--txt); }
[data-testid="stSidebarNavLink"][aria-current="page"] { background:rgba(126,226,168,.13); color:var(--acc); font-weight:600; }
[data-testid="stSidebarNavSeparator"], [data-testid="stNavSectionHeader"] { color:#5d6d65; }
.side-note { font-size:.76rem; color:var(--mut); line-height:1.5; margin-top:.9rem; }

/* page header */
.ph { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap; margin:.2rem 0 1.3rem 0; }
.ph .k { color:var(--acc); font-size:.74rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.ph .t { font-size:2rem; font-weight:800; color:var(--txt); line-height:1.15; margin:.15rem 0 .3rem 0; }
.ph .s { color:var(--mut); font-size:.97rem; max-width:52rem; line-height:1.55; }
.ph .r { display:flex; gap:.5rem; flex-wrap:wrap; }
.pill { background:var(--panel); border:1px solid var(--line); color:var(--mut); border-radius:999px; padding:.35rem .85rem; font-size:.8rem; font-weight:500; white-space:nowrap; }
.pill.on { color:var(--acc); border-color:rgba(126,226,168,.35); background:rgba(126,226,168,.08); }

/* stat cards */
.stat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr)); gap:14px; margin:.2rem 0 1.3rem 0; }
.stat { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:1.05rem 1.2rem; }
.stat-top { display:flex; justify-content:space-between; align-items:center; }
.stat .v { font-size:2rem; font-weight:800; color:var(--txt); line-height:1.1; }
.stat .ic { width:26px; height:26px; stroke:var(--acc); fill:none; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }
.stat .l { color:var(--txt); font-weight:600; font-size:.93rem; margin-top:.5rem; }
.stat .s { color:var(--mut); font-size:.8rem; margin-top:.15rem; line-height:1.35; }

/* cards */
.card { background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:1.25rem 1.45rem; margin-bottom:1rem; }
.card h4 { margin:0 0 .55rem 0; font-size:1.02rem; color:var(--txt); font-weight:700; }
.card p, .card li { color:#b7c5bd; line-height:1.65; font-size:.93rem; }
.card ul { margin:.3rem 0 0 1.1rem; padding:0; }
.card .row { display:flex; justify-content:space-between; padding:.5rem 0; border-bottom:1px solid var(--line); font-size:.88rem; color:var(--mut); }
.card .row:last-child { border-bottom:none; }
.card .row b { color:var(--txt); font-weight:600; }

.callout { border-left:3px solid var(--acc); background:rgba(126,226,168,.07); border-radius:10px; padding:.85rem 1.05rem; color:#cfe9db; margin:.6rem 0 1rem 0; font-size:.91rem; line-height:1.55; }
.callout.warn { border-color:var(--warn); background:rgba(242,184,75,.08); color:#f1deb4; }
.callout code { background:rgba(255,255,255,.08); color:inherit; }

.sec { display:flex; align-items:center; gap:.6rem; margin:1.6rem 0 .5rem 0; }
.sec .dot { width:8px; height:24px; border-radius:6px; background:var(--acc); }
.sec h3 { margin:0; padding:0; font-size:1.2rem; font-weight:700; }
.sec-sub { color:var(--mut); margin:-.15rem 0 .8rem 1.05rem; font-size:.9rem; }

.steps { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin-bottom:1rem; }
.step { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:.95rem 1.05rem; }
.step .n { width:28px; height:28px; border-radius:50%; background:var(--acc); color:#07110c; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:.85rem; margin-bottom:.55rem; }
.step b { color:var(--txt); } .step span { display:block; color:var(--mut); font-size:.82rem; margin-top:.2rem; line-height:1.4; }

.trk { display:flex; gap:.75rem; align-items:flex-start; padding:.6rem 0; border-bottom:1px solid var(--line); }
.trk:last-child { border-bottom:none; }
.trk .d { width:12px; height:12px; border-radius:50%; border:2px solid #3a4742; margin-top:.3rem; flex:none; }
.trk.done .d { background:var(--acc); border-color:var(--acc); box-shadow:0 0 0 4px rgba(126,226,168,.15); }
.trk b { color:var(--txt); font-size:.9rem; display:block; } .trk i { color:var(--mut); font-size:.78rem; font-style:normal; }

.chip { display:inline-block; padding:.26rem .72rem; margin:.17rem .25rem .17rem 0; border-radius:999px; font-size:.8rem; font-weight:500;
  background:rgba(126,226,168,.1); color:#9be9bd; border:1px solid rgba(126,226,168,.25); }
.chip.alt { background:rgba(148,163,184,.1); color:#cbd5e1; border-color:rgba(148,163,184,.25); }

.flow { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin:.6rem 0 1rem 0; }
.flow .node { background:var(--panel2); border:1px solid var(--line); border-radius:12px; padding:.55rem .9rem; font-weight:600; color:var(--txt); font-size:.88rem; }
.flow .node.meta { background:var(--acc); color:#07110c; border:none; }
.flow .arrow { color:var(--mut); font-size:1.1rem; }

[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:.85rem 1.1rem; }
[data-testid="stMetricLabel"] p { color:var(--mut); font-weight:600; }
[data-testid="stImage"] img { border-radius:14px; border:1px solid var(--line); background:#fff; padding:6px; }
.stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding:.55rem 1rem; font-weight:600; }
[data-testid="stExpander"] { background:var(--panel); border:1px solid var(--line); border-radius:14px; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] { background:var(--acc); color:#07110c; border:none; font-weight:700; border-radius:12px; }
.stButton > button { border-radius:12px; }
.footer { text-align:center; color:#5d6d65; font-size:.8rem; margin-top:3rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ICONS = {
    "db": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M7 15l4-5 3 3 5-7"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12l3 3 5-6"/>',
    "scale": '<path d="M12 3v18M6 21h12M5 7h14"/><path d="M5 7l-3 7a3 3 0 006 0zM19 7l-3 7a3 3 0 006 0z"/>',
    "flask": '<path d="M9 3h6M10 3v6l-5 9a2 2 0 002 3h10a2 2 0 002-3l-5-9V3"/>',
    "pulse": '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
}


def _logo() -> str:
    p = Path(tempfile.gettempdir()) / "pcos_detection_logo.svg"
    if not p.exists():
        p.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="210" height="40" viewBox="0 0 210 40">'
            '<rect x="2" y="4" width="32" height="32" rx="9" fill="#7ee2a8"/>'
            '<path d="M8 21h5l3-8 5 15 3-9 2 2h5" fill="none" stroke="#07110c" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" transform="translate(-1 -1)"/>'
            '<text x="44" y="26" font-family="Inter,Arial,sans-serif" font-size="17" font-weight="700" fill="#e8f0eb">PCOS Detection</text></svg>'
        )
    return str(p)


try:
    st.logo(_logo(), size="large")
except Exception:  # noqa: BLE001  (older Streamlit without st.logo)
    pass


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def H(markup: str) -> None:
    st.markdown(markup.strip(), unsafe_allow_html=True)


def page_header(kicker: str, title: str, subtitle: str) -> None:
    loaded = get_file_bytes() is not None
    trained = bool(st.session_state.get("trained"))
    pills = (
        f'<span class="pill {"on" if loaded else ""}">● Dataset {"loaded" if loaded else "not loaded"}</span>'
        f'<span class="pill {"on" if trained else ""}">● Models {"trained" if trained else "not trained"}</span>'
    )
    H(f'<div class="ph"><div><div class="k">{html.escape(kicker)}</div><div class="t">{html.escape(title)}</div>'
      f'<div class="s">{subtitle}</div></div><div class="r">{pills}</div></div>')


def section(title: str, sub: str | None = None) -> None:
    H(f'<div class="sec"><div class="dot"></div><h3>{html.escape(title)}</h3></div>')
    if sub:
        H(f'<div class="sec-sub">{sub}</div>')


def stat_cards(items: list[tuple]) -> None:
    cards = ""
    for value, label, sub, icon in items:
        svg = f'<svg class="ic" viewBox="0 0 24 24">{ICONS.get(icon, ICONS["db"])}</svg>'
        cards += (f'<div class="stat"><div class="stat-top"><div class="v">{html.escape(str(value))}</div>{svg}</div>'
                  f'<div class="l">{html.escape(label)}</div><div class="s">{sub}</div></div>')
    H(f'<div class="stat-grid">{cards}</div>')


def callout(text: str, kind: str = "") -> None:
    H(f'<div class="callout {kind}">{text}</div>')


def chips(items, kind: str = "") -> None:
    H("".join(f'<span class="chip {kind}">{html.escape(str(i))}</span>' for i in items))


def show_img(data: bytes, caption: str | None = None) -> None:
    st.image(data, caption=caption, width="stretch")


def table(df, **kw) -> None:
    st.dataframe(df, width="stretch", hide_index=kw.pop("hide_index", True), **kw)


def fmt_counts(d: dict) -> str:
    return " · ".join(f"class {k}: {v}" for k, v in d.items())


# --------------------------------------------------------------------------- #
# Data access (cached)
# --------------------------------------------------------------------------- #
def get_file_bytes() -> bytes | None:
    up = st.session_state.get("uploaded_bytes")
    if up:
        return up
    if DEFAULT_DATA.exists():
        return DEFAULT_DATA.read_bytes()
    return None


@st.cache_data(show_spinner="Cleaning and preparing the dataset…")
def get_prep(file_bytes: bytes) -> dict:
    return prepare_data(read_dataset(BytesIO(file_bytes)))


@st.cache_data(show_spinner=False, max_entries=600)
def get_feature_png(kind: str, feature: str, key: str, _prep: dict) -> bytes:
    return feature_plot(kind, _prep, feature)


@st.cache_resource(show_spinner=False)
def get_fs(key: str, _prep: dict) -> dict:
    return run_feature_selection(_prep)


@st.cache_resource(show_spinner=False)
def get_models(key: str, _fs: dict, _progress=None) -> dict:
    return run_modeling(_fs, _progress)


def require_prep() -> tuple[dict, str]:
    fb = get_file_bytes()
    if fb is None:
        page_header("Dataset needed", "Upload the PCOS dataset to continue",
                    "This page runs the real pipeline on your data. Upload <b>PCOS_data_without_infertility.xlsx</b> in the sidebar, "
                    "or place it in a <code>data/</code> folder next to <code>app.py</code>.")
        st.stop()
    try:
        return get_prep(fb), hashlib.md5(fb).hexdigest()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not process the file: {e}")
        st.caption("Expected the PCOS workbook 'PCOS_data_without_infertility.xlsx' (the sheet with the 'PCOS (Y/N)' column).")
        with st.expander("Technical details (send this if you need help)"):
            st.code(traceback.format_exc())
        st.stop()


def feature_gallery(kind: str, prep: dict, key: str, features: list[str], cols: int = 3) -> None:
    mode = st.radio("View", ["All features (grid)", "Single feature"], horizontal=True, key=f"{kind}_mode",
                    label_visibility="collapsed")
    if mode == "Single feature":
        f = st.selectbox("Feature", features, key=f"{kind}_sel")
        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            show_img(get_feature_png(kind, f, key, prep))
        return
    with st.spinner("Rendering plots…"):
        grid = st.columns(cols)
        for i, f in enumerate(features):
            with grid[i % cols]:
                show_img(get_feature_png(kind, f, key, prep))


# --------------------------------------------------------------------------- #
# Static content
# --------------------------------------------------------------------------- #
FEATURE_DICT = {
    "Vitals & demographics": [
        ("Age (yrs)", "Age of patient in years"),
        ("Blood Group", "A+ = 11, A- = 12, B+ = 13, B- = 14, O+ = 15, O- = 16, AB+ = 17, AB- = 18"),
        ("Pulse rate(bpm)", "Heart rate in beats per minute (adult resting range 60–100)"),
        ("RR (breaths/min)", "Respiration rate (adult resting range 12–16)"),
        ("Hb(g/dl)", "Hemoglobin level (normal for women 12.3–15.3 g/dL)"),
        ("BP _Systolic (mmHg)", "Pressure in the arteries when the heart beats"),
        ("BP _Diastolic (mmHg)", "Pressure in the arteries when the heart rests between beats"),
    ],
    "Body measurements": [
        ("Weight (Kg)", "Weight of patient in kg"),
        ("Height(Cm)", "Height of patient in centimetres"),
        ("BMI", "Body mass index"),
        ("Hip(inch)", "Hip size in inches"),
        ("Waist(inch)", "Waist size in inches"),
        ("Waist:Hip Ratio", "Waist divided by hip"),
    ],
    "Menstrual & reproductive": [
        ("Cycle(R/I)", "Cycle type, Regular / Irregular (numerically coded)"),
        ("Cycle length(days)", "Length of menstrual cycle; the average is about 28 days"),
        ("Marraige Status (Yrs)", "Years of marriage"),
        ("Pregnant(Y/N)", "Whether the patient is pregnant"),
        ("No. of aborptions", "Number of abortions, if any (437 of 541 patients had none)"),
    ],
    "Hormones & lab tests": [
        ("I beta-HCG(mIU/mL)", "Beta-hCG, first measurement"),
        ("II beta-HCG(mIU/mL)", "Beta-hCG, second measurement (< 5 negative, > 25 positive for pregnancy)"),
        ("FSH(mIU/mL)", "Follicle-stimulating hormone"),
        ("LH(mIU/mL)", "Luteinizing hormone"),
        ("FSH/LH", "Ratio of FSH to LH"),
        ("TSH (mIU/L)", "Thyroid-stimulating hormone (normal 0.4–4.0)"),
        ("AMH(ng/mL)", "Anti-Müllerian hormone"),
        ("PRL(ng/mL)", "Prolactin level"),
        ("Vit D3 (ng/mL)", "Vitamin D level (normal ≥ 20 ng/mL)"),
        ("PRG(ng/mL)", "Progesterone level"),
        ("RBS(mg/dl)", "Random blood sugar test"),
    ],
    "Symptoms & lifestyle": [
        ("Weight gain(Y/N)", "Has there been weight gain"),
        ("hair growth(Y/N)", "Has there been excess hair growth"),
        ("Skin darkening (Y/N)", "Skin darkening issues"),
        ("Hair loss(Y/N)", "Hair loss issues"),
        ("Pimples(Y/N)", "Pimple issues"),
        ("Fast food (Y/N)", "Is fast food part of the diet"),
        ("Reg.Exercise(Y/N)", "Regular exercise"),
    ],
    "Ultrasound": [
        ("Follicle No. (L)", "Number of follicles, left side"),
        ("Follicle No. (R)", "Number of follicles, right side"),
        ("Avg. F size (L) (mm)", "Average follicle size, left, in mm"),
        ("Avg. F size (R) (mm)", "Average follicle size, right, in mm"),
        ("Endometrium (mm)", "Endometrium thickness in mm"),
    ],
}

# Figures printed in the notebook (rounded as shown there); used only as a reference.
NOTEBOOK_REF = {
    "Logistic-Regression features": {"rfc": (0.99, 0.9848), "knn": (1.00, 1.0), "adb": (0.96, 0.9562), "xgb": (0.97, 0.9697)},
    "Random-Forest features": {"rfc": (0.97, 0.9697), "knn": (1.00, 1.0), "adb": (0.96, 0.9562), "xgb": (0.94, 0.9394)},
}
MODEL_LABEL = {"rfc": "Random Forest", "knn": "K-Nearest Neighbours", "adb": "AdaBoost", "xgb": "XGBoost", "stacked": "Stacked ensemble"}

PIPELINE_STEPS = [
    ("Clean", "Drop ID columns, fix dtypes, fill missing values"),
    ("Explore", "Class balance, correlations, redundant features"),
    ("Preprocess", "Selective outlier removal and MinMax scaling"),
    ("Resample", "ADASYN, SMOTE and SMOTE-ENN to fix imbalance"),
    ("Select features", "RFE with Logistic Regression and Random Forest"),
    ("Stack & tune", "RFC + KNN + AdaBoost + XGBoost, GridSearchCV"),
    ("Evaluate", "Accuracy, precision, recall, F1, ROC / AUC"),
]


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    page_header("Dashboard", "PCOS Detection Model",
                "A classification pipeline for the early, low-cost detection of Polycystic Ovary Syndrome from clinical, "
                "hormonal and lifestyle data, built to help close the gap of undiagnosed cases.")
    stat_cards([
        ("8–13%", "of reproductive-age women", "are affected by PCOS (WHO)", "pulse"),
        ("70%", "of cases undiagnosed", "a leading cause of infertility", "target"),
        ("52%", "prevalence in Pakistani women", "vs. 20–25% in the UK white population", "chart"),
        ("541", "clinical records", "43 parameters from 10 hospitals", "db"),
    ])

    main, side = st.columns([3, 1.25], gap="medium")
    with main:
        H("""<div class="card"><h4>What is PCOS?</h4>
<p>Polycystic Ovary Syndrome is a common hormonal condition that usually starts in adolescence. It can cause hormonal imbalance,
irregular periods, excess androgen levels and cysts in the ovaries. Irregular periods, usually with a lack of ovulation, can make it
difficult to become pregnant.</p></div>
<div class="card"><h4>Project aim</h4>
<p>To address the undiagnosed-case problem, this project builds a classification model that aims to improve diagnostic accuracy while making
it more cost-effective, using blood tests, menstrual-cycle data, lifestyle factors, age and more.</p></div>""")
        section("The pipeline", "Each step has its own page in the sidebar and runs the real notebook code.")
        H('<div class="steps">' + "".join(
            f'<div class="step"><div class="n">{i}</div><b>{t}</b><span>{d}</span></div>'
            for i, (t, d) in enumerate(PIPELINE_STEPS, 1)) + "</div>")
        callout("<b>Educational project.</b> This app showcases a research notebook. It is not a medical device and must not be used for diagnosis.", "warn")

    with side:
        loaded = get_file_bytes() is not None
        trained = bool(st.session_state.get("trained"))
        rows = ""
        for i, (t, d) in enumerate(PIPELINE_STEPS):
            done = trained if i >= 5 else loaded
            rows += f'<div class="trk {"done" if done else ""}"><span class="d"></span><div><b>{t}</b><i>{d}</i></div></div>'
        H(f'<div class="card"><h4>Pipeline progress</h4>{rows}</div>')

        if loaded:
            try:
                prep = get_prep(get_file_bytes())
                cc = prep["class_counts_raw"]
                total = sum(cc.values()) or 1
                H(f'<div class="card"><h4>Dataset snapshot</h4>'
                  f'<div class="row"><span>Records</span><b>{prep["after_unnamed"].shape[0]}</b></div>'
                  f'<div class="row"><span>Features</span><b>{prep["after_unnamed"].shape[1] - 3}</b></div>'
                  f'<div class="row"><span>PCOS positive</span><b>{cc.get(1, 0)} ({cc.get(1, 0) / total:.0%})</b></div>'
                  f'<div class="row"><span>PCOS negative</span><b>{cc.get(0, 0)} ({cc.get(0, 0) / total:.0%})</b></div></div>')
            except Exception:  # noqa: BLE001
                callout("The dataset could not be read. Open any pipeline page to see the details.", "warn")
        else:
            callout("No dataset yet. Upload it from the sidebar to unlock the pipeline pages.")


def page_dataset() -> None:
    page_header("Step 0", "The dataset",
                "A clinical PCOS dataset with 43 physical and clinical parameters on 541 fertile female patients, "
                "collected across 10 hospitals in Kerala, India.")
    stat_cards([("541", "samples", "patient records", "db"), ("43", "parameters", "1 target + 42 columns", "list"),
                ("10", "hospitals", "across Kerala, India", "layers"), ("PCOS", "target (Y/N)", "does the patient have PCOS?", "target")])

    t1, t2, t3 = st.tabs(["Feature dictionary", "Data preview", "Summary statistics"])
    with t1:
        for group, rows in FEATURE_DICT.items():
            with st.expander(f"{group}  ·  {len(rows)} features", expanded=group == "Hormones & lab tests"):
                table(pd.DataFrame(rows, columns=["Feature", "Description"]))
        callout("<b>Identifiers</b> (<i>Sl. No</i>, <i>Patient File No.</i>) are dropped during cleaning, as they only enumerate records.")
    prep = None
    if get_file_bytes():
        try:
            prep = get_prep(get_file_bytes())
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not process the file: {e}")
    with t2:
        if prep is None:
            callout("Upload the dataset in the sidebar to preview it here.", "warn")
        else:
            st.caption(f"First rows of the data sheet (after removing the empty unnamed column) · "
                       f"{prep['after_unnamed'].shape[0]} rows × {prep['after_unnamed'].shape[1]} columns")
            preview = prep["after_unnamed"].head(25).copy()
            for c in preview.columns:  # the raw sheet mixes strings into numeric columns
                if not pd.api.types.is_numeric_dtype(preview[c]):
                    preview[c] = preview[c].astype(str)
            table(preview, hide_index=False)
    with t3:
        if prep is None:
            callout("Upload the dataset in the sidebar to see summary statistics.", "warn")
        else:
            table(prep["describe"].T.reset_index().rename(columns={"index": "Feature"}))


def page_cleaning() -> None:
    prep, _ = require_prep()
    page_header("Step 1", "Data cleaning & type normalisation",
                "Remove non-informative columns, convert numeric strings to numbers and fill the few missing values, "
                "keeping every record because information integrity matters in medicine.")
    if prep.get("missing_optional"):
        callout("This version of the file has no column named: <b>" + ", ".join(html.escape(c.strip()) for c in prep["missing_optional"])
                + "</b>. It was skipped; everything else ran as normal.", "warn")

    if prep.get("tidy_removed_rows") or prep.get("tidy_empty_columns"):
        callout(f"Tidy-up: removed <b>{prep['tidy_removed_rows']}</b> blank / unlabelled rows and "
                f"<b>{len(prep['tidy_empty_columns'])}</b> empty columns "
                + (f"({html.escape(', '.join(prep['tidy_empty_columns']))})" if prep["tidy_empty_columns"] else "") + ".", "warn")

    section("1 · Drop non-informative columns")
    stat_cards([
        (f"{prep['after_unnamed'].shape[1]}", "columns after dropping 'Unnamed'", "an empty column with no relation to the target", "layers"),
        (f"{prep['after_unnamed'].shape[1] - 2}", "columns after dropping IDs", "<i>Sl. No</i> and <i>Patient File No.</i> removed", "list"),
    ])

    section("2 · Convert numeric strings to numbers",
            "These columns were numeric <i>strings</i>, so they were converted with <code>pd.to_numeric</code> rather than encoded.")
    if prep["converted_columns"]:
        chips(prep["converted_columns"])
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Before")
        table(prep["dtypes_before"], height=320)
    with c2:
        st.caption("After")
        table(prep["dtypes_after"], height=320)

    section("3 · Fill the missing values",
            "Means for continuous columns and the mode for the binary one, instead of dropping records.")
    t1, t2 = st.tabs(["Before filling", "After filling"])
    with t1:
        show_img(prep["missing_before_png"])
    with t2:
        show_img(prep["missing_after_png"])
    table(prep["fills"].style.format({"Fill value": "{:.4f}"}))


def page_eda() -> None:
    prep, key = require_prep()
    page_header("Step 2", "Exploratory data analysis",
                "Understand class balance, spot which features move with PCOS and find redundant, highly correlated features.")
    cc = prep["class_counts_raw"]
    total = sum(cc.values())
    stat_cards([(total, "records", "after cleaning", "db"),
                (cc.get(0, 0), "PCOS negative", f"{cc.get(0, 0) / total:.1%} of records", "check"),
                (cc.get(1, 0), "PCOS positive", f"{cc.get(1, 0) / total:.1%} of records", "target")])

    t1, t2, t3 = st.tabs(["Binary features", "Correlation with target", "Redundant features"])
    with t1:
        st.caption("Count plots show how balanced each binary feature is across PCOS-positive and PCOS-negative patients.")
        feature_gallery("count", prep, key, prep["binary_features"], cols=4)
    with t2:
        st.caption("Absolute correlation of every feature with the target, and the heatmap of the 17 most correlated features.")
        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            s = prep["corr_with_target"].drop(TARGET).sort_values(ascending=False)
            table(s.rename("Correlation").reset_index().rename(columns={"index": "Feature"}),
                  column_config={"Correlation": st.column_config.ProgressColumn("Correlation", min_value=0, max_value=1, format="%.3f")},
                  height=620)
        with c2:
            show_img(prep["corr_heatmap_png"])
    with t3:
        thr = prep["correlation_threshold"]
        st.caption(f"Feature pairs with mutual correlation of at least {thr:.0%}.")
        table(pd.DataFrame(prep["highly_correlated_pairs"], columns=["Feature A", "Feature B"]))
        callout("<b>Dropped:</b> <code>Weight (Kg)</code> (its information is already in <code>BMI</code>) and <code>Hip(inch)</code> "
                "(<code>Waist</code> is more readily available). <code>FSH</code> and <code>FSH/LH</code> were <b>kept</b>: both are closely tied to reproductive "
                "function and medically significant for PCOS diagnosis.")


def page_preprocess() -> None:
    prep, key = require_prep()
    page_header("Step 3", "Preprocessing",
                "Selective outlier removal, then MinMax scaling to [0, 1] so distance-based models like KNN are not distorted.")
    t1, t2, t3, t4 = st.tabs(["Outliers", "Outlier removal", "Scaling", "Distributions"])
    nb = prep["nonbinary"]
    with t1:
        st.caption("Box plots of every continuous feature, split by PCOS status, before any outlier removal.")
        feature_gallery("box_raw", prep, key, nb)
    with t2:
        callout("<b>Caution:</b> medical outliers can be clinically significant, and blanket removal can introduce bias. Outliers were removed only from features "
                "where their distribution is even (symmetric) across PCOS-positive and PCOS-negative records.", "warn")
        o = prep["outlier_old_shape"]
        stat_cards([(f"{o[0]} × {o[1]}", "shape before", "rows × columns", "layers"),
                    (f"{prep['data'].shape[0]} × {prep['data'].shape[1]}", "shape after", "rows × columns", "check")])
        table(prep["outlier_log"])
        skipped = prep["outlier_log"][prep["outlier_log"]["Status"].str.startswith("skipped")]
        if len(skipped):
            callout("Some features were <b>not</b> filtered: in this file the outlier rule would have removed most of the rows "
                    "(typically because most patients share the same value). The rest of the pipeline ran normally.", "warn")
    with t3:
        st.caption("Box plots after outlier removal and MinMax scaling.")
        feature_gallery("box_scaled", prep, key, nb)
    with t4:
        kind = st.segmented_control("Plot type", ["Box + strip", "Violin + strip"], default="Box + strip")
        feature_gallery("box_strip" if kind != "Violin + strip" else "violin", prep, key, nb)


def page_resampling() -> None:
    prep, _ = require_prep()
    page_header("Step 4", "Class balancing",
                "PCOS-positive cases are the minority class, which biases models. Three resampling methods were applied to the <b>training split only</b> "
                "(80/20 split, <code>random_state=42</code>) to avoid leakage.")
    cc = prep["class_counts_filtered"]
    stat_cards([(cc.get(0, 0), "PCOS negative", "after outlier removal", "check"), (cc.get(1, 0), "PCOS positive", "after outlier removal", "target"),
                (prep["n_train"], "training rows", "before resampling", "layers"), (prep["n_test"], "test rows", "held out", "scale")])

    cols = st.columns(3, gap="medium")
    desc = {"ADASYN": "Adaptive Synthetic Sampling", "SMOTE": "Synthetic Minority Over-sampling (strategy 1.0)",
            "SMOTE-ENN": "SMOTE + Edited Nearest Neighbours"}
    for col, (name, (before, after)) in zip(cols, prep["resampling"].items()):
        with col:
            chosen = name == "SMOTE-ENN"
            H(f'<div class="card"><h4>{name} {"✅" if chosen else ""}</h4><p style="margin:0">{desc[name]}</p></div>')
            m1, m2 = st.columns(2)
            m1.metric("Original", f"{sum(before.values())}", fmt_counts(before), delta_color="off", delta_arrow="off")
            m2.metric("Resampled", f"{sum(after.values())}", fmt_counts(after), delta_color="off", delta_arrow="off")
    callout("<b>SMOTE-ENN</b> is carried forward. It oversamples the minority class and then removes noisy samples with Edited Nearest Neighbours, "
            "which was judged more medically appropriate after testing and literature review.")


def page_features() -> None:
    prep, key = require_prep()
    page_header("Step 5", "Feature selection",
                "Wrapper-style <b>Recursive Feature Elimination (RFE)</b> picks the 15 most informative features, once with Logistic Regression "
                "and once with a Random Forest. The resampled data is split 80/20 again before selection.")
    with st.spinner("Running Recursive Feature Elimination…"):
        fs = get_fs(key, prep)

    stat_cards([(f"{fs['acc_lr']:.4f}", "Logistic Regression accuracy", "on RFE-selected features", "chart"),
                (f"{fs['acc_rfc']:.4f}", "Random Forest accuracy", "on RFE-selected features", "chart"),
                (fs["n_train"], "training rows", "after resampling", "layers"), (fs["n_test"], "test rows", "held out", "scale")])
    callout("The notebook does not seed this second split or the Random Forest, so figures vary slightly between runs. "
            "Results are cached for the session, so they stay consistent while you explore.", "warn")

    t1, t2, t3 = st.tabs(["Logistic Regression", "Random Forest", "Compare"])
    for tab, tag, name in [(t1, "lr", "Logistic Regression"), (t2, "rfc", "Random Forest")]:
        with tab:
            st.markdown(f"**15 features selected by {name}**")
            chips(fs[f"features_{tag}"], "" if tag == "lr" else "alt")
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.caption("Feature ranking (shorter bar = higher rank)")
                show_img(fs[f"rank_{tag}_png"])
            with c2:
                st.caption("Contribution scores of the selected features")
                show_img(fs[f"score_{tag}_png"])
    with t3:
        a, b = set(fs["features_lr"]), set(fs["features_rfc"])
        rows = [(f, f in a, f in b) for f in sorted(a | b)]
        df = pd.DataFrame(rows, columns=["Feature", "Logistic Regression", "Random Forest"])
        df["Chosen by both"] = df["Logistic Regression"] & df["Random Forest"]
        st.markdown(f"**{len(a & b)}** features were chosen by both selectors.")
        table(df.sort_values(["Chosen by both", "Feature"], ascending=[False, True]))


def page_modeling() -> None:
    prep, key = require_prep()
    page_header("Step 6", "Model stacking & optimisation",
                "Four base learners are tuned with 5-fold <code>GridSearchCV</code> and combined by a stacking classifier "
                "with a Random Forest as the meta-model.")
    H('<div class="flow"><span class="node">Random Forest</span><span class="node">KNN</span><span class="node">AdaBoost</span>'
      '<span class="node">XGBoost</span><span class="arrow">➜</span><span class="node meta">Stacking · meta-model: Random Forest</span>'
      '<span class="arrow">➜</span><span class="node">PCOS / No PCOS</span></div>')

    with st.expander("Hyper-parameter grids searched", expanded=False):
        cols = st.columns(2)
        for i, name in enumerate(["rfc", "knn", "adb", "xgb"]):
            with cols[i % 2]:
                st.markdown(f"**{MODEL_LABEL[name]}**")
                table(pd.DataFrame([(k, ", ".join(map(str, v))) for k, v in param_grid[name].items()], columns=["Parameter", "Values"]))

    fs = get_fs(key, prep)
    st.session_state.setdefault("trained", False)
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Train & evaluate models", type="primary", icon=":material/play_arrow:", width="stretch"):
            st.session_state["trained"] = True
    with c2:
        st.caption("Runs two grid searches and the stacked models. This can take several minutes the first time; afterwards it is cached.")

    if not st.session_state["trained"]:
        callout("Press <b>Train &amp; evaluate models</b> to run the search. Results then appear here and on the <b>Results</b> page.")
        return
    with st.status("Training…", expanded=True) as status:
        mod = get_models(key, fs, _progress=lambda m: st.write(m))
        status.update(label="Models trained", state="complete", expanded=False)

    section("Best parameters found")
    t1, t2 = st.tabs(["Logistic-Regression features", "Random-Forest features"])
    for tab, tag in [(t1, "lr"), (t2, "rfc")]:
        with tab:
            rows = [(MODEL_LABEL[m], ", ".join(f"{k}={v}" for k, v in p.items())) for m, p in mod[tag]["best_params"].items()]
            table(pd.DataFrame(rows, columns=["Model", "Best parameters"]))
    callout("Head to the <b>Results</b> page for metrics, confusion matrices and ROC curves.")


def _result_block(tag: str, res: dict, roc_png: bytes) -> None:
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        st.caption("ROC curves, all models")
        show_img(roc_png)
    with c2:
        name = st.segmented_control("Model", list(res), default=list(res)[0], format_func=lambda m: MODEL_LABEL[m],
                                    key=f"seg_{tag}") or list(res)[0]
        r = res[name]
        m = st.columns(5)
        m[0].metric("Accuracy", f"{r['accuracy']:.4f}")
        m[1].metric("Precision", f"{r['precision']:.4f}")
        m[2].metric("Recall", f"{r['recall']:.4f}")
        m[3].metric("F1-score", f"{r['f1']:.4f}")
        m[4].metric("AUC", f"{r['auc']:.4f}")
        st.caption("Precision, recall and F1 are for the PCOS-positive class.")
        cm = pd.DataFrame(r["cm"], index=["Actual: No PCOS", "Actual: PCOS"], columns=["Predicted: No PCOS", "Predicted: PCOS"])
        st.markdown("**Confusion matrix**")
        st.dataframe(cm.style.background_gradient(cmap="Greens", vmin=0), width="stretch")
        st.markdown("**Classification report**")
        rep = r["report"].rename(index={"0.0": "No PCOS (0)", "1.0": "PCOS (1)"})
        st.dataframe(rep.style.format("{:.3f}", subset=["precision", "recall", "f1-score"]).format("{:.0f}", subset=["support"]), width="stretch")
        with st.expander("Raw report as printed by the notebook"):
            st.code(r["report_text"])


def page_results() -> None:
    prep, key = require_prep()
    page_header("Step 7", "Results & conclusion", "How every model performed on the held-out test set, for both feature-selection strategies.")
    if not st.session_state.get("trained"):
        callout("No models trained yet. Run the training step first.", "warn")
        st.page_link(PG_MODEL, label="Go to Model Training", icon=":material/arrow_forward:")
        st.stop()
    fs = get_fs(key, prep)
    with st.spinner("Loading trained models…"):
        mod = get_models(key, fs)

    best = max(((t, m, r) for t in ("lr", "rfc") for m, r in mod[t]["results"].items()), key=lambda x: (x[2]["accuracy"], x[2]["auc"]))
    stat_cards([(f"{best[2]['accuracy']:.2%}", "best accuracy", f"{MODEL_LABEL[best[1]]} · {'Logistic-Regression' if best[0] == 'lr' else 'Random-Forest'} features", "target"),
                (f"{best[2]['auc']:.4f}", "best AUC", "on the held-out test set", "chart"),
                (fs["n_test"], "test samples", "never seen during training", "check")])

    t1, t2, t3, t4 = st.tabs(["Logistic-Regression features", "Random-Forest features", "Comparison", "Conclusion"])
    with t1:
        _result_block("lr", mod["lr"]["results"], mod["lr"]["roc_png"])
    with t2:
        _result_block("rfc", mod["rfc"]["results"], mod["rfc"]["roc_png"])
    with t3:
        rows = []
        for tag, label in [("lr", "Logistic Regression"), ("rfc", "Random Forest")]:
            for m, r in mod[tag]["results"].items():
                rows.append({"Features selected by": label, "Model": MODEL_LABEL[m], "Accuracy": r["accuracy"], "Precision": r["precision"],
                             "Recall": r["recall"], "F1": r["f1"], "AUC": r["auc"]})
        df = pd.DataFrame(rows)
        metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        table(df.style.format({c: "{:.4f}" for c in metrics}).background_gradient(subset=metrics, cmap="Greens", vmin=0.5, vmax=1.15))
        with st.expander("Figures reported in the original notebook run"):
            ref = [{"Features selected by": k, "Model": MODEL_LABEL[m], "Accuracy": a, "AUC": u} for k, d in NOTEBOOK_REF.items() for m, (a, u) in d.items()]
            table(pd.DataFrame(ref))
            st.caption("The notebook's second split and several models are unseeded, so a fresh run can differ slightly from these.")
    with t4:
        H("""<div class="card"><h4>Conclusion</h4>
<p>The work met its objective of a high-precision, cost-effective model for the early detection of PCOS that supports medical professionals without
relying on more invasive methods.</p>
<ul>
<li>Recursive Feature Elimination with Logistic Regression and a Random Forest, plus selective outlier removal and scaling, improved efficiency and precision.</li>
<li>Removing outliers from the <b>entire</b> dataset indiscriminately hurt performance, likely through loss of crucial information.</li>
<li>In the notebook, KNN reached 1.0 on accuracy, precision, recall, F1 and AUC, whichever selector was used.</li>
<li>Models generally did better on features chosen by Logistic Regression than by the Random Forest.</li>
</ul></div>""")
        callout("<b>Interpretation note:</b> near-perfect scores on a single small test split should be read with care; they are not evidence of clinical readiness.", "warn")


# --------------------------------------------------------------------------- #
# Navigation + sidebar
# --------------------------------------------------------------------------- #
PG_HOME = st.Page(page_overview, title="Dashboard", icon=":material/dashboard:", default=True)
PG_DATA = st.Page(page_dataset, title="Dataset", icon=":material/database:", url_path="dataset")
PG_CLEAN = st.Page(page_cleaning, title="Data Cleaning", icon=":material/cleaning_services:", url_path="cleaning")
PG_EDA = st.Page(page_eda, title="Exploration", icon=":material/query_stats:", url_path="eda")
PG_PRE = st.Page(page_preprocess, title="Preprocessing", icon=":material/tune:", url_path="preprocessing")
PG_RES = st.Page(page_resampling, title="Class Balancing", icon=":material/balance:", url_path="balancing")
PG_FEAT = st.Page(page_features, title="Feature Selection", icon=":material/checklist:", url_path="features")
PG_MODEL = st.Page(page_modeling, title="Model Training", icon=":material/hub:", url_path="training")
PG_RESULTS = st.Page(page_results, title="Results", icon=":material/emoji_events:", url_path="results")

nav = st.navigation({"Project": [PG_HOME, PG_DATA], "Pipeline": [PG_CLEAN, PG_EDA, PG_PRE, PG_RES, PG_FEAT, PG_MODEL], "Outcome": [PG_RESULTS]})

with st.sidebar:
    st.divider()
    st.markdown("**Dataset**")
    up = st.file_uploader("Upload PCOS_data_without_infertility.xlsx", type=["xlsx"], label_visibility="collapsed")
    if up is not None:
        st.session_state["uploaded_bytes"] = up.getvalue()
    elif "uploaded_bytes" in st.session_state:
        st.session_state.pop("uploaded_bytes")
    src = "uploaded file" if st.session_state.get("uploaded_bytes") else ("data/ folder" if DEFAULT_DATA.exists() else None)
    if src:
        st.success(f"Using {src}", icon=":material/check_circle:")
    else:
        st.warning("No dataset found yet", icon=":material/info:")
    H('<div class="side-note">Educational project · not a medical device.</div>')

nav.run()
H('<div class="footer">PCOS Detection · educational machine-learning showcase</div>')
