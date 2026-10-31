"""Model comparison, feature ranking and SVM evaluation for binary tabular data.

The input is a pandas DataFrame containing numerical features and a binary
outcome column.
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from minepy import MINE
from skrebate import ReliefF
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

SEED = 42
MODEL_NAMES = ("KNN", "Naive Bayes", "LR", "SVM", "XGBoost", "RF", "Stacking", "MLP", "CNN")
RANKING_METHODS = ("MIC", "Relief", "RF", "Chi2")


def split_xy(frame, target="Label"):
    if target not in frame:
        raise ValueError(f"Missing target column: {target}")
    X = frame.drop(columns=[target]).copy()
    y = frame[target].to_numpy()
    if not X.columns.is_unique or X.shape[1] == 0:
        raise ValueError("Feature names must be unique and nonempty")
    if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite features must be handled before analysis")
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError("Target must contain both 0 and 1")
    return X.astype(float), np.asarray(y, dtype=int)


class ConvNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool1d(2, ceil_mode=True), nn.Flatten(),
            nn.Linear(32 * ((n_features + 1) // 2), 1),
        )

    def forward(self, x):
        return self.network(x)


class CNNClassifier(ClassifierMixin, BaseEstimator):
    """A small one-dimensional CNN with a scikit-learn-compatible API."""

    def __init__(self, epochs=10, batch_size=16, lr=0.001, random_state=SEED):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or set(np.unique(y)) != {0, 1}:
            raise ValueError("CNN requires a 2D matrix and both binary classes")
        torch.manual_seed(self.random_state)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.model_ = ConvNet(self.n_features_in_)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.random_state)
        dataset = TensorDataset(torch.from_numpy(X).unsqueeze(1), torch.tensor(y, dtype=torch.float32).view(-1, 1))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(batch_x), batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, "model_")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected CNN feature shape")
        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model_(torch.from_numpy(X).unsqueeze(1))).numpy().ravel()
        return np.column_stack((1 - p, p))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_models(random_state=SEED, include_cnn=True):
    """The nine classifiers used for model comparison."""
    models = {
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "LR": LogisticRegression(max_iter=2000, random_state=random_state),
        "SVM": SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=random_state),
        "XGBoost": XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=random_state, n_jobs=1),
        "RF": RandomForestClassifier(random_state=random_state, n_jobs=1),
        "MLP": MLPClassifier(random_state=random_state, max_iter=300),
    }
    if include_cnn:
        models["CNN"] = CNNClassifier(random_state=random_state)
    # Stacking includes the same base learner families, with a logistic meta-model.
    base = [(name, clone(model)) for name, model in models.items()]
    models["Stacking"] = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=2000, random_state=random_state),
        cv=3, n_jobs=1,
    )
    return {name: models[name] for name in MODEL_NAMES if name in models}


def _positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X)).ravel()
    raise TypeError("A continuous decision score is required to calculate AUC")


def compare_models(X, y, n_splits=3, random_state=SEED, include_cnn=True):
    """Stratified CV comparison using continuous scores; returns fold and summary tables."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []
    for name, estimator in make_models(random_state, include_cnn).items():
        for fold, (tr, va) in enumerate(cv.split(X, y), 1):
            model = clone(estimator)
            model.fit(X[tr], y[tr])
            score = _positive_scores(model, X[va])
            results.append({"Model": name, "Fold": fold,
                            "AUC": roc_auc_score(y[va], score),
                            "AUPRC": average_precision_score(y[va], score)})
    fold_results = pd.DataFrame(results)
    summary = (fold_results.groupby("Model", as_index=False)
               .agg(AUC=("AUC", "mean"), AUC_SD=("AUC", "std"),
                    AUPRC=("AUPRC", "mean"), AUPRC_SD=("AUPRC", "std"))
               .sort_values("AUC", ascending=False).reset_index(drop=True))
    return summary, fold_results


def rank_features(X, y, methods=RANKING_METHODS, random_state=SEED):
    """Four feature rankings. Equal scores follow pandas' original default sort order."""
    X = pd.DataFrame(X).copy()
    y = np.asarray(y, dtype=int)
    if len(X) != len(y):
        raise ValueError("Feature and label sample counts differ")
    if not set(methods).issubset(RANKING_METHODS):
        raise ValueError("Unsupported ranking method")
    rankings = {}
    if "MIC" in methods:
        scores = []
        for col in X.columns:
            mine = MINE(alpha=0.6, c=15)
            mine.compute_score(X[col].to_numpy(dtype=float), y)
            scores.append(mine.mic())
        rankings["MIC"] = scores
    if "Relief" in methods:
        model = ReliefF(n_neighbors=10, n_features_to_select=X.shape[1])
        model.fit(X.to_numpy(dtype=float), y)
        rankings["Relief"] = model.feature_importances_
    if "RF" in methods:
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X, y)
        rankings["RF"] = model.feature_importances_
    if "Chi2" in methods:
        shifted = X - X.min(axis=0)
        scores, _ = chi2(shifted, y)
        rankings["Chi2"] = np.nan_to_num(scores, nan=0.0)
    ordered = pd.DataFrame(index=range(X.shape[1]))
    for method in methods:
        ranking = pd.DataFrame({"Feature": X.columns, "Score": rankings[method]})
        ranking = ranking.sort_values("Score", ascending=False).reset_index(drop=True)
        ordered[method] = ranking["Feature"]
    return ordered


def incremental_feature_selection(X, y, rankings, model=None, max_features=None,
                                  n_splits=3, random_state=SEED):
    """Cross-validation with a precomputed ranking (not a leakage-free nested CV)."""
    X = pd.DataFrame(X)
    y = np.asarray(y, dtype=int)
    if model is None:
        model = SVC(probability=True, random_state=random_state)
    max_features = min(max_features or len(X.columns), len(X.columns))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []
    for n in range(1, max_features + 1):
        row = {"Num_Features": n}
        for method in rankings.columns:
            cols = rankings[method].head(n).tolist()
            values = []
            for tr, va in cv.split(X, y):
                fitted = clone(model).fit(X.iloc[tr][cols], y[tr])
                values.append(roc_auc_score(y[va], _positive_scores(fitted, X.iloc[va][cols])))
            row[method] = float(np.mean(values))
        records.append(row)
    return pd.DataFrame(records)


def tune_svm(X, y, ranking, n_features, n_splits=3, random_state=SEED):
    """Tune C on the training data only; rank should also be derived from training data."""
    X = pd.DataFrame(X)
    y = np.asarray(y, dtype=int)
    selected = list(ranking[:n_features])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        SVC(kernel="rbf", gamma="scale", probability=True, random_state=random_state),
        param_distributions={"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
        n_iter=7, cv=cv, scoring="roc_auc", random_state=random_state, n_jobs=1,
    )
    search.fit(X[selected], y)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_)


def evaluate_model(model, X, y):
    """AUC from continuous scores; classification metrics from the model's prediction rule."""
    y = np.asarray(y, dtype=int)
    scores = _positive_scores(model, X)
    prediction = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    metrics = {"AUC": roc_auc_score(y, scores),
               "Accuracy": accuracy_score(y, prediction),
               "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
               "Specificity": tn / (tn + fp) if tn + fp else np.nan,
               "Balanced_Accuracy": balanced_accuracy_score(y, prediction)}
    return metrics, scores


def bootstrap_auc_ci(y, scores, n_bootstrap=2000, random_state=SEED):
    """Stratified percentile confidence interval for an already-fitted model's AUC."""
    y, scores = np.asarray(y, dtype=int), np.asarray(scores, dtype=float)
    rng = np.random.default_rng(random_state)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("Both classes are needed")
    aucs = []
    for _ in range(n_bootstrap):
        idx = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def plot_roc(y, scores, output=None):
    fpr, tpr, _ = roc_curve(y, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y, scores):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False)
    fig.tight_layout()
    if output:
        fig.savefig(output, bbox_inches="tight")
    return fig


def run_analysis(training, testing, output_dir, target="Label", n_features=145,
                 compare_all=True, include_cnn=True):
    """Execute the analysis using user-supplied training and test DataFrames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, y_train = split_xy(training, target)
    X_test, y_test = split_xy(testing, target)
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test features must have identical order")
    if n_features > X_train.shape[1]:
        raise ValueError("n_features exceeds input dimension")
    if compare_all:
        summary, folds = compare_models(X_train, y_train, include_cnn=include_cnn)
        summary.to_csv(output_dir / "model_comparison.csv", index=False)
        folds.to_csv(output_dir / "model_comparison_folds.csv", index=False)
    ranking = rank_features(X_train, y_train)
    ranking.to_csv(output_dir / "feature_ranking.csv", index=False)
    curve = incremental_feature_selection(X_train, y_train, ranking, max_features=n_features)
    curve.to_csv(output_dir / "feature_selection_curve.csv", index=False)
    selected = ranking["MIC"].head(n_features).tolist()
    model, params, cv_auc = tune_svm(X_train, y_train, selected, n_features)
    train_metrics, train_scores = evaluate_model(model, X_train[selected], y_train)
    test_metrics, test_scores = evaluate_model(model, X_test[selected], y_test)
    table = pd.DataFrame([{"Dataset": "Training", **train_metrics}, {"Dataset": "Test", **test_metrics}])
    table.to_csv(output_dir / "final_performance.csv", index=False)
    pd.DataFrame({"Feature": selected}).to_csv(output_dir / "selected_features.csv", index=False)
    plot_roc(y_test, test_scores, output_dir / "test_roc.pdf")
    plt.close("all")
    return {"performance": table, "best_params": params, "cv_auc": cv_auc,
            "test_auc_ci": bootstrap_auc_ci(y_test, test_scores)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run binary feature selection and model evaluation")
    parser.add_argument("--train", required=True, help="Training CSV with a binary outcome column")
    parser.add_argument("--test", required=True, help="Test CSV with the same features and outcome")
    parser.add_argument("--output", default="results")
    parser.add_argument("--target", default="Label")
    parser.add_argument("--features", type=int, default=145)
    parser.add_argument("--skip-comparison", action="store_true")
    args = parser.parse_args()
    result = run_analysis(
        pd.read_csv(args.train), pd.read_csv(args.test), args.output,
        target=args.target, n_features=args.features,
        compare_all=not args.skip_comparison,
    )
    print(result["performance"].to_string(index=False))
    print("Best SVM parameters:", result["best_params"])
    print("Test AUC 95% CI:", result["test_auc_ci"])
