"""Utilities for binary variant-based classification.

Input is a numeric feature table (samples x variants) and binary labels.
No participant data or project-specific paths are included.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.svm import SVC

SEED = 42


def validate_xy(X, y):
    X = pd.DataFrame(X).copy()
    if X.columns.has_duplicates or X.index.has_duplicates:
        raise ValueError('Duplicate feature names or sample indices')
    if X.shape[1] < 1 or X.shape[0] < 2:
        raise ValueError('Empty feature matrix')
    X = X.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('Missing or infinite values: handle these within training folds')
    y = np.asarray(y).ravel()
    if len(y) != len(X) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected matching binary labels encoded as 0 and 1')
    return X, y.astype(int)


def mic_ranking(X, y, *, alpha=0.6, c=15, legacy_ties=True):
    """MIC ranking using minepy. legacy_ties=True matches the source notebook's pandas sort.

    MIC ties can be extensive with sparse binary features. The legacy ordering is
    dependent on column order, library versions and pandas sorting behavior.
    """
    try:
        from minepy import MINE
    except ImportError as exc:
        raise ImportError('Install minepy to compute MIC ranking') from exc
    X, y = validate_xy(X, y)
    names, scores = [], []
    for name in X.columns:
        mine = MINE(alpha=alpha, c=c)
        mine.compute_score(X[name].to_numpy(), y)
        names.append(name)
        scores.append(float(mine.mic()))
    ranking = pd.DataFrame({'Feature': names, 'MIC_score': scores})
    if legacy_ties:
        # Exact original notebook approach; no secondary tie-breaking key.
        return ranking.sort_values('MIC_score', ascending=False).reset_index(drop=True)
    # Documented, portable tie-break rule for new analyses.
    return ranking.sort_values(['MIC_score', 'Feature'], ascending=[False, True], kind='mergesort').reset_index(drop=True)


def rank_features(X, y, method='MIC', legacy_ties=True, random_state=SEED):
    """Rank features. For historical MIC results use method='MIC', legacy_ties=True."""
    X, y = validate_xy(X, y)
    if method == 'MIC':
        return mic_ranking(X, y, legacy_ties=legacy_ties)
    if method == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=random_state).fit(X, y)
        scores = model.feature_importances_
    elif method == 'Chi2':
        from sklearn.feature_selection import chi2
        scores = chi2(X - X.min(axis=0), y)[0]
        scores = np.nan_to_num(scores, nan=0.0)
    elif method == 'Relief':
        try:
            from skrebate import ReliefF
        except ImportError as exc:
            raise ImportError('Install skrebate to use ReliefF') from exc
        model = ReliefF(n_neighbors=10, n_features_to_select=X.shape[1]).fit(X.to_numpy(), y)
        scores = model.feature_importances_
    else:
        raise ValueError('method must be MIC, RF, Chi2 or Relief')
    return pd.DataFrame({'Feature': X.columns, 'Score': scores}).sort_values('Score', ascending=False).reset_index(drop=True)


def make_svm(*, C=1.0, gamma='scale', probability=True, random_state=SEED):
    return SVC(C=C, kernel='rbf', gamma=gamma, probability=probability, random_state=random_state)


def compare_models(X, y, models=None, *, folds=3, random_state=SEED):
    """Mean fold-wise AUC and average precision, with continuous scores.

    No data-driven feature screening happens inside this function; pass an
    unscreened or externally fixed matrix, or use a fold-wise pipeline.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    X, y = validate_xy(X, y)
    if models is None:
        models = {
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'LDA': LinearDiscriminantAnalysis(),
            'LR': LogisticRegression(max_iter=2000, random_state=random_state),
            'SVM': make_svm(random_state=random_state),
            'RF': RandomForestClassifier(random_state=random_state),
        }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    rows = []
    for name, model in models.items():
        res = cross_validate(clone(model), X, y, cv=cv,
                             scoring={'AUC': 'roc_auc', 'AUPRC': 'average_precision'},
                             error_score='raise')
        rows.append({'Model': name, 'AUC_mean': np.mean(res['test_AUC']),
                     'AUC_sd': np.std(res['test_AUC'], ddof=1),
                     'AUPRC_mean': np.mean(res['test_AUPRC'])})
    return pd.DataFrame(rows).sort_values('AUC_mean', ascending=False).reset_index(drop=True)


def incremental_feature_selection(X, y, ordered_features, sizes, *, model=None,
                                  folds=3, random_state=SEED):
    """Historical fixed-ranking IFS; CV scores can be optimistic when ranking used y.

    This reproduces the *logic*, not an unbiased estimate. If the ranking was
    learned from the same cohort, use nested CV with fold-wise re-ranking.
    """
    X, y = validate_xy(X, y)
    if model is None:
        model = make_svm(random_state=random_state)
    ordering = list(ordered_features)
    if len(ordering) != len(set(ordering)) or not set(ordering).issubset(X.columns):
        raise ValueError('Invalid feature ranking')
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    rows = []
    for k in sizes:
        if not 1 <= k <= len(ordering):
            raise ValueError(f'Invalid feature count: {k}')
        result = cross_validate(clone(model), X[ordering[:k]], y, cv=cv,
                                scoring='roc_auc', error_score='raise')
        rows.append({'N_features': k, 'CV_AUC_mean': result['test_score'].mean(),
                     'CV_AUC_sd': result['test_score'].std(ddof=1)})
    return pd.DataFrame(rows)


def tune_svm(X, y, *, param_distributions=None, folds=3,
             n_iter=10, random_state=SEED):
    """Tune an SVM on provided training data only."""
    X, y = validate_xy(X, y)
    if param_distributions is None:
        param_distributions = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(make_svm(random_state=random_state),
                                param_distributions=param_distributions,
                                n_iter=n_iter, cv=cv, scoring='roc_auc',
                                random_state=random_state, n_jobs=1,
                                error_score='raise')
    search.fit(X, y)
    return search


def evaluate(model, X, y):
    """AUC from continuous scores; classification metrics from model.predict.

    No threshold is optimized on evaluation labels.
    """
    X, y = validate_xy(X, y)
    scores = model.decision_function(X) if hasattr(model, 'decision_function') else model.predict_proba(X)[:, 1]
    predictions = np.asarray(model.predict(X))
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {'AUC': float(roc_auc_score(y, scores)),
            'Accuracy': float((tp + tn) / len(y)),
            'Sensitivity': float(tp / (tp + fn)),
            'Specificity': float(tn / (tn + fp)),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}


def bootstrap_auc_ci(y, scores, *, n_bootstrap=2000, random_state=SEED):
    y = np.asarray(y).ravel()
    scores = np.asarray(scores).ravel()
    if len(y) != len(scores) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected matched scores and binary labels')
    rng = np.random.default_rng(random_state)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aucs = []
    for _ in range(n_bootstrap):
        idx = np.r_[rng.choice(pos, len(pos), replace=True),
                    rng.choice(neg, len(neg), replace=True)]
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    return np.percentile(aucs, [2.5, 97.5])
