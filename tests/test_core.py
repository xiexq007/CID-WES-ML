import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from insomnia_ml import make_svm, evaluate, bootstrap_auc_ci, incremental_feature_selection, compare_models


def dataset():
    a, y = make_classification(n_samples=80, n_features=12, n_informative=6,
                               n_redundant=1, random_state=42)
    return pd.DataFrame(a, columns=[f'v{i}' for i in range(a.shape[1])]), y


def test_svm_evaluation_and_bootstrap():
    X, y = dataset()
    a, b = train_test_split(np.arange(len(y)), stratify=y, test_size=.25, random_state=42)
    model = make_svm().fit(X.iloc[a], y[a])
    out = evaluate(model, X.iloc[b], y[b])
    assert 0 <= out['AUC'] <= 1
    assert sum(out[k] for k in ['TN', 'FP', 'FN', 'TP']) == len(b)
    ci = bootstrap_auc_ci(y[b], model.decision_function(X.iloc[b]), n_bootstrap=40)
    assert 0 <= ci[0] <= ci[1] <= 1


def test_ifs_and_model_compare():
    X, y = dataset()
    result = incremental_feature_selection(X, y, list(X.columns), [2, 5], folds=3)
    assert result['N_features'].tolist() == [2, 5]
    result = compare_models(X, y, models={'SVM': make_svm()}, folds=3)
    assert result['Model'].tolist() == ['SVM']


def test_validate_bad_labels():
    from insomnia_ml.core import validate_xy
    X, y = dataset()
    try:
        validate_xy(X, np.full(len(y), 1))
    except ValueError:
        pass
    else:
        raise AssertionError('Expected invalid-label failure')


def test_mic_ranking_with_mock_mine(monkeypatch):
    import sys
    import types
    from insomnia_ml import mic_ranking

    class FakeMINE:
        def __init__(self, alpha, c):
            assert alpha == 0.6 and c == 15
        def compute_score(self, feature, labels):
            self.value = float(np.mean(feature))
        def mic(self):
            return self.value

    monkeypatch.setitem(sys.modules, 'minepy', types.SimpleNamespace(MINE=FakeMINE))
    X = pd.DataFrame({'b': [1, 1, 0, 1], 'a': [0, 0, 0, 0], 'c': [1, 0, 1, 1]})
    y = np.array([0, 1, 0, 1])
    rank = mic_ranking(X, y, legacy_ties=False)
    assert rank['Feature'].tolist() == ['b', 'c', 'a']
