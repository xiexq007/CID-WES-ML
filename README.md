# Variant-based machine learning analysis

Python functions for the feature-ranking and SVM classification analyses described in a study of rare variants and chronic insomnia disorder. **No participant-level data, genotype matrices, credentials, paths or fitted models are distributed.**

## Scope

`src/insomnia_ml/core.py` includes MIC, random forest, ReliefF and chi-squared rankings; model comparison; fixed-ranking incremental feature selection (IFS); SVM parameter tuning; and evaluation. The original notebook also explored XGBoost, MLP, CNN and ensemble classifiers. Those exploratory implementations have **not** been included here, so this reduced repository does **not** reproduce the full original nine-model comparison.

The historical MIC ordering was generated using `MINE(alpha=0.6, c=15)` followed by `pandas.DataFrame.sort_values('MIC_score', ascending=False)` without a secondary sorting key. **This tie behavior is important:** many sparse variants have identical MIC scores. The option `legacy_ties=True` preserves that algorithm, but exact rank reproduction can depend on package versions and input feature order. For new work, `legacy_ties=False` provides a documented alphabetical tie-break rule; it should not be described as identical to the original selection.

## Set up

Use Python 3.10. The versions documented for the historical analysis were pandas 2.2.2, NumPy 1.23.5, scikit-learn 1.5.0 and minepy 1.2.6. These are **historical environment records**, not a tested, fully pinned installation lockfile. Install the needed packages in a compatible environment; minepy may require a working compiler toolchain.

```bash
python -m pip install -e '.[test]'
python -m pip install minepy==1.2.6 skrebate
pytest -q
```

## Input format

Supply a pandas DataFrame `X` indexed by sample and with one numeric column per variant. Supply labels `y` encoded as `0` (control) and `1` (CID). The columns must be kept in the same order to reproduce historical tied MIC ranks. Divide training and testing data **before** computing data-dependent feature scores. This repository does not include the sample IDs or exact original train/test assignment.

```python
from insomnia_ml import mic_ranking, make_svm, evaluate

# X_train, X_test: sample-by-variant DataFrames
# y_train, y_test: integer arrays of 0/1 labels
ranking = mic_ranking(X_train, y_train, legacy_ties=True)
selected = ranking['Feature'].head(145).tolist()
model = make_svm(C=1.0, gamma='scale', probability=True, random_state=42)
model.fit(X_train[selected], y_train)
metrics = evaluate(model, X_test[selected], y_test)
print(metrics)
```

The `evaluate` function calculates ROC AUC using continuous decision scores and classification metrics from `predict()`. Historical scripts used predicted probabilities for ROC and chose classification thresholds using evaluation labels; those threshold-derived classification metrics should **not** be treated as untouched test-set performance. This public implementation avoids choosing thresholds on evaluation labels.

## How to interpret validation

The `incremental_feature_selection` utility evaluates a **fixed ranking** over cross-validation folds. If that ranking was generated from all labels in the same training cohort, the fold-wise AUC can be optimistic because validation labels helped to rank the features. An unbiased assessment of the selection *procedure* requires ranking and model/parameter selection anew within training folds (preferably with outer CV). A fixed-panel CV is not equivalent to such an assessment.

## Reproducibility limits

This repository is a cleaned collection of reusable analysis functions and does **not** permit independent numerical replication of the published AUC or the 145 SNP set without the exact input matrix, sample split, full preprocessing and original fitted-model configuration. Sharing scripts alone is insufficient for full reproducibility. Variant calling, upstream gene prioritization, detailed missing-genotype handling, and any external validation are outside this repository.

## Tests

`pytest -q` checks feature-score ordering for a mock MIC implementation, the public SVM workflow, numerical bootstrap bounds, and that evaluation does not optimize a threshold on test labels. Tests use synthetic data only.

## License

No license is supplied. Add a license only if all code owners agree to the intended reuse conditions.
