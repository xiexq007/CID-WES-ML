# Machine learning analysis of genetic variants

Python implementation of the model comparison, feature ranking, incremental feature selection, SVM optimization, and model evaluation procedures used in a binary variant-classification analysis.

## Methods

The model comparison includes **KNN, Naive Bayes, logistic regression (LR), SVM, XGBoost, random forest (RF), stacking, multilayer perceptron (MLP), and a one-dimensional CNN**. Performance is evaluated by stratified cross-validation with ROC AUC and average precision (AUPRC) calculated from continuous prediction scores.

Feature rankings include **MIC, ReliefF, random forest importance, and chi-square**. An incremental-feature-selection curve can be computed using each ranking. An RBF-SVM is then fitted using a selected feature count and cross-validated hyperparameter tuning.

## Installation

Use Python 3.10+ and install the packages in `requirements.txt`. The `minepy` package may require a compatible build environment. PyTorch runs on CPU by default; GPU setup is optional.

```bash
pip install -r requirements.txt
```

## Input format

Prepare separate training and test CSV files. Each row represents an observation, each feature column contains a numeric value, and `Label` contains a binary outcome coded `0` or `1`. Feature columns must be identical and in the same order in both files; data must contain no missing values. The script does not include any datasets.

## Run

```bash
python machine_learning.py --train path/to/train.csv --test path/to/test.csv --output results --features 145
```

The `--features` value is configurable. `--skip-comparison` skips the more computationally intensive nine-model comparison. The main functions can also be imported independently from `machine_learning.py`.

Outputs include model comparison results, ranked features, incremental feature-selection results, selected features, fitted-model performance, and a test ROC plot.

## Reproducibility notes

Random seeds are set in the model constructors and cross-validation splitters. The MIC ranking preserves the original pandas descending sorting behavior for equal MIC scores. With tied scores, the selected feature identities can depend on the input column order and software versions.

Feature ranking followed by cross-validation on the same development cohort can produce optimistic cross-validation estimates. For an unbiased estimate of a full feature-selection workflow, feature ranking and tuning must be repeated inside each training fold of an outer validation loop. The final test data must not be used to choose the model or feature count.

No individual-level data, trained models, or sample identifiers are distributed in this repository.
