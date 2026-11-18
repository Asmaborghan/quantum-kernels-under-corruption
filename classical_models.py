"""Classical baselines and the shared evaluation settings.

Every model is tuned by an inner 3-fold cross-validation on ROC-AUC; the grids below are
what the inner loop may choose from. The same 5 seeds x 5 outer folds are used for every
model, quantum and classical, so all fold scores are paired.
"""
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SEEDS = [0, 1, 2, 3, 4]
SCORING = {"accuracy": "accuracy", "f1": "f1_macro", "auc": "roc_auc"}   # macro-F1: both classes count
GAMMAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]


def get_model(name, seed=0):
    """A model and the hyperparameter grid its inner loop is allowed to tune."""
    if name == "svm_rbf":
        return SVC(kernel="rbf", class_weight="balanced"), {"gamma": GAMMAS, "C": [0.1, 1, 10]}
    if name == "random_forest":
        return (RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1),
                {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 5]})
    if name == "logistic_regression":
        return LogisticRegression(class_weight="balanced", max_iter=1000), {"C": [0.01, 0.1, 1, 10]}
    raise ValueError(f"Unknown model: {name}")
