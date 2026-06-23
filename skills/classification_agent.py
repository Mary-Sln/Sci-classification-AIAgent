"""
Classification Agent
Evaluates SVM, Logistic Regression, SIMCA, PLS-DA, LDA, and Random Forest
on the selected feature matrix, via fold-safe Stratified 5-Fold CV.

IMPLEMENTATION NOTES (read before trusting SIMCA/PLS-DA results):

- SIMCA: scikit-learn has no native SIMCA classifier. This implementation
  fits one PCA model PER CLASS using only that class's training samples,
  then assigns each test sample to the class whose PCA model gives the
  LOWEST reconstruction error (Q-residual). This is a simplified, "hard
  assignment" variant of SIMCA used for benchmark comparison against other
  discriminant classifiers. True SIMCA is a one-class modelling method that
  can also reject a sample as belonging to NO class or to MULTIPLE classes
  -- that nuance is lost here. Treat SIMCA's leaderboard score as a fair
  comparison point, not as the full SIMCA methodology.

- PLS-DA: implemented via PLSRegression regressing onto a one-hot encoded
  target, then assigning the class via argmax of predicted scores. This is
  the standard way to do PLS-DA when a dedicated implementation isn't
  available.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder

SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SKILLS_DIR)
RES_DIR = os.path.join(BASE_DIR, "results")


# --- Custom classifiers (no native sklearn equivalent) ---

class SimcaClassifier:
    """Per-class PCA models; assigns the class with lowest reconstruction
    (Q-residual) error. See module docstring for caveats."""

    def __init__(self, max_components=5):
        self.max_components = max_components
        self.class_models = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            X_cls = X[y == cls]
            # Cap components by available samples/features in this class
            n_comp = min(self.max_components, X_cls.shape[0] - 1, X_cls.shape[1])
            n_comp = max(n_comp, 1)
            pca = PCA(n_components=n_comp, random_state=42)
            pca.fit(X_cls)
            self.class_models[cls] = pca
        return self

    def predict(self, X):
        residuals = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            pca = self.class_models[cls]
            X_proj = pca.transform(X)
            X_recon = pca.inverse_transform(X_proj)
            residuals[:, i] = np.sum((X - X_recon) ** 2, axis=1)
        best_idx = np.argmin(residuals, axis=1)
        return self.classes_[best_idx]


class PlsDaClassifier:
    """PLS Regression onto one-hot targets; predicts class via argmax."""

    def __init__(self, n_components=10):
        self.n_components = n_components
        self.encoder = OneHotEncoder(sparse_output=False)
        self.pls = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        n_comp = max(n_comp, 1)
        self.pls = PLSRegression(n_components=n_comp)
        y_onehot = self.encoder.fit_transform(y.reshape(-1, 1))
        self.pls.fit(X, y_onehot)
        return self

    def predict(self, X):
        y_pred_scores = self.pls.predict(X)
        best_idx = np.argmax(y_pred_scores, axis=1)
        # Map back from one-hot column order to actual class labels
        category_order = self.encoder.categories_[0]
        return category_order[best_idx]


def build_models():
    """Returns dict of name -> (model_factory, needs_scaling).
    model_factory is a zero-arg callable so each fold gets a fresh model."""
    return {
        "SVM (RBF)": (lambda: SVC(kernel='rbf', random_state=42), True),
        "Logistic Regression": (lambda: LogisticRegression(max_iter=1500, random_state=42), True),
        "SIMCA": (lambda: SimcaClassifier(max_components=5), True),
        "PLS-DA": (lambda: PlsDaClassifier(n_components=10), True),
        "LDA": (lambda: LinearDiscriminantAnalysis(), True),
        "Random Forest": (lambda: RandomForestClassifier(n_estimators=200, random_state=42), False),
    }


def evaluate_model(X, y, model_factory, needs_scaling):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_splits = 5 if np.min(class_counts) >= 5 else int(np.min(class_counts))
    if n_splits < 2:
        print(f"    [!] WARNING: smallest class has only {np.min(class_counts)} sample(s). "
              f"Cannot do honest CV.")
        return None, None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, sens, specs, precs = [], [], [], [], []
    last_cm = None
    last_labels = None

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if needs_scaling:
            scaler = StandardScaler()
            X_train_use = scaler.fit_transform(X_train)
            X_test_use = scaler.transform(X_test)
        else:
            X_train_use, X_test_use = X_train, X_test

        model = model_factory()
        model.fit(X_train_use, y_train)
        y_pred = model.predict(X_test_use)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='macro'))
        sens.append(recall_score(y_test, y_pred, average='macro'))
        precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))

        labels_sorted = np.unique(y)
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        specs.append(np.mean(tn / (tn + fp + 1e-8)))

        last_cm = cm  # keep most recent fold's confusion matrix for inspection
        last_labels = labels_sorted

    metrics = (np.mean(accs), np.mean(f1s), np.mean(sens), np.mean(specs), np.mean(precs))
    return metrics, (last_cm, last_labels)


def main():
    print("[+] Classification Agent Initiated...")

    X_path = os.path.join(RES_DIR, "X_selected_features.npy")
    y_path = os.path.join(RES_DIR, "y_labels.npy")

    for p in [X_path, y_path]:
        if not os.path.exists(p):
            print(f"[-] Missing: {p}. Please run feature_selection_agent.py first.")
            sys.exit(1)

    X = np.load(X_path)
    y = np.load(y_path)
    print(f"[+] Loaded selected feature matrix: {X.shape} | Labels: {len(y)}")

    models = build_models()
    results = []
    confusion_matrices = {}

    print("\n[->] Evaluating classifiers (fold-safe scaling, Stratified CV)...")
    for name, (factory, needs_scaling) in models.items():
        print(f"    Evaluating: {name}")
        metrics, cm_info = evaluate_model(X, y, factory, needs_scaling)
        if metrics is None:
            continue
        acc, f1, sen, spec, prec = metrics
        score = (0.40 * acc) + (0.25 * f1) + (0.15 * sen) + (0.15 * spec) + (0.05 * prec)
        results.append({
            "Model": name, "Accuracy": acc, "F1_Macro": f1,
            "Sensitivity": sen, "Specificity": spec,
            "Precision": prec, "Composite_Score": score
        })
        if cm_info is not None:
            confusion_matrices[name] = cm_info

    df_results = pd.DataFrame(results).sort_values(by="Composite_Score", ascending=False)
    df_results.to_csv(os.path.join(RES_DIR, "classification_results.csv"), index=False)

    best_model = df_results.iloc[0]["Model"]
    best_score = df_results.iloc[0]["Composite_Score"]

    print(f"\n[+] Classification Complete!")
    print(f"    Best Model: {best_model} | Score: {best_score:.4f}")
    print(df_results.to_string(index=False))

    # Save confusion matrix for the best model as a labeled CSV
    if best_model in confusion_matrices:
        cm, labels = confusion_matrices[best_model]
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(os.path.join(RES_DIR, f"confusion_matrix_{best_model.replace(' ', '_')}.csv"))
        print(f"\n[*] Confusion matrix for best model saved to results/"
              f"confusion_matrix_{best_model.replace(' ', '_')}.csv")
        print(f"    (NOTE: this is from the LAST CV fold only -- useful for a quick look, "
              f"not a full out-of-sample summary)")

    with open(os.path.join(RES_DIR, "best_classifier.txt"), "w") as f:
        f.write(best_model)


if __name__ == "__main__":
    main()