"""
Feature Selection Agent
Tests K in [30, 50, 100, 150, 200, 300, 500] using SelectKBest (f_classif),
performed STRICTLY inside each CV fold to avoid leakage (same principle
applied to MSC and baseline correction earlier in the pipeline).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SKILLS_DIR)
RES_DIR = os.path.join(BASE_DIR, "results")

K_CANDIDATES = [30, 50, 100, 150, 200, 300, 500]


def evaluate_k(X, y, k):
    """Stratified 5-Fold CV evaluation. SelectKBest is fit on the TRAINING
    fold only, then applied to both train and test -- this is the fold-safe
    pattern. Using f_classif (ANOVA F-test) which only requires labels +
    features, computed independently per fold."""
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_splits = 5 if np.min(class_counts) >= 5 else int(np.min(class_counts))
    if n_splits < 2:
        print(f"    [!] WARNING: smallest class has only {np.min(class_counts)} sample(s). "
              f"Skipping K={k} -- cannot do honest CV.")
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, sens, specs, precs = [], [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- Fold-safe feature selection: fit on train fold ONLY ---
        selector = SelectKBest(score_func=f_classif, k=min(k, X_train.shape[1]))
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sel)
        X_test_s = scaler.transform(X_test_sel)

        model = LogisticRegression(max_iter=1500, random_state=42)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='macro'))
        sens.append(recall_score(y_test, y_pred, average='macro'))
        precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        specs.append(np.mean(tn / (tn + fp + 1e-8)))

    return np.mean(accs), np.mean(f1s), np.mean(sens), np.mean(specs), np.mean(precs)


def main():
    print("[+] Feature Selection Agent Initiated...")

    X_path = os.path.join(RES_DIR, "X_final_chemometrics.npy")
    y_path = os.path.join(RES_DIR, "y_labels.npy")
    fr_path = os.path.join(RES_DIR, "frequencies.npy")

    for p in [X_path, y_path]:
        if not os.path.exists(p):
            print(f"[-] Missing: {p}. Please run preprocessing_agent.py first.")
            sys.exit(1)

    X = np.load(X_path)
    y = np.load(y_path)
    frequencies = np.load(fr_path) if os.path.exists(fr_path) else np.arange(X.shape[1])

    print(f"[+] Loaded preprocessed matrix: {X.shape} | Labels: {len(y)}")

    # Don't test K values larger than the available number of channels
    valid_k = [k for k in K_CANDIDATES if k <= X.shape[1]]
    if len(valid_k) < len(K_CANDIDATES):
        skipped = set(K_CANDIDATES) - set(valid_k)
        print(f"[*] Skipping K values larger than available channels ({X.shape[1]}): {skipped}")

    results = []
    print("\n[->] Screening K values (fold-safe SelectKBest)...")
    for k in valid_k:
        print(f"    Evaluating K={k}")
        out = evaluate_k(X, y, k)
        if out is None:
            continue
        acc, f1, sen, spec, prec = out
        score = (0.40 * acc) + (0.25 * f1) + (0.15 * sen) + (0.15 * spec) + (0.05 * prec)
        results.append({
            "K": k, "Accuracy": acc, "F1_Macro": f1,
            "Sensitivity": sen, "Specificity": spec,
            "Precision": prec, "Optimization_Score": score
        })

    if not results:
        print("[-] No valid K evaluations completed. Check class sizes.")
        sys.exit(1)

    df_results = pd.DataFrame(results).sort_values(by="Optimization_Score", ascending=False)
    df_results.to_csv(os.path.join(RES_DIR, "feature_selection_results.csv"), index=False)

    best_k = int(df_results.iloc[0]["K"])
    best_score = df_results.iloc[0]["Optimization_Score"]

    print(f"\n[+] Feature Selection Complete!")
    print(f"    Best K: {best_k} | Score: {best_score:.4f}")
    print(df_results.to_string(index=False))

    # --- Refit SelectKBest on the FULL dataset using the winning K ---
    # This is standard practice after a method/hyperparameter is chosen via
    # fold-safe CV -- it is not used for scoring, so it is not leakage.
    final_selector = SelectKBest(score_func=f_classif, k=best_k)
    X_selected = final_selector.fit_transform(X, y)
    selected_mask = final_selector.get_support()
    selected_frequencies = frequencies[selected_mask]

    np.save(os.path.join(RES_DIR, "X_selected_features.npy"), X_selected)
    np.save(os.path.join(RES_DIR, "selected_feature_mask.npy"), selected_mask)
    np.save(os.path.join(RES_DIR, "selected_frequencies.npy"), selected_frequencies)
    with open(os.path.join(RES_DIR, "best_k.txt"), "w") as f:
        f.write(str(best_k))

    print(f"[*] Selected matrix saved to results/X_selected_features.npy ({X_selected.shape})")
    print(f"[*] Selected wavenumbers saved to results/selected_frequencies.npy")
    print(f"    (Useful later for interpreting WHICH spectral regions discriminate the classes)")


if __name__ == "__main__":
    main()