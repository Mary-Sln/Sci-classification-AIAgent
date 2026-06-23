import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.sparse import linalg, diags
from scipy.spatial import ConvexHull
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Path Configurations
SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SKILLS_DIR)
SPECTRUMS_DIR = os.path.join(BASE_DIR, "spectrums")
RES_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)

# --- Mathematical Baseline Correction Functions ---

def baseline_als(y, lam=1e5, p=0.01, niter=10):
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L)).tocsr()
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L)).tocsr()
        Z = W + lam * D.T @ D
        z = linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def baseline_airpls(y, lam=1e5, niter=20):
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L)).tocsr()
    w = np.ones(L)
    for i in range(niter):
        W = diags(w, 0, shape=(L, L)).tocsr()
        Z = W + lam * D.T @ D
        z = linalg.spsolve(Z, w * y)
        d = y - z
        d_neg = d[d < 0]
        if len(d_neg) == 0:
            break
        w = np.exp(i * np.abs(d) / np.abs(d_neg).sum())
        w[d >= 0] = 0
    return z

def baseline_arpls(y, lam=1e5, ratio=0.01, niter=20):
    L = len(y)
    D = diags([1, -2, 1], [0, 1, 2], shape=(L-2, L)).tocsr()
    w = np.ones(L)
    for _ in range(niter):
        W = diags(w, 0, shape=(L, L)).tocsr()
        Z = W + lam * D.T @ D
        z = linalg.spsolve(Z, w * y)
        d = y - z
        w = 1.0 / (1.0 + np.exp(2 * (d - d[d<0].mean()) / (d[d<0].std() + 1e-8)))
        w[d >= 0] = ratio
    return z

def baseline_rubberband(y):
    L = len(y)
    x = np.arange(L)
    points = np.vstack((x, y)).T
    augmented = np.vstack([points, [0, max(y)*2], [L-1, max(y)*2]])
    hull = ConvexHull(augmented)
    vertices = sorted([v for v in hull.vertices if v < L])
    return np.interp(x, x[vertices], y[vertices])

def baseline_polynomial(y, order=3):
    x = np.arange(len(y))
    poly_coeffs = np.polyfit(x, y, order)
    return np.polyval(poly_coeffs, x)

# --- Downstream Classification Evaluation Loop ---

def evaluate_baseline_performance(X, y_labels):
    unique_classes, class_counts = np.unique(y_labels, return_counts=True)
    n_splits = 5 if np.min(class_counts) >= 5 else int(np.min(class_counts))

    if n_splits < 2:
        print(f"    [!] WARNING: smallest class has only {np.min(class_counts)} sample(s) -- "
              f"cannot do honest CV. Falling back to fit-and-evaluate-on-train, which is "
              f"OPTIMISTIC and not a real out-of-sample estimate. Treat this score with caution.")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1500, random_state=42)
        model.fit(X_scaled, y_labels)
        y_pred = model.predict(X_scaled)
        acc = accuracy_score(y_labels, y_pred)
        return acc, acc, acc, acc, acc

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s, sens, specs, precs = [], [], [], [], []

    for train_idx, test_idx in skf.split(X, y_labels):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_labels[train_idx], y_labels[test_idx]

        # Scale strictly within the fold to avoid leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1500, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

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


def load_spectra_data():
    csv_files = glob.glob(os.path.join(SPECTRUMS_DIR, "*.csv"))
    if not csv_files:
        print(f"[-] Error: No CSV files found in {SPECTRUMS_DIR}")
        sys.exit(1)

    csv_files = sorted(csv_files, key=lambda x: os.path.basename(x))

    def get_label(filename):
        base = os.path.basename(filename)
        if base.startswith("Ad"):
            return "Adulteration"
        elif base.startswith("A"):
            return "Arabica"
        elif base.startswith("F"):
            return "Festtags"
        elif base.startswith("G"):
            return "Guatemala"
        elif base.startswith("L"):
            return "Lavazza"
        elif base.startswith("S"):
            return "Sumatra"
        else:
            raise ValueError(f"Unknown prefix for filename: {filename}")

    # --- Load the first file to establish the reference frequency grid ---
    first_df = pd.read_csv(csv_files[0])
    if first_df.shape[1] != 2:
        print(f"[-] Error: {csv_files[0]} has {first_df.shape[1]} columns, expected 2 "
              f"(frequency, intensity).")
        sys.exit(1)
    reference_frequencies = first_df.iloc[:, 0].values
    reference_length = len(reference_frequencies)

    global_spectra = []
    labels = []
    sample_ids = []
    mismatched_files = []

    for f in csv_files:
        df = pd.read_csv(f)

        # --- Validate column count ---
        if df.shape[1] != 2:
            mismatched_files.append((os.path.basename(f), f"{df.shape[1]} columns (expected 2)"))
            continue

        # --- Validate frequency grid consistency against the reference ---
        this_frequencies = df.iloc[:, 0].values
        if len(this_frequencies) != reference_length:
            mismatched_files.append(
                (os.path.basename(f), f"{len(this_frequencies)} points (expected {reference_length})"))
            continue
        if not np.allclose(this_frequencies, reference_frequencies, rtol=1e-3):
            mismatched_files.append((os.path.basename(f), "frequency grid differs from reference"))
            continue

        global_spectra.append(df.iloc[:, 1].values)
        labels.append(get_label(f))
        sample_ids.append(os.path.splitext(os.path.basename(f))[0])

    # --- Stop if anything didn't match, rather than silently dropping samples ---
    if mismatched_files:
        print(f"\n[-] Error: {len(mismatched_files)} file(s) failed validation and were "
              f"NOT included. Fix these before proceeding:")
        for fname, reason in mismatched_files:
            print(f"      - {fname}: {reason}")
        print("\n[-] Stopping. Re-run after resolving the files above, or confirm with the "
              "user whether it's acceptable to exclude them.")
        sys.exit(1)

    X = np.array(global_spectra, dtype=float)
    y = np.array(labels)
    sample_ids = np.array(sample_ids)

    return X, y, reference_frequencies, sample_ids


def main():
    print("[+] Baseline Correction Agent Started...")
    X_raw, y_labels, frequencies, sample_ids = load_spectra_data()
    print(f"[+] Loaded {len(X_raw)} spectral profiles with {X_raw.shape[1]} wavelength channels.")

    unique_classes, class_counts = np.unique(y_labels, return_counts=True)
    print(f"[+] Group counts: {dict(zip(unique_classes, class_counts))}")

    # NOTE: variance filtering is computed on the FULL dataset before any CV
    # split. This is a deliberate, low-risk simplification (it uses no label
    # information -- it only removes near-constant/dead channels) but is
    # technically still a mild form of leakage since test folds influence
    # which channels are kept. Documented here rather than silently assumed.
    variances = np.var(X_raw, axis=0)
    keep_mask = variances >= 1e-8
    removed_count = len(keep_mask) - np.sum(keep_mask)
    if removed_count > 0:
        print(f"[*] Variance Filtering: Stripped {removed_count} near-constant channels (Variance < 1e-8).")
        X_raw = X_raw[:, keep_mask]
        frequencies = frequencies[keep_mask]

    np.save(os.path.join(RES_DIR, "X_raw.npy"), X_raw)
    np.save(os.path.join(RES_DIR, "frequencies.npy"), frequencies)
    np.save(os.path.join(RES_DIR, "sample_ids.npy"), sample_ids)

    methods = ["No correction", "ALS", "airPLS", "arPLS", "Rubberband", "Polynomial baseline"]
    comparison_results = []
    corrected_matrices = {}

    for method in methods:
        print(f"[->] Running Method: {method}")
        X_corr = np.zeros_like(X_raw)
        total_samples = len(X_raw)

        for i in range(total_samples):
            sys.stdout.write(f"\r    Processing Spectrum: {i+1}/{total_samples}")
            sys.stdout.flush()

            if method == "No correction":
                X_corr[i] = X_raw[i]
            elif method == "ALS":
                X_corr[i] = X_raw[i] - baseline_als(X_raw[i])
            elif method == "airPLS":
                X_corr[i] = X_raw[i] - baseline_airpls(X_raw[i])
            elif method == "arPLS":
                X_corr[i] = X_raw[i] - baseline_arpls(X_raw[i])
            elif method == "Rubberband":
                X_corr[i] = X_raw[i] - baseline_rubberband(X_raw[i])
            elif method == "Polynomial baseline":
                X_corr[i] = X_raw[i] - baseline_polynomial(X_raw[i], order=3)

        print("\n    Evaluating classification score...")
        corrected_matrices[method] = X_corr
        acc, f1, sen, spec, prec = evaluate_baseline_performance(X_corr, y_labels)
        score = (0.40 * acc) + (0.25 * f1) + (0.15 * sen) + (0.15 * spec) + (0.05 * prec)

        comparison_results.append({
            "Method": method, "Accuracy": acc, "F1": f1,
            "Sensitivity": sen, "Specificity": spec, "Precision": prec, "Composite_Score": score
        })

    df_comp = pd.DataFrame(comparison_results).sort_values(by="Composite_Score", ascending=False)
    df_comp.to_csv(os.path.join(RES_DIR, "baseline_comparison.csv"), index=False)

    best_method = df_comp.iloc[0]["Method"]
    print(f"\n[+] Evaluation Done! Supreme Winner: {best_method} | Score: {df_comp.iloc[0]['Composite_Score']:.4f}")
    print(df_comp.to_string(index=False))

    np.save(os.path.join(RES_DIR, "X_preprocessed.npy"), corrected_matrices[best_method])
    np.save(os.path.join(RES_DIR, "y_labels.npy"), y_labels)

    # Save best baseline name for next stages
    with open(os.path.join(RES_DIR, "best_baseline_method.txt"), "w") as f:
        f.write(best_method)


if __name__ == "__main__":
    main()
